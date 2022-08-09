import argparse
import csv
import functools as ft

import sys
from pathlib import Path
from shutil import copyfile
from typing import Callable, Dict, NamedTuple, Optional, Union

import einops
import numpy as np
import opt_einsum as oe
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

sys.path.append('../..')
from pisr.model import SuperResolution
from pisr.loss import KolmogorovLoss
from pisr.sampling import get_low_res_grid

from pisr.data import load_data, train_validation_split, generate_dataloader

from pisr.utils.config import ExperimentConfig
from pisr.utils.loss_tracker import LossTracker

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# machine constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == 'cuda' else {}


class WandbConfig(NamedTuple):

    entity: Optional[str]
    project: Optional[str]
    group: Optional[str]


def initialise_csv(csv_path: Path) -> None:

    """Initialise the results .csv file.

    Parameters
    ----------
    csv_path: Path
        Path for the .csv file to create and write header to.
    """

    lt = LossTracker()
    with open(csv_path, 'w+', newline='') as f_out:
        writer = csv.writer(f_out, delimiter=',')
        writer.writerow(['epoch', *lt.get_fields(training=True), *lt.get_fields(training=False)])


def initialise_wandb(wandb_config: WandbConfig,
                     config: ExperimentConfig,
                     experiment_path: Path,
                     log_code: bool) -> Union[Run, RunDisabled, None]:

    """Initialise the Weights and Biases API.

    Parameters
    ----------
    wandb_config: WandbConfig
        Arguments for Weights and Biases.
    config: ExperimentConfig
        Configuration used to run the experiment.
    experiment_path: Path
        Path for the experiment -- location of code to copy
    log_code: bool
        Whether to save a copy of the code to Weights and Biases.
    """

    wandb_run = None
    if wandb_config.entity:

        # initialise W&B API
        wandb_run = wandb.init(
            config=config.config,
            entity=wandb_config.entity,
            project=wandb_config.project,
            group=wandb_config.group,
            name=str(experiment_path)
        )

    # log current code state to W&B
    if log_code and isinstance(wandb_run, Run):
        wandb_run.log_code(str(Path.cwd()))

    return wandb_run


def initialise_model(config: ExperimentConfig, model_path: Optional[Path] = None) -> nn.Module:

    """Iniitalise CNN Model for experiment.

    Parameters
    ----------
    config: ExperimentConfig
        Parameters to use for the experiment.
    model_path: Optional[Path]
        Optional model to load.

    Returns
    -------
    model: nn.Module
        Initialised model.
    """

    # get activation function
    activation_fn = getattr(nn, config.ACTIVATION)()

    # initialise model
    model = SuperResolution()

    # load model from file if applicable.
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.to(torch.float)
    model.to(DEVICE)

    return model


def train_loop(model: nn.Module,
               dataloader: DataLoader,
               factor: int,
               loss_fn: KolmogorovLoss,
               optimizer: Optional[torch.optim.Optimizer] = None,
               s_lambda: float = 1e3,
               set_train: bool = False) -> LossTracker:

    """Run a single training / evaluation loop.

    Parameters
    ----------
    model: nn.Module
        Model to use for evaluation.
    dataloader: DataLoader
        Generator to retrieve data for evaluation from.
    factor: int
        Factor by which to super-resolve the data.
    loss_fn: KolmogorovLoss
        Loss function for calculating physics-informed aspect of the loss.
    optimizer: Optional[torch.optim.Optimizer]
        Optimiser used to update the weights of the model, when applicable.
    s_lambda: float
        Scaling parameter for the loss.
    set_train: bool
        Determine whether run in training / evaluation mode.

    Returns
    -------
    LossTracker
        Loss tracking object to hold information about the training progress.    
    """

    # data-driven loss
    batched_sensor_loss = (0.0 + 0.0j)
    batched_l2_sensor_loss = (0.0 + 0.0j)

    # physics-based loss
    batched_momentum_loss = (0.0 + 0.0j)
    batched_continuity_loss = (0.0 + 0.0j)
    
    # l2 loss against actual data
    batched_l2_actual_loss = (0.0 + 0.0j)

    model.train(mode=set_train)
    for hi_res in dataloader:

        lo_res = get_low_res_grid(hi_res, factor=factor)

        # move tensors to device
        lo_res = lo_res.to(DEVICE)
        hi_res = hi_res.to(DEVICE)

        pred_hi_res = model(lo_res)

        # LOSS :: 01 :: Sensor Locations
        pred_sensor_measurements = get_low_res_grid(pred_hi_res, factor=factor)
        mag_sensor_err = oe.contract('... -> ', (pred_sensor_measurements - lo_res) ** 2) 

        sensor_loss = mag_sensor_err / lo_res.numel()
        l2_sensor_loss = torch.sqrt(mag_sensor_err / oe.contract('... -> ', lo_res ** 2))

        # LOSS :: 02 :: Momentum Loss
        momentum_loss = loss_fn.calc_residual_loss(pred_hi_res)

        # LOSS :: 03 :: Continuity Loss
        continuity_loss = torch.zeros_like(momentum_loss)
        if loss_fn.constraints:
            continuity_loss = loss_fn.calc_constraint_loss(pred_hi_res)

        # LOSS :: 04 :: Actual Loss
        l2_actual_loss = torch.sqrt(oe.contract('... -> ', (hi_res - pred_hi_res) ** 2) / oe.contract('... -> ', hi_res ** 2))

        # LOSS :: 05 :: Total Loss
        total_loss = s_lambda * sensor_loss + momentum_loss + loss_fn.constants * continuity_loss

        # update gradients
        if set_train and optimizer:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    # normalise and find absolute value
    batched_sensor_loss /= len(dataloader.dataset)                                                       # type: ignore
    batched_l2_sensor_loss /= len(dataloader.dataset)                                                    # type: ignore
    batched_momentum_loss /= len(dataloader.dataset)                                                     # type: ignore
    batched_continuity_loss /= len(dataloader.dataset)                                                   # type: ignore
    batched_l2_actual_loss /= len(dataloader.dataset)                                                    # type: ignore

    loss_dict: Dict[str, float] = {
        'sensor_loss': batched_sensor_loss,
        'l2_sensor_loss': batched_l2_sensor_loss,
        'momentum_loss': batched_momentum_loss,
        'continuity_loss': batched_continuity_loss,
        'l2_actual_loss': batched_l2_actual_loss
    }

    return LossTracker(**loss_dict)


def main(args: argparse.Namespace) -> None:

    """Run the Experiment.

    Parameters
    ----------
    args: argparse.Namespace
        Command-line arguments to dictate experiment run.
    """

    if args.run_gpu is not None and args.run_gpu >= 0 and args.run_gpu < torch.cuda.device_count():

        global DEVICE
        global DEVICE_KWARGS

        if not torch.cuda.is_available():
            raise ValueError('Specified CUDA device unavailable.')

        DEVICE = torch.device(f'cuda:{args.run_gpu}')
        DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True}

    if args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, DEVICE)

    # load yaml configuration file
    config = ExperimentConfig()
    config.load_config(args.config_path)

    # initialise weights and biases
    wandb_config = WandbConfig(entity=args.wandb_entity, project=args.wandb_project, group=args.wandb_group)
    wandb_run = initialise_wandb(wandb_config, config, args.experiment_path, log_code=True)

    # setup the experiment path and copy config file
    args.experiment_path.mkdir(parents=True, exist_ok=True)
    copyfile(args.config_path, args.experiment_path / 'config.yml')

    # initialise csv
    csv_path = args.experiment_path / 'results.csv'
    initialise_csv(csv_path)

    # load data
    u_all: torch.Tensor = load_data(h5_file=args.data_path, config=config).to(torch.float)
    train_u, validation_u = train_validation_split(u_all, config.NTRAIN, config.NVALIDATION, step=config.TIME_STACK)

    train_loader: DataLoader = generate_dataloader(train_u, config.BATCH_SIZE, DEVICE_KWARGS)
    validation_loader: DataLoader = generate_dataloader(validation_u, config.BATCH_SIZE, DEVICE_KWARGS)

    # TODO >> Fill this in...
    loss_fn: KolmogorovLoss()

    # empirical observations suggest this works better without constraints
    loss_fn.constraints = False

    # initialise model / optimizer
    model = initialise_model(config, args.model_path)
    model.to(torch.float)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.L2)

    # generate training functions
    _loop_params = dict(model=model, loss_fn=loss_fn, simulation_dt=config.DT)

    train_fn = ft.partial(train_loop, **_loop_params, dataloader=train_loader, optimizer=optimizer, set_train=True)
    validation_fn = ft.partial(train_loop, **_loop_params, dataloader=validation_loader, set_train=False)

    # main training loop
    min_validation_loss = np.Inf
    for epoch in range(config.N_EPOCHS):

        lt_training: LossTracker = train_fn()
        lt_validation: LossTracker = validation_fn()

        # update global validation loss if model improves
        if lt_validation.total_loss < min_validation_loss:
            min_validation_loss = lt_validation.total_loss
            torch.save(model.state_dict(), args.experiment_path / 'autoencoder.pt')

        # log results to weights and biases
        if isinstance(wandb_run, Run):
            wandb_log = {**lt_training.get_dict(training=True), **lt_validation.get_dict(training=False)}
            wandb_run.log(data=wandb_log)

        # print update to stdout
        msg = f'Epoch: {epoch:05}'
        for k, v in lt_training.get_dict(training=True).items():
            msg += f' | {k}: {v:08.5e}'

        if epoch % args.log_freq == 0:
            print(msg)

        # write new results to .csv file
        _results = [epoch, *lt_training.get_loss_keys, *lt_validation.get_loss_keys]
        with open(csv_path, 'a', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)

    # upload results to weights and biases
    if isinstance(wandb_run, Run):
        artifact = wandb.Artifact(name=str(args.experiment_path).replace('/', '.'), type='dataset')
        artifact.add_dir(local_path=str(args.experiment_path))

        wandb_run.log_artifact(artifact_or_path=artifact)


if __name__ == '__main__':

    # read arguments from command line
    parser = argparse.ArgumentParser(description='PISR :: Physics-Informed Super-Resolution Experiment.')

    # arguments to define paths for experiment run
    parser.add_argument('-ep', '--experiment-path', type=Path, required=True)
    parser.add_argument('-dp', '--data-path', type=Path, required=True)
    parser.add_argument('-cp', '--config-path', type=Path, required=True)

    # argument to define optional path to load pre-trained model
    parser.add_argument('-mp', '--model-path', type=Path, required=False)

    parser.add_argument('-gpu', '--run-gpu', type=int, required=False)
    parser.add_argument('-mf', '--memory-fraction', type=float, required=False)

    parser.add_argument('-lf', '--log-freq', type=int, required=False, default=5)

    # arguments to define wandb parameters
    parser.add_argument('--wandb-entity', default=None, type=str)
    parser.add_argument('--wandb-project', default=None, type=str)
    parser.add_argument('--wandb-group', default=None, type=str)

    parsed_args = parser.parse_args()

    main(parsed_args)

