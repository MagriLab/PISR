"""TODO

## Weights and Biases
- Better checks for whether to log.
- Force online / offline syncing.
- Tag management.
- Log images of results.

## Writing FLAGS to file.

"""

import csv
import functools as ft
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import opt_einsum as oe
import torch
import wandb
from absl import app, flags
from ml_collections import config_flags
from torch import nn
from torch.utils.data import DataLoader
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run


sys.path.append('../..')

import warnings

from pisr.config import BASE_CONFIG, WANDB_CONFIG
from pisr.data import generate_dataloader, load_data, train_validation_split
from pisr.experimental import define_path as pisr_flags
from pisr.loss import KolmogorovLoss
from pisr.model import SRCNN
from pisr.sampling import get_low_res_grid
from pisr.utils.loss_tracker import LossTracker


warnings.filterwarnings('ignore', category=UserWarning)



FLAGS = flags.FLAGS

_CONFIG = config_flags.DEFINE_config_dict('config', BASE_CONFIG)
_WANBD_CONFIG = config_flags.DEFINE_config_dict('wandb', WANDB_CONFIG)

_EXPERIMENT_PATH = pisr_flags.DEFINE_path(
    'experiment_path',
    None,
    'Directory to store experiment results'
)

_DATA_PATH = pisr_flags.DEFINE_path(
    'data_path',
    None,
    'Path to .h5 file storing the data.'
)

_GPU = flags.DEFINE_integer(
    'run_gpu',
    0,
    'Which GPU to run on.'
)

_MEMORY_FRACTION = flags.DEFINE_float(
    'memory_fraction',
    None,
    'Memory fraction of GPU to use.'
)

_LOG_FREQUENCY = flags.DEFINE_integer(
    'log_frequency',
    1,
    'Frequency at which to log results.'
)

_CUDNN_BENCHMARKS = flags.DEFINE_boolean(
    'cudnn_benchmarks',
    True,
    'Whether to use CUDNN benchmarks or not.'
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True} if DEVICE == 'cuda' else {}


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


def initialise_wandb() -> Optional[Run | RunDisabled]:

    """Initialise the Weights and Biases API."""

    wandb_run = None
    if FLAGS.wandb.entity:

        # initialise W&B API
        wandb_run = wandb.init(
            config=FLAGS.config.to_dict(),
            entity=FLAGS.wandb.entity,
            project=FLAGS.wandb.project,
            group=FLAGS.wandb.group,
            name=str(FLAGS.experiment_path.name)
        )

    # log current code state to W&B
    if wandb_run:
        wandb_run.log_code(str(Path.cwd()))

    return wandb_run


def initialise_model() -> nn.Module:

    """Initalise CNN Model for experiment.

    Returns
    -------
    model: nn.Module
        Initialised model.
    """

    if not FLAGS.config.EXPERIMENT.SR_FACTOR % 2 == 1:
        raise ValueError('SR_FACTOR must be odd for super-resolution.')

    # initialise model
    model = SRCNN(lr_nx=FLAGS.config.EXPERIMENT.NX_LR, upscaling=FLAGS.config.EXPERIMENT.SR_FACTOR, mode='bicubic')

    model.to(torch.float)
    model.to(DEVICE)

    return model


def train_loop(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: KolmogorovLoss,
               optimizer: Optional[torch.optim.Optimizer] = None,
               set_train: bool = False) -> LossTracker:

    """Run a single training / evaluation loop.

    Parameters
    ----------
    model: nn.Module
        Model to use for evaluation.
    dataloader: DataLoader
        Generator to retrieve data for evaluation from.
    loss_fn: KolmogorovLoss
        Loss function for calculating physics-informed aspect of the loss.
    optimizer: Optional[torch.optim.Optimizer]
        Optimiser used to update the weights of the model, when applicable.
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

    batched_total_loss = (0.0 + 0.0j)

    # l2 loss against actual data
    batched_l2_actual_loss = (0.0 + 0.0j)


    model.train(mode=set_train)
    for hi_res in dataloader:

        hi_res = hi_res.to(DEVICE, non_blocking=True)
        lo_res = get_low_res_grid(hi_res, factor=FLAGS.config.EXPERIMENT.SR_FACTOR)

        pred_hi_res = model(lo_res)

        # LOSS :: 01 :: Sensor Locations
        pred_sensor_measurements = get_low_res_grid(pred_hi_res, factor=FLAGS.config.EXPERIMENT.SR_FACTOR)
        mag_sensor_err = oe.contract('... -> ', (pred_sensor_measurements - lo_res) ** 2)

        sensor_loss = mag_sensor_err / lo_res.numel()
        l2_sensor_loss = torch.sqrt(mag_sensor_err / oe.contract('... -> ', lo_res ** 2))                  # type: ignore

        # LOSS :: 02 :: Momentum Loss
        momentum_loss = loss_fn.calc_residual_loss(pred_hi_res)

        # LOSS :: 03 :: Continuity Loss
        continuity_loss = torch.zeros_like(momentum_loss)
        if loss_fn.constraints:
            continuity_loss = loss_fn.calc_constraint_loss(pred_hi_res)

        # LOSS :: 04 :: Actual Loss
        l2_actual_loss = torch.sqrt(oe.contract('... -> ', (hi_res - pred_hi_res) ** 2) / oe.contract('... -> ', hi_res ** 2)) # type: ignore

        # LOSS :: 05 :: Total Loss
        total_loss = FLAGS.config.TRAINING.LAMBDA * sensor_loss + momentum_loss + loss_fn.constraints * continuity_loss

        # update batch losses
        batched_sensor_loss += sensor_loss.item() * hi_res.size(0)
        batched_l2_sensor_loss += l2_sensor_loss.item() * hi_res.size(0)

        batched_momentum_loss += momentum_loss.item() * hi_res.size(0)
        batched_continuity_loss += continuity_loss.item() * hi_res.size(0)

        batched_total_loss += total_loss.item() * hi_res.size(0)

        batched_l2_actual_loss += l2_actual_loss.item() * hi_res.size(0)

        # update gradients
        if set_train and optimizer:

            optimizer.zero_grad(set_to_none=True)

            total_loss.backward()
            optimizer.step()

    # normalise and find absolute value
    batched_sensor_loss = float(abs(batched_sensor_loss)) / len(dataloader.dataset)                      # type: ignore
    batched_l2_sensor_loss = float(abs(batched_l2_sensor_loss)) / len(dataloader.dataset)                # type: ignore
    batched_momentum_loss = float(abs(batched_momentum_loss)) / len(dataloader.dataset)                  # type: ignore
    batched_continuity_loss = float(abs(batched_continuity_loss)) / len(dataloader.dataset)              # type: ignore
    batched_total_loss = float(abs(batched_total_loss)) / len(dataloader.dataset)                        # type: ignore
    batched_l2_actual_loss = float(abs(batched_l2_actual_loss)) / len(dataloader.dataset)                # type: ignore

    loss_dict: dict[str, float] = {
        'sensor_loss': batched_sensor_loss,
        'l2_sensor_loss': batched_l2_sensor_loss,
        'momentum_loss': batched_momentum_loss,
        'continuity_loss': batched_continuity_loss,
        'total_loss': batched_total_loss,
        'l2_actual_loss': batched_l2_actual_loss
    }

    return LossTracker(**loss_dict)


def main(_):

    if FLAGS.run_gpu is not None and FLAGS.run_gpu >= 0 and FLAGS.run_gpu < torch.cuda.device_count():

        global DEVICE
        global DEVICE_KWARGS

        if not torch.cuda.is_available():
            raise ValueError('Specified CUDA device unavailable.')

        DEVICE = torch.device(f'cuda:{FLAGS.run_gpu}')
        DEVICE_KWARGS = {'num_workers': 1, 'pin_memory': True}

    if FLAGS.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(FLAGS.memory_fraction, DEVICE)

    # easy access to config
    config = FLAGS.config

    # initialise weights and biases
    wandb_run = initialise_wandb()

    # setup the experiment path
    FLAGS.experiment_path.mkdir(parents=True, exist_ok=True)

    # save config to yaml
    with open(FLAGS.experiment_path / 'config.yml', 'w') as f:
        config.to_yaml(stream=f)

    # initialise csv
    csv_path = FLAGS.experiment_path / 'results.csv'
    initialise_csv(csv_path)

    # load data
    u_all = load_data(h5_file=FLAGS.data_path, config=config)
    train_u, validation_u = train_validation_split(u_all, config.DATA.NTRAIN, config.DATA.NVALIDATION, step=config.DATA.TAU)

    # set `drop_last = True` if using: `torch.backends.cudnn.benchmark = True`
    dataloader_kwargs = dict(shuffle=True, drop_last=FLAGS.cudnn_benchmarks)

    train_loader = generate_dataloader(train_u, config.TRAINING.BATCH_SIZE, dataloader_kwargs, DEVICE_KWARGS)
    validation_loader = generate_dataloader(validation_u, config.TRAINING.BATCH_SIZE, dataloader_kwargs, DEVICE_KWARGS)

    # define loss function -- disable constraints
    loss_fn = KolmogorovLoss(nk=config.SIMULATION.NK, re=config.SIMULATION.RE, dt=config.SIMULATION.DT, fwt_lb=config.TRAINING.FWT_LB, device=DEVICE)
    loss_fn.constraints = False

    # initialise model / optimizer
    model = initialise_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.LR, weight_decay=config.TRAINING.L2)

    # generate training functions
    _loop_params = dict(model=model, loss_fn=loss_fn)

    train_fn = ft.partial(train_loop, **_loop_params, dataloader=train_loader, optimizer=optimizer, set_train=True)
    validation_fn = ft.partial(train_loop, **_loop_params, dataloader=validation_loader, set_train=False)

    # main training loop
    min_validation_loss = np.Inf
    for epoch in range(config.TRAINING.N_EPOCHS):

        lt_training: LossTracker = train_fn()
        lt_validation: LossTracker = validation_fn()

        # update global validation loss if model improves
        if lt_validation.total_loss < min_validation_loss:
            min_validation_loss = lt_validation.total_loss
            torch.save(model.state_dict(), FLAGS.experiment_path / 'model.pt')

        # log results to weights and biases
        if isinstance(wandb_run, Run):
            wandb_log = {**lt_training.get_dict(training=True), **lt_validation.get_dict(training=False)}
            wandb_run.log(data=wandb_log)

        # print update to stdout
        msg = f'Epoch: {epoch:05}'
        for k, v in lt_training.get_dict(training=True).items():
            msg += f' | {k}: {v:08.5e}'

        if epoch % FLAGS.log_frequency == 0:
            print(msg)

        # write new results to .csv file
        _results = [epoch, *lt_training.get_loss_keys, *lt_validation.get_loss_keys]
        with open(csv_path, 'a', newline='') as f_out:
            writer = csv.writer(f_out, delimiter=',')
            writer.writerow(_results)

    # upload results to weights and biases
    if isinstance(wandb_run, Run):

        artifact_name = str(FLAGS.experiment_path).replace('/', '.')

        # define artifact
        results_artifact = wandb.Artifact(name=artifact_name, type='results')
        results_artifact.add_dir(local_path=str(FLAGS.experiment_path))

        model_artifact = wandb.Artifact(name=artifact_name, type='model')
        model_artifact.add_file(str(FLAGS.experiment_path / 'model.pt'))

        for artifact in (results_artifact, model_artifact):
            wandb_run.log_artifact(artifact_or_path=artifact)


if __name__ == '__main__':
    app.run(main)
