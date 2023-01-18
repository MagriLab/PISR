import subprocess
import sys
from multiprocessing import Pool, Queue
from pathlib import Path

import torch
from absl import app, flags
from ml_collections import config_flags

from ..pisr.experimental import define_path as pisr_flags
from .configs.wandb import WANDB_CONFIG


FLAGS = flags.FLAGS

_WANDB_CONFIG = config_flags.DEFINE_config_dict('wandb', WANDB_CONFIG)

_BASE_EXPERIMENT_PATH = pisr_flags.DEFINE_path('base_experiment_path', None, 'Base experiment path')
_CONFIG_PATH = pisr_flags.DEFINE_path('config_path', None, 'Config file to use')
_DATA_PATH = pisr_flags.DEFINE_path('data_path', None, 'Path to .h5 file storing the data.')

_EXPERIMENT = flags.DEFINE_string('experiment', None, 'Experiment to run')
_N_REPEATS = flags.DEFINE_integer('n_repeats', 1, 'Number of repeats to run')

flags.mark_flags_as_required(['config_path', 'base_experiment_path', 'data_path'])


NUM_GPUS = torch.cuda.device_count()
PROCS_PER_GPU = 1

queue = Queue()

# TODO :: Make this general -- enumerate each experiment, not all will have 5.
CASE_IDX = (1, 2, 3, 4, 5)


class Job:

    def __init__(self, case_idx: int, experiment_path: Path) -> None:

        self.case_idx = case_idx
        self.experiment_path = experiment_path

    def __str__(self) -> str:
        return f'Job(case_idx={self.case_idx}, experiment_path={self.experiment_path})'


def run_job(job: Job) -> None:

    gpu_id = queue.get()

    try:

        # base command
        cmd = f'{sys.executable} -m src.experiments.base_experiment'

        # required commands
        cmd += f' --config {FLAGS.config_path}:{FLAGS.experiment}@C{job.case_idx}'
        cmd += f' --data_path {FLAGS.data_path}'
        cmd += f' --experiment_path {job.experiment_path}'
        cmd += f' --run_gpu {gpu_id}'

        # add weights and biases information
        for k in filter(lambda _k: wandb_dict[_k], wandb_dict := FLAGS.wandb):             # type: ignore
            cmd += f' --wandb.{k} {wandb_dict[k]}'

        # define logging paths
        stdout_path = job.experiment_path / 'stdout.log'
        stderr_path = job.experiment_path / 'stderr.log'

        # produce path
        job.experiment_path.mkdir(parents=True, exist_ok=False)

        print(f'Starting {job}')
        print(f'>> {cmd}')
        print()

        with open(stdout_path, 'w+') as out, open(stderr_path, 'w+') as err:
            _ = subprocess.run(cmd.split(' '), stdout=out, stderr=err, check=False)

    finally:
        queue.put(gpu_id)


def main(_) -> None:

    # initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROCS_PER_GPU):
            queue.put(gpu_ids)

    # create list of jobs to run
    job_list = []
    for case_idx in CASE_IDX:
        for idx_run in range(FLAGS.n_repeats):
            job_list.append(Job(case_idx, FLAGS.base_experiment_path / f'{case_idx:02}' / f'{idx_run:02}'))

    # multiprocessing to run jobs
    with Pool(processes=PROCS_PER_GPU * NUM_GPUS) as pool:
        pool.map(run_job, job_list)


if __name__ == "__main__":
    app.run(main)
