from typing import Any, Dict

import h5py
import numpy as np
import tqdm
from absl import app, flags
from kolsol.numpy.solver import KolSol

from ..pisr.experimental import define_path as pisr_flags


FLAGS = flags.FLAGS

_DATA_PATH = pisr_flags.DEFINE_path(
    'data_path',
    None,
    'Path to save data to.'
)

_RE = flags.DEFINE_float(
    're',
    42.0,
    'Reynolds number of the flow.'
)

_DT = flags.DEFINE_float(
    'dt',
    5e-3,
    'Time-step for the simulation,'
)

_TIME_SIMULATION = flags.DEFINE_float(
    'time_simulation',
    120.0,
    'Number of seconds to run simulation for.'
)

_TIME_TRANSIENT = flags.DEFINE_float(
    'time_transient',
    180.0,
    'Number of seconds to run transient simulation for.'
)

_NK = flags.DEFINE_integer(
    'nk',
    30,
    'Number of symmetric wavenumbers to solve with.'
)

flags.mark_flags_as_required(['data_path'])


NF: float = 4.0
NDIM: int = 2


def setup_directory() -> None:

    """Sets up the relevant simulation directory."""

    if not FLAGS.data_path.suffix == '.h5':
        raise ValueError('setup_directory() :: Must pass .h5 data_path')

    if FLAGS.data_path.exists():
        raise FileExistsError(f'setup_directory() :: {FLAGS.data_path} already exists.')

    FLAGS.data_path.parent.mkdir(parents=True, exist_ok=True)


def write_h5(data: Dict[str, Any]) -> None:

    """Writes results dictionary to .h5 file.

    Parameters
    ----------
    data: Dict[str, Any]
        Data to write to file.
    """

    with h5py.File(FLAGS.data_path, 'w') as hf:

        for k, v in data.items():
            hf.create_dataset(k, data=v)


def main(_) -> None:

    """Generate Kolmogorov Flow Data."""

    print('00 :: Initialising Kolmogorov Flow Solver.')

    setup_directory()

    cds = KolSol(nk=FLAGS.nk, nf=NF, re=FLAGS.re, ndim=NDIM)                                           # type: ignore
    field_hat = cds.random_field(magnitude=10.0, sigma=1.2)

    # define time-arrays for simulation run
    t_arange = np.arange(0.0, FLAGS.time_simulation, FLAGS.dt)
    transients_arange = np.arange(0.0, FLAGS.time_transient, FLAGS.dt)

    nt = t_arange.shape[0]
    nt_transients = transients_arange.shape[0]

    # setup recording arrays - only need to record fourier field
    velocity_hat_arr = np.zeros(shape=(nt, cds.nk_grid, cds.nk_grid, NDIM), dtype=np.complex128)

    # integrate over transients
    msg = '01 :: Integrating over transients.'
    for _ in tqdm.trange(nt_transients, desc=msg):
        field_hat += FLAGS.dt * cds.dynamics(field_hat)

    # integrate over simulation domain
    msg = '02 :: Integrating over simulation domain'
    for t in tqdm.trange(nt, desc=msg):

        # time integrate
        field_hat += FLAGS.dt * cds.dynamics(field_hat)

        # record metrics
        velocity_hat_arr[t, ...] = field_hat

    data_dict = {
        're': FLAGS.re,
        'dt': FLAGS.dt,
        'nk': FLAGS.nk,
        'time': t_arange,
        'velocity_field_hat': velocity_hat_arr,
    }

    print('02 :: Writing results to file.')
    write_h5(data_dict)

    print('03 :: Simulation Done.')


if __name__ == '__main__':
    app.run(main)
