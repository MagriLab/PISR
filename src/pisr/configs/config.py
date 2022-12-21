import ml_collections
from ml_collections import config_dict


def get_config(sr_factor: str) -> ml_collections.ConfigDict:

    config = base_config()

    # make sure string is of the right format
    if not sr_factor.endswith('x'):
        raise ValueError('Must provide sr_factor as the following: "7x".')

    _factor = int(sr_factor[:-1])

    if _factor % 2 == 0:
        raise ValueError('Must provide odd super-resolution factor.')

    config.experiment.sr_factor = _factor

    return config


def base_config() -> ml_collections.ConfigDict:

    config = ml_collections.ConfigDict()

    # data configuration
    config.data = ml_collections.ConfigDict()

    config.data.ntrain = 2048
    config.data.nvalidation = 256
    config.data.tau = 2

    # simulation configuration
    config.simulation = ml_collections.ConfigDict()

    config.simulation.nk = 30
    config.simulation.dt = 5e-3
    config.simulation.re = 42.0
    config.simulation.nu = 2

    # experiment configuration
    config.experiment = ml_collections.ConfigDict()

    config.experiment.nx_lr = 10
    config.experiment.sr_factor = config_dict.placeholder(int)

    # training configuration
    config.training = ml_collections.ConfigDict()

    config.training.n_epochs = 1000
    config.training.batch_size = 128
    config.training.lr = 3e-4
    config.training.l2 = 0.0
    config.training.lambda_weight = 1e6
    config.training.fwt_lb = 1.0

    return config
