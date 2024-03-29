import ml_collections


def get_config() -> ml_collections.ConfigDict:

    config = ml_collections.ConfigDict()

    # data configuration
    config.data = ml_collections.ConfigDict()

    config.data.ntrain = 2048
    config.data.nvalidation = 256
    config.data.tau = 2

    # simulation configuration
    config.simulation = ml_collections.ConfigDict()

    config.simulation.nk = 35
    config.simulation.dt = 5e-3
    config.simulation.re = 42.0
    config.simulation.nu = 2

    # experiment configuration
    config.experiment = ml_collections.ConfigDict()

    config.experiment.nk = 35
    config.experiment.nx_lr = 10
    config.experiment.nx_hr = 70

    config.experiment.sr_factor = ml_collections.ConfigDict()
    config.experiment.sr_factor.upsample = 7
    config.experiment.sr_factor.downsample = 7

    config.experiment.noise_std = 0.0

    # training configuration
    config.training = ml_collections.ConfigDict()

    config.training.n_epochs = 1000
    config.training.batch_size = 32
    config.training.lr = 3e-4
    config.training.lr_gamma = 0.9
    config.training.l2 = 0.0
    config.training.lambda_weight = 1e-4
    config.training.fwt_lb = 1.0

    return config
