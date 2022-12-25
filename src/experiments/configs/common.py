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

    config.simulation.nk = 30
    config.simulation.dt = 5e-3
    config.simulation.re = 42.0
    config.simulation.nu = 2

    # experiment configuration
    config.experiment = ml_collections.ConfigDict()

    config.experiment.nx_lr = 10
    config.experiment.sr_factor = 7

    # training configuration
    config.training = ml_collections.ConfigDict()

    config.training.n_epochs = 1000
    config.training.batch_size = 128
    config.training.lr = 3e-4
    config.training.l2 = 0.0
    config.training.lambda_weight = 1e-6
    config.training.fwt_lb = 1.0

    return config
