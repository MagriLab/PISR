from typing import Any, TypeAlias


FlattenedDict: TypeAlias = dict[str, Any]


def get_e01_config(case: str) -> FlattenedDict:

    """Experiment 01 -- Effect of SR Factor, k in {7, 9, 11, 13, 15}"""

    sr_factor = {
        'C1': 7,
        'C2': 9,
        'C3': 11,
        'C4': 13,
        'C5': 15,
    }[case]

    config_diff = {
        'experiment.sr_factor.upsample': sr_factor,
        'experiment.sr_factor.downsample': sr_factor
    }

    return config_diff


def get_e02_config(case: str) -> FlattenedDict:

    """Experiment 02 -- Effect of FWT, fwt_lb in {0.001, 0.01, 0.1, 0.5, 1.0}"""

    fwt_lb = {
        'C1': 0.001,
        'C2': 0.01,
        'C3': 0.1,
        'C4': 0.5,
        'C5': 1.0
    }[case]

    config_diff = {'training.fwt_lb': fwt_lb}

    return config_diff


def get_e03_config(case: str) -> FlattenedDict:

    """Experiment 03 -- Effect of lambda weight, lambda in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}"""

    lambda_weight = {
        'C1': 1e-6,
        'C2': 1e-5,
        'C3': 1e-4,
        'C4': 1e-3,
        'C5': 1e-2,
    }[case]

    config_diff = {'training.lambda_weight': lambda_weight}

    return config_diff


def get_e04_config(case: str) -> FlattenedDict:

    """Experiment 04 -- Effect of learning rate, eta in {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}"""

    learning_rate = {
        'C1': 1e-6,
        'C2': 1e-5,
        'C3': 1e-4,
        'C4': 1e-3,
        'C5': 1e-2,
    }[case]

    config_diff = {'training.lr': learning_rate}

    return config_diff


def get_e05_config(case: str) -> FlattenedDict:

    """Experiment 05 -- Effect of batch size, bs in {32, 64, 128, 256, 512}"""

    batch_size = {
        'C1': 32,
        'C2': 64,
        'C3': 128,
        'C4': 256,
        'C5': 512,
    }[case]

    config_diff = {'training.batch_size': batch_size}

    return config_diff


def get_e06_config(case: str) -> FlattenedDict:

    """Experiment 06 -- Effect of white-noise, std in {1e-4, 1e-3, 1e-2, 1e-1, 1e0}"""

    std = {
        'C1': 1e-4,
        'C2': 1e-3,
        'C3': 1e-2,
        'C4': 1e-1,
        'C5': 1.0,
    }[case]

    config_diff = {'experiment.noise_std': std}

    return config_diff


def get_e07_config(case: str) -> FlattenedDict:

    """Experiment 07-- Effect of data with noise."""

    std = 1e-3

    ntrain = {
        'C1': 2 ** 10,
        'C2': 2 ** 11,
        'C3': 2 ** 12,
        'C4': 2 ** 13,
        'C5': 2 ** 14,
    }[case]

    config_diff = {
        'experiment.noise_std': std,
        'data.ntrain': ntrain,
    }

    return config_diff


def get_e08_config(case: str) -> FlattenedDict:

    """Experiment 08 -- Effet of learning rate schedule, gamma in {1.0, 0.99, 0.95, 0.9, 0.85}"""

    gamma = {
        'C1': 1.0,
        'C2': 0.99,
        'C3': 0.95,
        'C4': 0.90,
        'C5': 0.85,
    }[case]

    config_diff = {'training.lr_gamma': gamma}

    return config_diff


def get_e09_config(case: str) -> FlattenedDict:

    """Experiment 09 -- Effet of tau in {2, 3, 5, 7, 9}"""

    tau = {
        'C1': 2,
        'C2': 3,
        'C3': 5,
        'C4': 7,
        'C5': 9,
    }[case]

    # should not hard code this...
    product = 1890

    ntrain = int(product / tau)
    nvalidation = 256

    config_diff = {
        'data.ntrain': ntrain,
        'data.nvalidation': nvalidation,
        'data.tau': tau
    }

    return config_diff


def get_e10_config(case: str) -> FlattenedDict:

    """Experiment 10 -- Effet of tau in {2, 3, 5, 7, 9} - fixed ntrain."""

    tau = {
        'C1': 2,
        'C2': 3,
        'C3': 5,
        'C4': 7,
        'C5': 9,
    }[case]

    config_diff = {'data.tau': tau}

    return config_diff
