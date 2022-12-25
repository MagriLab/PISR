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

    config_diff = {'experiment.sr_factor': sr_factor}

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
