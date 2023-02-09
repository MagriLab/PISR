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

    nx_lr = 10
    nx_hr = nx_lr * sr_factor

    # for e01 \kappa_o = \kappa
    simulation_nk = nx_hr // 2

    config_diff = {
        'simulation.nk': simulation_nk,
        'experiment.nx_lr': nx_lr,
        'experiment.nx_hr': nx_hr,
        'experiment.nk': simulation_nk,
        'experiment.sr_factor.upsample': sr_factor,
        'experiment.sr_factor.downsample': sr_factor
    }

    return config_diff


def get_e02_config(case: str) -> FlattenedDict:

    """Experiment 02 -- varying kappo < kappa_o, kappa in {5, 7, 9, 11, 13}"""

    sr_factor = {
        'C1': 5,
        'C2': 7,
        'C3': 9,
        'C4': 11,
        'C5': 13,
    }[case]

    simulation_nk = 75

    nx_lr = 10
    downample_factor = (2 * simulation_nk) // nx_lr

    nx_hr = nx_lr * sr_factor
    experiment_nk = nx_hr // 2

    config_diff = {
        'simulation.nk': simulation_nk,
        'experiment.nx_lr': nx_lr,
        'experiment.nx_hr': nx_hr,
        'experiment.nk': experiment_nk,
        'experiment.sr_factor.upsample': sr_factor,
        'experiment.sr_factor.downsample': downample_factor
    }

    return config_diff
