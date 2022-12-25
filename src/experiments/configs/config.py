import ml_collections

from src.experiments.configs import common, experiment_config


def get_config(exp_case: str) -> ml_collections.ConfigDict:

    exp, case = exp_case.split('@')

    config = common.get_config()
    get_experiment_config_diff = getattr(experiment_config, f'get_{exp}_config')

    config_diff = get_experiment_config_diff(case)
    config.update_from_flattened_dict(config_diff)

    return config
