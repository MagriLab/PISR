from ml_collections import config_dict


BASE_CONFIG = config_dict.ConfigDict({
    'DATA': {
        'NTRAIN': 2056,
        'NVALIDATION': 256,
        'TAU': 2,
    },
    'SIMULATION': {
        'NK': 30,
        'DT': 0.005,
        'RE': 42.0,
        'NU': 2,
    },
    'EXPERIMENT': {
        'NX_LR': 10,
        'SR_FACTOR': 15,
    },
    'TRAINING': {
        'N_EPOCHS': 1000,
        'BATCH_SIZE': 128,
        'LR': 0.0003,
        'L2': 0.0,
        'LAMBDA': 1e6,
        'FWT_LB': 1.0,
    }
})


WANDB_CONFIG = config_dict.ConfigDict({
    'entity': config_dict.placeholder(str),
    'project': config_dict.placeholder(str),
    'group': config_dict.placeholder(str),
    'job_type': config_dict.placeholder(str),
    'notes': config_dict.placeholder(str)
})
