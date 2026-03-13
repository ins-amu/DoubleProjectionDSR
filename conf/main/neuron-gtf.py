tool = "gtf"

default = {
    "experiment": "main",
    "name": "neuron",
    "path_to_data": None,
    "path_to_inputs": "",
    "run": 1,
    "device": "cpu",

    "model": "clippedShallowPLRNN",
    "observation_model": "Identity",
    "latent_dim": None,
    "num_bases": 50,
    "hidden_dim": None,

    "use_gtf": True,
    "gtf_alpha": None,
    "gtf_alpha_method": "constant",
    "sequence_length": 200,
    "batch_size": 16,
    "lat_model_regularization": 0.0,
    "obs_model_regularization": 1e-7,
    "partial_forcing": True,

    "epochs": 5000,
    "batches_per_epoch": 50,
    "start_lr": 1e-3,
    "end_lr": 1e-6,
    "scalar_saving_interval": 1000,
    "image_saving_interval":  1000,

    "D_stsp_bins": 30,
    "D_stsp_scaling": 1.0,
    "PSE_smoothing": 20.0,
    "PE_n": 20,

    "MAR_ratio": 0.0,
    "MAR_lambda": 0.0,
    "optimizer": "RADAM",
    "gradient_clipping_norm": 0.0,
    "teacher_forcing_interval": 16,
    "gtf_alpha_decay": 0.999,
    "alpha_update_interval": 5,
    "gaussian_noise_level": 0.00
}

nseeds = 16

variants = {
    'v1': {"path_to_data": "data/neuron/neuron.v1.train.npy", "observation_model": "Identity", "latent_dim": 8, "hidden_dim": 256},
    'v2': {"path_to_data": "data/neuron/neuron.v2.train.npy", "observation_model": "Affine",   "latent_dim": 8, "hidden_dim": 256},
}

params = [
    {
        'gtf_alpha': alpha
    }
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
]

_default_evaluation = {
    'variant': 'default',
    'variables': None,
    'n_embed': 300,
    'max_nt': 40000,
    'pe_n': [10, 20, 50, 100],
    'pw_c': [0.1, 0.25, 0.5, 1.0],
    'pw_maxn': 300,
    'max_eval': 2000,
    'n_samples': 20,
}

evaluation = {
    'v1': _default_evaluation | {'variables': [0]},
    'v2': _default_evaluation | {'variables': [7]},
}

