

tool = "dsrn"

default = """

variables:
    name: "doublewell"
    variant: "0000"

data:
    path: "data/doublewell/doublewell.npz"
    chunk_size: 300
    batch_size: 16
    subsample: 1
    use_vars: [0]

model:
    model: "rssm"
    nh: 32
    ns: 4
    nobs: 1
    nf: 16
    n_h2s: 16
    n_hs2o: 16
    n_fh2s: 16
    n_f2h0: 16
    n_ignore: 50
    encoder:
        n_channels:  24
        n_levels:     7
        kernel_size:  7
    causal_encoder:
        n_channels:  24
        n_levels:     7
        kernel_size:  7
        causal: 1

training:
    seed: 42
    n_iter: 30000
    learning_rate: 0.001
    gradient_clip: 100.
    lr_milestones: [1000, 10000, 20000]
    lr_gamma: 0.3
    args:
        forcing_interval:
            method: 'linear'
            init_value: 1
            value: null
            t0:  5000
            tmax: 10000
            dtype: 'int'
        overshooting: null

output:
    plot_every: 5000
    save_every: 5000
    plot_direc: ""
    save_direc: ""
    logfile:    ""
"""

nseeds = 16


variants = {
    'v1': {('training', 'args', 'overshooting'): 'none'},
    'v2': {('training', 'args', 'overshooting'): 'observation'},
}


params = {
    'v1': [{
        ('training', 'args', 'forcing_interval', 'value'): 1., 
    }],
    'v2': [{
        ('training', 'args', 'forcing_interval', 'value'): tau, 
    } for tau in [3, 10, 20, 40, 100]]
}


evaluation = {
    'variant': 'default',
    'variables': [0],
    'n_embed': 300,
    'max_nt': 40000,
    'pe_n': [10, 20, 50, 100],
    'pw_c': [0.1, 0.25, 0.5, 1.0],
    'pw_maxn': 300,
    'max_eval': 2000,
    'n_samples': 20,
}

