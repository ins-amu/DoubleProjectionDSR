

tool = "dsrn"

default = """

variables:
    name: "rnn"
    variant: "0000"

data:
    path: "data/rnn/rnn.npz"
    chunk_size: null
    overlap: -1
    batch_size: 16
    subsample: 1
    use_vars: [0]

model:
    model: "arlstm"
    n_hidden: 32
    n_obs: 1
    n_init: 8
    n_hidden_init: 32
    init_chunk: null
    n_samples:  1
    encoder:
        n_channels:  24
        n_levels:     7
        kernel_size:  7


training:
    seed: 42
    n_iter: 30000
    learning_rate: 0.001
    args:
        gamma:
            method: 'linear'
            value: null
            tmax: 5000
    gradient_clip: 100.
    lr_milestones: [1000, 10000, 20000]
    lr_gamma: 0.3


output:
    plot_every: 5000
    save_every: 5000    
    plot_direc: ""
    save_direc: ""
    logfile:    ""
"""

nseeds = 16

variants = {
    'v1': {},
}


params = [
    {
        ('training', 'args', 'gamma', 'value'): gamma,
        ('data', 'chunk_size'):  300 + predict_len,
        ('model', 'init_chunk'): 300
    }
    for gamma in [0., 0.2, 0.4, 0.6, 0.8, 1.0]
    for predict_len in [20,50,100,200]
]


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
