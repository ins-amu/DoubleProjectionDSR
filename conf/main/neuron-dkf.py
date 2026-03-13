

tool = "dsrn"

default = """
variables:
    name: "neuron"
    variant: "0000"

data:
    path: "data/neuron/neuron.npz"
    chunk_size: 300
    batch_size: 16
    subsample: 1
    use_vars: [0]

model:
    model: 'dkf'
    n_states:  8
    n_obs:     1
    n_samples: 4
    olv: None
    n_ignore: 50
    encoder:
        n_channels: 24
        n_levels:    7
        kernel_size: 7
        rnn_size: 32
    causal_encoder:
        n_channels: 24
        n_levels:    7
        kernel_size: 7
        causal: 1
        rnn_size: 32
    f:
        hidden_size: 256
        alpha: 1.
        slv:  -4.
        slv_trainable: True
    g:
        type: 'linear'
    lambda_g: 0.3


training:
    seed: 42
    n_iter: 30000
    learning_rate: 0.001
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
        ('model', 'olv'): olv,
        ('model', 'f', 'slv'): slv,
    }
    for olv in [-4., -2., 0.]
    for slv in [-8., -6., -4., -2.]
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
