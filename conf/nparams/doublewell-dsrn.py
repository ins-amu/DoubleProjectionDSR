

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
    model: "dsrn"
    n_states:   8
    n_guiding:  null
    stoch_vars: null
    n_obs: 1
    n_samples:  4
    olv: null
    glv: null
    n_ignore: 50
    guiding_encoder:
        n_channels:  24
        n_levels:     7
        kernel_size:  7
    causal_encoder:
        n_channels: 24
        n_levels:    7
        kernel_size: 7
        causal: 1        
    noise_encoder:
        n_channels: 24
        n_levels:    7
        kernel_size: 7
        rnn_size: 32
    f:
        arch: 'mlptanh'
        hidden_size: null
        alpha: 1.
        slv: -8.
        slv_trainable: True
    g:
        type: 'nonlinear'
        readout: 'det'
        n_hidden: 32

    lambda_x: 0.01
    lambda_g: 0.3
    lambda_z: 0.0


training:
    seed: 42
    n_iter: 30000
    learning_rate: 0.001
    forcing_interval: null
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

nseeds = 4

variants = {
    'v1': {('model', 'n_guiding'): 7, ('model', 'stoch_vars'): 1,
           ('model', 'olv'): -2., ('model', 'glv'): 0., ('training', 'forcing_interval'): 200},
    'v3': {('model', 'n_guiding'): 7, ('model', 'stoch_vars'): 0,
           ('model', 'olv'): -2., ('model', 'glv'): 0., ('training', 'forcing_interval'): 60},
}


params = [
    {
        ('model', 'f', 'hidden_size'): nf
    }
    for nf in [16, 32, 64, 128, 256, 512, 1024, 2048]
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
