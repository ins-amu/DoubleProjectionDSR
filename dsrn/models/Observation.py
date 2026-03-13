import torch.nn as nn

class IdentityObs(nn.Module):
    def __init__(self, vars):
        super().__init__()
        self.vars = vars

    def forward(self, z):
        return z[:,:,self.vars]

    def weights(self):
        return []


def set_readout(n_states, stoch_vars, readout):
    if readout == "all":
        readout_vars = range(0, n_states)
    elif readout == "det":
        readout_vars = [v for v in range(n_states) if v not in stoch_vars]
    elif readout == "stoch":
        readout_vars = stoch_vars
    else:
        readout_vars = readout
    return readout_vars


class LinearObs(nn.Module):
    def __init__(self, n_states, stoch_vars, n_obs, readout):
        super().__init__()
        self.readout_vars = set_readout(n_states, stoch_vars, readout)
        self.g = nn.Linear(len(self.readout_vars), n_obs)

    def forward(self, z):
        return self.g(z[:,:,self.readout_vars])

    def weights(self):
        return [self.g.weight]


class NonlinearObs(nn.Module):
    def __init__(self, n_states, stoch_vars, n_obs, readout, n_hidden):
        super().__init__()
        self.readout_vars = set_readout(n_states, stoch_vars, readout)
        self.g = nn.Sequential(
            nn.Linear(len(self.readout_vars), n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_obs)
        )

    def forward(self, z):
        return self.g(z[:,:,self.readout_vars])

    def weights(self):
        return [self.g[0].weight, self.g[2].weight]