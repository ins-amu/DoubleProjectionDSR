
from typing import Optional, List

import numpy as np

import torch
import torch.nn as nn
import torch.jit as jit


class PLRNNCell(nn.Module):
    def __init__(self, input_size, state_size):
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size

        self.A = nn.Parameter(0.9 * torch.ones(state_size, dtype=torch.float32))
        self.W = nn.Parameter(torch.zeros(state_size, state_size, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(state_size, dtype=torch.float32))
        nn.init.normal_(self.W, mean=0., std=0.01)

        self.slv = nn.Parameter(-10. * torch.ones(input_size, dtype=torch.float32))

    def forward(self, input, z):
        # input shape: (bs, input_size)
        # z shape: (bs, state_size)

        W = self.W - torch.diag(self.W.diagonal())
        znext = self.A[None,:] * z + torch.matmul(torch.relu(z), W) + self.b
        znext[:,self.state_size - self.input_size:] += torch.exp(0.5 * self.slv) * input

        # TODO: Use the input matrix as in the MLPCell

        return znext


class MLPCell(nn.Module):
    def __init__(self, input_size, state_size, hidden_size, alpha=None, slv=-6.,
                 slv_trainable=False, input_variables=None):
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size

        if input_variables is None:
            input_variables = torch.tensor(range(self.state_size-self.input_size, self.state_size),
                                                dtype=torch.int)
        else:
            input_variables = torch.tensor(input_variables, dtype=torch.int)

        self.Wi = torch.zeros((input_size, state_size))
        self.Wi[torch.arange(0, input_size), input_variables] = 1

        self.W1 = nn.Parameter(torch.zeros((state_size, hidden_size), dtype=torch.float32))
        self.W2 = nn.Parameter(torch.zeros((hidden_size, state_size), dtype=torch.float32))
        self.b1 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.b2 = nn.Parameter(torch.zeros(state_size,  dtype=torch.float32))
        nn.init.normal_(self.W1, mean=0., std=0.01)
        nn.init.normal_(self.W2, mean=0., std=0.01)

        if slv_trainable:
            self.slv = nn.Parameter(slv * torch.ones(input_size, dtype=torch.float32))
        else:
            self.slv = torch.tensor(slv, dtype=torch.float32)

        # Set scaling
        if alpha is not None:
            if not hasattr(alpha, "__len__") or len(alpha) == 1:
                self.alpha = alpha * torch.ones(state_size)
            elif len(alpha) == state_size:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                raise ValueError(f"Size of alpha does not match the state size: {len(alpha)} != {state_size}")
        else:
            self.alpha = torch.ones(state_size)


    def forward(self, input, z):
        # input shape: (bs, input_size)
        # z shape: (bs, state_size)

        z_ = torch.relu(torch.matmul(z, self.W1) + self.b1)
        znext = z + self.alpha * (torch.matmul(z_, self.W2) + self.b2)
        znext = znext + torch.matmul(torch.exp(0.5 * self.slv) * input, self.Wi)

        return znext
    
class MLPTanhCell(nn.Module):
    def __init__(self, input_size, state_size, hidden_size, alpha=None, slv=-6.,
                 slv_trainable=False, input_variables=None):
        super().__init__()

        self.input_size = input_size
        self.state_size = state_size

        if input_variables is None:
            input_variables = torch.tensor(range(self.state_size-self.input_size, self.state_size),
                                                dtype=torch.int)
        else:
            input_variables = torch.tensor(input_variables, dtype=torch.int)

        self.Wi = torch.zeros((input_size, state_size))
        self.Wi[torch.arange(0, input_size), input_variables] = 1

        self.W1 = nn.Parameter(torch.zeros((state_size, hidden_size), dtype=torch.float32))
        self.W2 = nn.Parameter(torch.zeros((hidden_size, state_size), dtype=torch.float32))
        self.b1 = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.b2 = nn.Parameter(torch.zeros(state_size,  dtype=torch.float32))
        nn.init.normal_(self.W1, mean=0., std=0.01)
        nn.init.normal_(self.W2, mean=0., std=0.01)

        if slv_trainable:
            self.slv = nn.Parameter(slv * torch.ones(input_size, dtype=torch.float32))
        else:
            self.slv = torch.tensor(slv, dtype=torch.float32)

        # Set scaling
        if alpha is not None:
            if not hasattr(alpha, "__len__") or len(alpha) == 1:
                self.alpha = alpha * torch.ones(state_size)
            elif len(alpha) == state_size:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                raise ValueError(f"Size of alpha does not match the state size: {len(alpha)} != {state_size}")
        else:
            self.alpha = torch.ones(state_size)


    def forward(self, input, z):
        # input shape: (bs, input_size)
        # z shape: (bs, state_size)

        # MLP
        z_ = torch.relu(torch.matmul(z, self.W1) + self.b1)
        znext = z + self.alpha * (torch.matmul(z_, self.W2) + self.b2)

        # Noise
        znext = znext + torch.matmul(torch.exp(0.5 * self.slv) * input, self.Wi)

        # Tanh
        znext = torch.tanh(znext)

        return znext    


class GRU2Cell(nn.Module):
    def __init__(self, input_size, state_size, hidden_size):
        super().__init__()

        self.Wz  = nn.Parameter(torch.zeros((input_size, state_size)))        
        self.Uz1 = nn.Parameter(torch.zeros((state_size, hidden_size)))
        self.Uz2 = nn.Parameter(torch.zeros((hidden_size, state_size)))
        self.bz1 = nn.Parameter(torch.zeros(hidden_size))
        self.bz2 = nn.Parameter(torch.zeros(state_size))

        self.Wr  = nn.Parameter(torch.zeros((input_size, state_size)))        
        self.Ur1 = nn.Parameter(torch.zeros((state_size, hidden_size)))
        self.Ur2 = nn.Parameter(torch.zeros((hidden_size, state_size)))
        self.br1 = nn.Parameter(torch.zeros(hidden_size))
        self.br2 = nn.Parameter(torch.zeros(state_size))

        self.Wh  = nn.Parameter(torch.zeros((input_size, state_size)))        
        self.Uh1 = nn.Parameter(torch.zeros((state_size, hidden_size)))
        self.Uh2 = nn.Parameter(torch.zeros((hidden_size, state_size)))
        self.bh1 = nn.Parameter(torch.zeros(hidden_size))        
        self.bh2 = nn.Parameter(torch.zeros(state_size))

        nn.init.normal_(self.Wz, mean=0., std=0.01)
        nn.init.normal_(self.Uz1, mean=0., std=0.01)
        nn.init.normal_(self.Uz2, mean=0., std=0.01)

        nn.init.normal_(self.Wr, mean=0., std=0.01)
        nn.init.normal_(self.Ur1, mean=0., std=0.01)
        nn.init.normal_(self.Ur2, mean=0., std=0.01)

        nn.init.normal_(self.Wh, mean=0., std=0.01)
        nn.init.normal_(self.Uh1, mean=0., std=0.01)
        nn.init.normal_(self.Uh2, mean=0., std=0.01)

    def forward(self, input, h):

        hz = torch.matmul(torch.relu(torch.matmul(h, self.Uz1) + self.bz1), self.Uz2) + self.bz2
        z = torch.sigmoid(torch.matmul(input, self.Wz) + hz)

        hh = torch.matmul(torch.relu(torch.matmul(h, self.Uh1) + self.bh1), self.Uh2) + self.bh2
        hhat = torch.tanh(torch.matmul(input, self.Wh) + hh)
            
        hnext = (1-z)*h + z*hhat

        return hnext




class GRUCell(nn.Module):
    def __init__(self, input_size, state_size):
        super().__init__()
        self.c = nn.GRUCell(input_size, state_size)


    def forward(self, input, z):
        # input shape: (bs, input_size)
        # z shape: (bs, state_size)

        znext = self.c(input, z)

        return znext


class RNNForcing(jit.ScriptModule):
    def __init__(self, input_size, state_size, arch, hidden_size: Optional[int]=128,
                 alpha=None, slv=-6., slv_trainable=False, input_variables=None):

        super().__init__()

        self.input_size = input_size
        self.state_size = state_size

        if input_variables is not None:
            assert len(input_variables) == self.input_size

        if arch == 'plrnn':
            self.cell = PLRNNCell(input_size, state_size)
        elif arch == 'mlp':
            self.cell = MLPCell(input_size, state_size, hidden_size, alpha, slv=slv,
                                slv_trainable=slv_trainable, input_variables=input_variables)
        elif arch == 'mlptanh':
            self.cell = MLPTanhCell(input_size, state_size, hidden_size, alpha, slv=slv,
                                    slv_trainable=slv_trainable, input_variables=input_variables)
        elif arch == 'gru':
            self.cell = GRUCell(input_size, state_size)
        elif arch == 'gru2':
            self.cell = GRU2Cell(input_size, state_size, hidden_size)
        else:
            raise ValueError(f"Unknown arch '{arch}'")

    @jit.script_method
    def forward(self, input, xteacher, z0, forcing: bool, forcing_interval: Optional[int]=None):
        # input size: (bs, nt, n_input)
        # xteacher size: (bs, nt, n_output)
        # z0 size: (bs, n_hidden)

        bs, nt, n_output = xteacher.shape
        if forcing_interval is None:
            forcing_interval = nt + 1

        outs = []
        z = z0[:,:]
        outs.append(z)

        for i in range(nt):
            if forcing and (i > 0) and (i % forcing_interval == 0):
                z = torch.cat([xteacher[:,i,:], z[:,n_output:]], dim=1)            

            z = self.cell(input[:,i,:], z)
            outs.append(z)

        outs = torch.stack(outs, dim=1)
        # outs shape: (bs, nt+1, n_hidden)

        return outs
