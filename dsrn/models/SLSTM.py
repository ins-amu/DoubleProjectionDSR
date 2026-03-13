from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.jit as jit


class SLSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, output_size, bias=True):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = nn.LSTMCell(input_size=input_size+output_size, hidden_size=hidden_size)
        self.proj = nn.Linear(hidden_size, 2*output_size, bias=bias)

    @jit.script_method
    def forward(self, input, h0, c0, y0):
        # input shape: (bs, nt, input_size)
        # h0 shape: (bs, hidden_size)
        # y0 shape: (bs, output_size)

        bs, nt, _ = input.shape

        h, c, y = h0, c0, y0

        mus = input.new_zeros((bs, nt, self.output_size))
        lvs = input.new_zeros((bs, nt, self.output_size))
        ys  = input.new_zeros((bs, nt, self.output_size))

        eps = torch.randn((nt, bs, y0.shape[1]), device=input.device)

        for i in range(nt):
            x = torch.cat((input[:,i,:], y), dim=1)

            h, c = self.cell(x, (h, c))
            mu, lv = torch.split(self.proj(h), (self.output_size, self.output_size), dim=1)
            y = eps[i].mul(torch.exp(0.5*lv)).add_(mu)

            mus[:,i,:] = mu
            lvs[:,i,:] = lv
            ys[:,i,:]  = y

        return ys, mus, lvs