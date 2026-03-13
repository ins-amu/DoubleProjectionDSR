from typing import Optional

import torch
import torch.nn as nn
import torch.jit as jit

from .DilatedCNNEncoder import DilatedCNNEncoder
from .stats import log_normal_pdf


class ARLSTM(jit.ScriptModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.cell = nn.LSTMCell(input_size=input_size+output_size, hidden_size=hidden_size)
        self.proj = nn.Linear(hidden_size, 2*output_size)

    @jit.script_method
    def forward(self, input: torch.Tensor, h0: torch.Tensor, c0: torch.Tensor, yt: torch.Tensor,
                gamma: Optional[float]=None, eps: Optional[torch.Tensor]=None):
        # h0 shape: (bs, hidden_size)
        # c0 shape: (bs, hidden_size)
        # yt shape: (bs, nt, output_size)
        #    yt[0] corresponds to the first input, ie. h1,c1 = f(h0,c0,yt[0])
        # eps shape: (bs, nt, output_size) (optional)

        bs, nt, _ = yt.shape

        h, c, y = h0, c0, yt[:,0,:]

        mus = yt.new_zeros((bs, nt, self.output_size))
        lvs = yt.new_zeros((bs, nt, self.output_size))
        ys  = yt.new_zeros((bs, nt, self.output_size))

        if gamma is None:
            gamma = torch.tensor(1.)

        # Sample noise and teacher replacement times
        if eps is None:
            eps = torch.randn((bs, nt, self.output_size), device=yt.device)
        use_teacher = torch.rand(nt) > gamma

        for i in range(nt):
            if use_teacher[i]:
                y = yt[:,i]

            x = torch.cat((input[:,i,:], y), dim=1)

            h, c = self.cell(x, (h, c))
            mu, lv = torch.split(self.proj(h), (self.output_size, self.output_size), dim=1)
            y = eps[:,i].mul(torch.exp(0.5*lv)).add_(mu)

            mus[:,i,:] = mu
            lvs[:,i,:] = lv
            ys[:,i,:]  = y

        return ys, mus, lvs
        


class ARLSTMModel(nn.Module):
    """Autoregressive LSTM with scheduled sampling"""

    def __init__(self, n_hidden, n_obs, n_init, n_hidden_init, init_chunk, encoder, n_samples=1):
        
        super().__init__()

        self.n_obs = n_obs
        self.init_chunk = init_chunk
        self.n_hidden = n_hidden
        self.n_samples = n_samples        
        
        self.encoder = DilatedCNNEncoder(n_obs, n_init, output='point', causal=1, **encoder)

        self.expand_init = nn.Sequential(
            nn.Linear(n_init, n_hidden_init),
            nn.ReLU(),
            nn.Linear(n_hidden_init, 2*n_hidden)
        )

        self.f = ARLSTM(input_size=0, hidden_size=n_hidden, output_size=n_obs)

        self._name = 'arlstm'
        self.causal_encoder = None


    def loss(self, x, gamma):
        # x shape: (bs, nt, nobs)
        bs, nt, _ = x.shape

        # Split to initial and estimated data
        xinit = x[:,:self.init_chunk,:]
        yt = x[:, self.init_chunk-1:,:]
        ntp = yt.shape[1]

        # Estimate initial conditions
        # h0, c0 = torch.split(self.expand_init(self.encoder(xinit)[:,-1,:]), 2, dim=1)
        h0, c0 = self.get_ic(xinit)

        # Repeat for multiple samples
        bsr = bs*self.n_samples
        h0 = torch.repeat_interleave(h0, self.n_samples, dim=0)
        c0 = torch.repeat_interleave(c0, self.n_samples, dim=0)
        yt = torch.repeat_interleave(yt, self.n_samples, dim=0)

        # Run the AR LSTM
        input = torch.empty((bsr, ntp, 0))
        _, mu, lv = self.f(input, h0, c0, yt, gamma=gamma)
        # mu, lv shape: (bs, ntp, nobs)

        # Loss function
        loss = -log_normal_pdf(yt[:,1:], mu[:,:-1], lv[:,:-1], raxis=(1,2))

        return torch.mean(loss)


    def sample(self, z0, nt, noise=True, nsamples=1):
        h0, c0, y0 = torch.split(z0, [self.n_hidden, self.n_hidden, self.n_obs], dim=1)
        h0 = torch.repeat_interleave(h0, nsamples, dim=0)
        c0 = torch.repeat_interleave(c0, nsamples, dim=0)
        y0 = torch.repeat_interleave(y0, nsamples, dim=0)
        bs = h0.shape[0]

        input = torch.zeros((bs, nt, 0))
        yt    = torch.zeros((bs, nt, self.n_obs))
        yt[:,0,:] = y0
        eps = torch.zeros((bs, nt, self.n_obs)) if noise is False else None

        yp = self.f(input, h0, c0, yt, gamma=1.0, eps=eps)[0]

        # Prepend y[0] to be consistent with other model
        yp = torch.cat([yt[:,[0],:], yp], dim=1)

        return yp
    

    def get_ic(self, x):
        h0, c0 = torch.split(self.expand_init(self.encoder(x)[:,-1,:]), self.n_hidden, dim=1)
        return h0, c0
    

    def get_latent_state_last(self, x, n=1):
        z0 = self.expand_init(self.encoder(x)[:,-1,:])

        # Append the last observation (input to the first layer)
        z0 = torch.cat([z0, x[:,-1,:]], dim=1)
        
        z0 = torch.repeat_interleave(z0,  n, dim=0)
        return z0