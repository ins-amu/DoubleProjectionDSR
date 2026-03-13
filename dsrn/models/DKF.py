from collections.abc import Sequence

import numpy as np

import torch
import torch.nn as nn

from .RNNForcing import RNNForcing
from .Observation import IdentityObs, LinearObs, NonlinearObs
from .DilatedCNNEncoder import DilatedCNNEncoder
from .stats import kldiv_diag, log_normal_pdf


class DKF(nn.Module):
    "Deep Kalman Filter model"

    def __init__(self, n_states, n_obs, n_samples=1, f=None, encoder=None, causal_encoder=None,
                 g=None, olv=0., lambda_g=0., n_ignore=None):

        super().__init__()

        self.n_obs = n_obs
        self.n_states = n_states
        self.n_samples = n_samples
        self.n_stoch = n_states

        self.olv = torch.tensor(olv)
        self.lambda_g = lambda_g

        # Skip first/last steps in evaluation
        if n_ignore is None:
            self.skip = (0,0)           
        elif isinstance(n_ignore, Sequence) and len(n_ignore) == 2:
            self.skip = (n_ignore[0], n_ignore[1])
        else:            
            self.skip = (n_ignore, n_ignore)

        # Dynamics model
        self.f = RNNForcing(input_size=n_states, state_size=n_states, arch='mlp', **f)

        # Observation model
        gtype = g.pop('type', 'identity')
        if gtype == 'identity':
            self.g = IdentityObs(vars=range(0, n_obs))
        elif gtype == 'linear':
            self.g = LinearObs(n_states=n_states, stoch_vars=[], n_obs=n_obs, readout='all')
        elif gtype == 'nonlinear':
            self.g = NonlinearObs(n_states=n_states, stoch_vars=[], n_obs=n_obs,
                                         readout='all', n_hidden=g['n_hidden'])
        else:
            raise ValueError(f"Unknown observation model '{gtype}'")

        # Encoder
        self.encoder = DilatedCNNEncoder(n_obs, n_states, output='ar', **encoder)

        # Additional causal encoder
        self.causal_encoder = None
        if causal_encoder is not None:
            if causal_encoder.get('causal', 0) != 1:
                raise ValueError("Causal encoder must be declared as causal.")
            self.causal_encoder = DilatedCNNEncoder(n_obs, n_states, output='ar', **causal_encoder)

        self.has_forcing = False
        self._name = 'dkf'


    def loss(self, x):
        # x shape: (bs, nt, nobs)
        bs, nt, nobs = x.shape

        # Project to system states
        z, zmu, zlv = self.encoder(x, self.n_samples)

        # Cut everything
        mask = torch.ones(nt, dtype=bool)
        mask[0:self.skip[0]]     = False
        mask[nt-self.skip[1]:nt] = False

        x = x[:,mask]
        z = z[:,mask]
        zmu = zmu[:,mask]
        zlv = zlv[:,mask]

        # Repeat data
        x = torch.repeat_interleave(x, self.n_samples, dim=0)

        dummy_eps = torch.zeros((bs*self.n_samples, nt - self.skip[0] - self.skip[1], self.n_states))
        zpred = self.f.cell(dummy_eps, z)

        # Cascade following Girin et al., Dynamical Variational Autoencoders: A Comprehensive Review,
        # p64.
        kl_z = (  kldiv_diag(zmu[:,0,:], zlv[:,0,:], torch.tensor(0.), torch.tensor(0.), axis=[1])
                + kldiv_diag(zmu[:,1:,:], zlv[:,1:,:], zpred[:,:-1,:], self.f.cell.slv, axis=[1,2]))

        # Observation loss for data
        logp_x_z = log_normal_pdf(x[:,:,:], self.g(z[:,:,:]), self.olv, raxis=(1,2))

        # Loss function
        loss = -logp_x_z + kl_z

        # g regularization
        for w in self.g.weights():
            loss += self.lambda_g * torch.norm(w, p=1)

        return torch.mean(loss)
    
    def ce_loss(self, x):
        # x shape: (bs, nt, nobs)

        if self.causal_encoder is None:
            return None
        
        nt = x.shape[1] 
        xg1 = self.encoder(x)[0]
        xg2 = self.causal_encoder(x)[0]
        loss = torch.norm((xg1 - xg2)[:,self.skip[0]:nt-self.skip[1],:], dim=(1,2))

        return torch.mean(loss)
        
    def sample(self, z0, nt, noise=True, eps=None, return_latent=False, nsamples=1):

        z0 = torch.repeat_interleave(z0, nsamples, dim=0)
        bs = z0.shape[0]

        if not noise:
            eps = torch.zeros((bs,nt,self.n_stoch))
        elif noise and (eps is None):
            eps = torch.randn((bs,nt,self.n_stoch))
        else:
            pass

        dummy_xg = torch.zeros((bs, nt, 0))
        z = self.f(eps, dummy_xg, z0, forcing=False)
        x = self.g(z)

        if return_latent:
            return x, z
        else:
            return x

    def get_latent_state(self, x, n=1, causal=False):
        encoder = self.causal_encoder if (causal and self.causal_encoder) else self.encoder
        return encoder(x, nsamples=n)[0]
    
    def get_latent_state_last(self,x, n=1):
        return self.get_latent_state(x, n, causal=True)[:,-1,:]

