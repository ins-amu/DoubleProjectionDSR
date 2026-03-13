from collections.abc import Sequence

import numpy as np

import torch
import torch.nn as nn

from .RNNForcing import RNNForcing
from .Observation import IdentityObs, LinearObs, NonlinearObs
from .DilatedCNNEncoder import DilatedCNNEncoder
from .stats import kldiv_diag, log_normal_pdf
from . import temporal_masking 

class DPDSR(nn.Module):
    def __init__(self, n_states, n_obs, stoch_vars=0, n_guiding=0, n_samples=1, olv=0., glv=0.,
                 guiding_encoder=None, noise_encoder=None, causal_encoder=None, f=None, g=None,
                 lambda_x=0., lambda_g=0., lambda_z=0., n_ignore=None):

        super().__init__()

        self.n_obs = n_obs
        self.n_states = n_states
        self.n_guiding = n_guiding
        self.n_samples = n_samples

        if hasattr(stoch_vars, '__len__'):
            self.n_stoch = len(stoch_vars)
            self.stoch_vars = stoch_vars
        else:
            self.n_stoch = stoch_vars
            self.stoch_vars = range(self.n_states-self.n_stoch, self.n_states)

        self.olv = torch.tensor(olv)

        if glv is None:
            self.use_glv = False
            self.glv = torch.tensor(0.)
        else:
            self.use_glv = True
            self.glv = torch.tensor(glv)

        self.lambda_x = lambda_x
        self.lambda_g = lambda_g
        self.lambda_z = lambda_z

        # Skip first/last steps in evaluation
        self.skip = temporal_masking.get_skip_tuple(n_ignore)

        # Dynamics model
        self.f = RNNForcing(input_size=self.n_stoch, state_size=n_states, **f)

        # Observation model
        gtype = g.pop('type', 'identity')
        if gtype == 'identity':
            self.g = IdentityObs(vars=range(0, n_obs))
        elif gtype == 'linear':
            self.g = LinearObs(n_states=n_states, stoch_vars=self.stoch_vars, n_obs=n_obs,
                                      readout=g['readout'])
        elif gtype == 'nonlinear':
            self.g = NonlinearObs(n_states=n_states, stoch_vars=self.stoch_vars, n_obs=n_obs,
                                         readout=g['readout'], n_hidden=g['n_hidden'])
        else:
            raise ValueError(f"Unknown observation model '{gtype}'")
        self.obs_is_guiding = (gtype == 'identity')

        # Guiding and noise encoders
        self.guiding_encoder = DilatedCNNEncoder(n_obs, n_guiding, output='point', **guiding_encoder)
        noise_output = 'normal' if noise_encoder['rnn_size'] == 0 else 'ar'
        self.noise_encoder = DilatedCNNEncoder(n_obs+n_guiding, self.n_stoch, output=noise_output,
                                               **noise_encoder)
        
        # Additional causal guiding encoder
        self.causal_encoder = None        
        if causal_encoder is not None:
            if causal_encoder.get('causal', 0) != 1:
                raise ValueError("Causal guiding encoder must be declared as causal.")
            self.causal_encoder = DilatedCNNEncoder(n_obs, n_guiding, output='point', 
                                                    **causal_encoder)

        # Initial conditions
        n_guided = n_obs+n_guiding if self.obs_is_guiding else n_guiding
        self.init = nn.Linear(n_obs + n_guiding, n_states - n_guided)
        nn.init.normal_(self.init.weight, mean=0., std=0.01)
        nn.init.zeros_(self.init.bias)

        self.has_forcing = True
        self._name = 'dsrn'


    def loss(self, x, forcing_interval=None):
        "Loss for a batch of samples x using forcing_interval"

        # x shape: (bs, nt, nobs)
        bs, nt, nobs = x.shape

        # Project to guiding time series
        zhat = self.guiding_encoder(x)
        x_and_zhat = torch.cat((x, zhat), dim=2)

        # Project to noise (if there is noise)
        eps, mu, lv = self.noise_encoder(x_and_zhat, self.n_samples)

        # Cut everything
        mask = temporal_masking.skip_tuple_to_mask(self.skip, nt)

        x = x[:,mask]
        zhat = zhat[:,mask]
        x_and_zhat = x_and_zhat[:,mask]
        eps = eps[:,mask]
        mu = mu[:,mask]
        lv = lv[:,mask]

        # KL divergence for noise
        kl_eps = kldiv_diag(mu, lv, torch.tensor(0.), torch.tensor(0.), axis=[1,2])

        # Estimate initial conditions z0 from x0 (linear projection is enough)
        if self.obs_is_guiding:
            z0 = torch.cat([x[:,0,:], zhat[:,0,:], self.init(x_and_zhat[:,0,:])], dim=1)
        else:
            z0 = torch.cat([zhat[:,0,:], self.init(x_and_zhat[:,0,:])], dim=1)

        # Repeat data
        x           = torch.repeat_interleave(x, self.n_samples, dim=0)
        zhat        = torch.repeat_interleave(zhat, self.n_samples, dim=0)
        x_and_zhat  = torch.repeat_interleave(x_and_zhat, self.n_samples, dim=0)
        z0          = torch.repeat_interleave(z0, self.n_samples, dim=0)

        # Simulate with teacher forcing
        xg = x_and_zhat if self.obs_is_guiding else zhat
        z = self.f(eps, xg, z0, forcing=True, forcing_interval=forcing_interval)

        # Observation loss for xguiding
        if self.use_glv:
            guiding_vars = (range(self.n_obs, self.n_obs+self.n_guiding) if self.obs_is_guiding
                            else range(0, self.n_guiding))
            logp_xg_z = log_normal_pdf(zhat, z[:,:-1,guiding_vars], self.glv, raxis=(1,2))
        else:
            logp_xg_z = torch.tensor(0.)

        # Observation loss for data
        logp_x_z = log_normal_pdf(x, self.g(z[:,:-1,:]), self.olv, raxis=(1,2))

        # Loss function
        loss = -logp_x_z  + kl_eps - logp_xg_z

        # Initial conditions
        logp_z0 = log_normal_pdf(z0, torch.tensor(0.), torch.tensor(0.), raxis=1)
        loss += -self.lambda_z * logp_z0

        # x regularization
        xg_mu = torch.mean(zhat, dim=(0,1))
        xg_sd = torch.std(zhat, dim=(0,1))
        xg_lv = 2*torch.log(xg_sd)
        loss += self.lambda_x * nt * kldiv_diag(xg_mu, xg_lv, torch.tensor(0.), torch.tensor(0.), axis=0)

        # g regularization
        for w in self.g.weights():
            loss += self.lambda_g * torch.norm(w, p=1)

        return torch.mean(loss)
    
    def ce_loss(self, x):
        "Causal encoder loss."
        # x shape: (bs, nt, nobs)

        if self.causal_encoder is None:
            return None
        
        nt = x.shape[1]
        xg1 = self.guiding_encoder(x)
        xg2 = self.causal_encoder(x)
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
        
        
    def get_latent_state_init(self, x, causal=False):
        "Approach using the init projection."

        encoder = self.causal_encoder if (causal and self.causal_encoder) else self.guiding_encoder
        zhat = encoder(x)
        x_and_zhat = torch.cat((x, zhat), dim=2)

        if self.obs_is_guiding:
            z = torch.cat([x, zhat, self.init(x_and_zhat)], dim=2)
        else:
            z = torch.cat([zhat, self.init(x_and_zhat)], dim=2)
        
        return z

    def get_latent_state_last(self, x, n=1):
       
        z0 = self.get_latent_state_init(x, causal=True)[:,-1,:]
        z0 = torch.repeat_interleave(z0, n, dim=0)

        return z0
        

    def get_latent_state_last_evo(self, x, n=1, skip_last=0):

        nt = x.shape[1]
    
        zhat = self.guiding_encoder(x)
        x_and_zhat = torch.cat((x, zhat), dim=2)

        if self.obs_is_guiding:
            z = torch.cat([x, zhat, self.init(x_and_zhat)], dim=2)
        else:
            z = torch.cat([zhat, self.init(x_and_zhat)], dim=2)

        z0 = z[:,0,:]
        eps = self.noise_encoder(x_and_zhat, n)[0]

        # Repeat data
        x           = torch.repeat_interleave(x, n, dim=0)
        zhat        = torch.repeat_interleave(zhat, n, dim=0)
        x_and_zhat  = torch.repeat_interleave(x_and_zhat, n, dim=0)
        z0          = torch.repeat_interleave(z0, n, dim=0)

        xg = x_and_zhat if self.obs_is_guiding else zhat
        z1 = self.f(eps[:,:nt-skip_last], xg[:,:nt-skip_last], z0, forcing=True, forcing_interval=1)[:,nt-1-skip_last]

        eps = torch.randn((n, skip_last, self.n_stoch))
        dummy_x = torch.zeros(n, skip_last, self.n_states)
        z2 = self.f(eps, dummy_x, z1, forcing=False)[:,skip_last]
        
        return z2