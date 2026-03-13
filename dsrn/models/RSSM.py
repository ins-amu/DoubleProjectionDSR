


import torch
import torch.nn as nn
import torch.jit as jit


from .DilatedCNNEncoder import DilatedCNNEncoder
from . import temporal_masking
from .stats import kldiv_diag, log_normal_pdf


class DiagGaussian():
    def __init__(self, mu, lv):
        self.mu = mu
        self.lv = lv

class RolloutRes():
    def __init__(self, po, ps, qs, h, forcing_times):
        self.po = po
        self.ps = ps
        self.qs = qs
        self.h = h
        self.forcing_times = forcing_times


class MaskedTanh(nn.Module):
    def __init__(self, mask: torch.Tensor):
        """
        mask: a boolean or {0,1} tensor that will be broadcast 
              to the shape of the inputs during forward().
        """
        super().__init__()
        # register as buffer so it moves with the model (GPU/CPU)
        self.register_buffer("mask", mask.bool())

    def forward(self, x):
        # Broadcast mask to match x's shape
        mask = self.mask
        for _ in range(x.ndim - mask.ndim):
            mask = mask.unsqueeze(0)

        # Apply tanh only where mask == True
        return torch.where(mask, torch.tanh(x), x)


class RSSM(nn.Module):
    def __init__(self, nh, ns, nobs, nf, n_h2s, n_hs2o, n_fh2s, n_f2h0,
                 encoder, causal_encoder=None, n_ignore=0, arch='gru'):
        """
        RSSM model from Hafner et al. (2019), https://doi.org/10.48550/arXiv.1811.04551

        nh: Number of deterministic states
        ns: Number of stochastic states
        no: Number of observed variables
        nf: Number of features to extract from observed variables
        """

        super().__init__()

        self.nh = nh
        self.ns = ns
        self.no = nobs
        self.nf = nf
        self.n_obs = nobs  # Two names for the same thing to keep consistency across models
        self.arch = arch
        
        if arch == 'gru':
            self.f = nn.GRUCell(input_size=ns, hidden_size=nh)
            self.n_hidden_states = nh
        elif arch == 'lstm':
            self.f = nn.LSTMCell(input_size=ns, hidden_size=nh)
            self.n_hidden_states = 2*nh
        else:
            raise ValueError(f"RSSM architecture '{arch}' not supported.")

        self.encoder = DilatedCNNEncoder(nobs, nf, output='point', **encoder)
        self.causal_encoder = None
        if causal_encoder is not None:
            if causal_encoder.get('causal', 0) != 1:
                raise ValueError("Causal encoder must be declared as causal.")
            self.causal_encoder = DilatedCNNEncoder(nobs, nf, output='point', **causal_encoder)


        # Transformation layers return mean and log variances (of a diagonal Gaussian)
        self.h2s  = nn.Sequential(nn.Linear(nh, n_h2s), nn.ReLU(), nn.Linear(n_h2s, 2*ns))
        self.hs2o = nn.Sequential(nn.Linear(nh + ns, n_hs2o), nn.ReLU(), nn.Linear(n_hs2o, 2*nobs))
        self.fh2s = nn.Sequential(nn.Linear(nf + nh, n_fh2s), nn.ReLU(), nn.Linear(n_fh2s, 2*ns))

        if self.arch == 'gru':
            self.f2h0 = nn.Sequential(nn.Linear(nf, n_f2h0), nn.ReLU(), nn.Linear(n_f2h0, self.n_hidden_states))
        elif self.arch == 'lstm':
            mask = torch.cat((torch.ones(self.nh), torch.zeros(self.nh)), dim=0)
            self.register_buffer("f2h0_mask", mask)
            self.f2h0 = nn.Sequential(nn.Linear(nf, n_f2h0), nn.ReLU(), nn.Linear(n_f2h0, self.n_hidden_states),
                                      MaskedTanh(mask))

        self.skip = temporal_masking.get_skip_tuple(n_ignore)

        self._name = 'rssm'


    def loss(self, x, forcing_interval=None, overshooting='none'):
        "Loss for a batch of samples x using forcing_interval"

        bs, nt, nobs = x.shape

        # Project observation to features
        features = self.encoder(x)     # shape: (bs, nt, nf)

        # Mask beginning/end
        mask = temporal_masking.skip_tuple_to_mask(self.skip, nt)
        x = x[:, mask]
        features = features[:, mask]
        
        # Calculate h0
        h0 = self.f2h0(features[:,0,:])

        if overshooting == 'none':
            r = self.simulate(h0, features, forcing=True, forcing_interval=1, obs_from_post=True)
            logp_o_s = log_normal_pdf(x, r.po.mu, r.po.lv, raxis=(1,2))
            kl = kldiv_diag(r.qs.mu, r.qs.lv, r.ps.mu, r.ps.lv, axis=(1,2))

        elif overshooting == 'latent':
            r1 = self.simulate(h0, features, forcing=True, forcing_interval=1, obs_from_post=True)
            rN = self.simulate(h0, features, forcing=True, forcing_interval=forcing_interval,
                               obs_from_post=True)
            
            logp_o_s = log_normal_pdf(x, r1.po.mu, r1.po.lv, raxis=(1,2))
            
            fmask = ~rN.forcing_times

            # Scaling coeffiecient is there to account for the mask
            # Note the detach() calls to prevent gradients through the posterior
            kl = (1./(1. + sum(fmask) / len(fmask))) * (
                  kldiv_diag(r1.qs.mu, r1.qs.lv, r1.ps.mu, r1.ps.lv, axis=(1,2))
                + kldiv_diag(rN.qs.mu.detach()[:,fmask], rN.qs.lv.detach()[:,fmask],
                             rN.ps.mu[:,fmask], rN.ps.mu[:,fmask],
                             axis=(1,2))
            )

        elif overshooting == 'observation':
            r1 = self.simulate(h0, features, forcing=True, forcing_interval=1, obs_from_post=True)
            rN = self.simulate(h0, features, forcing=True, forcing_interval=forcing_interval,
                               obs_from_post=False)
            
            logp_o_s = 0.5*(
                  log_normal_pdf(x, r1.po.mu, r1.po.lv, raxis=(1,2))
                + log_normal_pdf(x, rN.po.mu, rN.po.lv, raxis=(1,2))
            )
            kl = kldiv_diag(r1.qs.mu, r1.qs.lv, r1.ps.mu, r1.ps.lv, axis=(1,2))

        else:
            raise ValueError(f"Unexpected overshooting type '{overshooting}'")

  
        loss = -logp_o_s + kl

        return torch.mean(loss)

    
    def ce_loss(self, x):
        # x shape: (bs, nt, nobs)

        if self.causal_encoder is None:
            return None
        
        nt = x.shape[1] 
        xg1 = self.encoder(x)
        xg2 = self.causal_encoder(x)
        loss = torch.norm((xg1 - xg2)[:,self.skip[0]:nt-self.skip[1],:], dim=(1,2))

        return torch.mean(loss)


    def simulate(self, h0, features, eps_prior=None, eps_post=None,
                 forcing=False, forcing_interval=None, obs_from_post=True):

        bs, nt, nf = features.shape
        
        mus_po = h0.new_zeros((bs, nt, self.no))
        lvs_po = h0.new_zeros((bs, nt, self.no))
        mus_ps = h0.new_zeros((bs, nt, self.ns))
        lvs_ps = h0.new_zeros((bs, nt, self.ns))
        mus_qs  = h0.new_zeros((bs, nt, self.ns))
        lvs_qs  = h0.new_zeros((bs, nt, self.ns))
        hs = h0.new_zeros((bs, nt, self.n_hidden_states))
        forcing_times = h0.new_zeros(nt, dtype=bool)

        if eps_prior is None:
            eps_prior = torch.randn((bs, nt, self.ns), device=h0.device)

        if eps_post is None:
            eps_post  = torch.randn((bs, nt, self.ns), device=h0.device)

        if forcing_interval is None:
            forcing_interval = nt + 1

        h = h0
        h_readout = h0[:,:self.nh]
        for i in range(nt):
            hs[:,i] = h

            mus_ps[:,i], lvs_ps[:,i] = torch.split(self.h2s(h_readout), self.ns, dim=1)
            s_prior = eps_prior[:,i].mul(torch.exp(0.5*lvs_ps[:,i])).add_(mus_ps[:,i])

            mus_qs[:,i], lvs_qs[:,i] = torch.split(
                self.fh2s(torch.cat((features[:,i], h_readout), dim=1)),
                self.ns, dim=1)
            s_post = eps_post[:,i].mul(torch.exp(0.5*lvs_qs[:,i])).add_(mus_qs[:,i])

            if forcing and ((i+1) % forcing_interval == 0):
                s = s_post
                if i + 1 < nt:
                    forcing_times[i+1] = True
            else:
                s = s_prior

            s_obs = s_post if obs_from_post else s_prior
            mus_po[:,i], lvs_po[:,i] = torch.split(self.hs2o(torch.cat((h_readout, s_obs), dim=1)), self.no, dim=1)

            if self.arch == 'gru':
                h = self.f(s, h)
            elif self.arch == 'lstm':
                h = torch.cat(self.f(s, torch.split(h, self.nh, dim=1)), dim=1)

            h_readout = h[:,:self.nh]
    
        return RolloutRes(DiagGaussian(mus_po, lvs_po), 
                          DiagGaussian(mus_ps, lvs_ps), 
                          DiagGaussian(mus_qs, lvs_qs), 
                          hs, forcing_times)

    def get_latent_state(self, x, causal=False):
        # Quick estimation of the latent state using the initialization projection

        encoder = self.causal_encoder if (causal and self.causal_encoder) else self.encoder

        features = encoder(x)     # shape: (bs, nt, nf)
        h = self.f2h0(features)   # shape: (bs, nt, nh)
        return h

    def get_latent_state_evo(self, x):
        bs, nt, _ = x.shape
        features = self.encoder(x)       # shape: (bs, nt, nf)
        h0 = self.f2h0(features)[:,0,:]  # shape: (bs, nh)
        h = self.simulate(h0, features, forcing=True, forcing_interval=1).h
        return h
    
    def get_latent_state_last(self, x, n=1):
        # x = torch.repeat_interleave(x, n, dim=0)
        # z0 = self.get_latent_state_evo(x)[:,-1,:]

        z0 = self.get_latent_state(x, causal=True)[:,-1,:]
        z0 = torch.repeat_interleave(z0, n, dim=0)

        return z0

    def sample(self, h0, nt, noise=True, eps=None, return_latent=False, nsamples=1):
        
        # Repeat and rename
        h0 = torch.repeat_interleave(h0, nsamples, dim=0)
        bs = h0.shape[0]

        if not noise:
            eps = torch.zeros((bs, nt+1, self.ns))

        dummy_features = torch.zeros((bs, nt+1, self.nf))
        res = self.simulate(
            h0, dummy_features, eps_prior=eps, forcing=False, obs_from_post=False)
       
        if return_latent:
            return res.po.mu, res.h, res.ps.mu, res.ps.lv
        else:
            return res.po.mu
