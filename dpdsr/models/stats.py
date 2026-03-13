
import numpy as np

import torch



def kldiv_diag(mu1, lv1, mu2, lv2, axis=1):
    return 0.5 * torch.sum((lv2 - lv1 + (torch.exp(lv1) + (mu1 - mu2)**2)/(torch.exp(lv2)) - 1.), dim=axis)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = np.log(2. * np.pi)
    lnp = -0.5 * ((sample - mean)**2. * torch.exp(-logvar) + logvar + log2pi)
    if raxis is None:
        return lnp
    else:
        return torch.sum(lnp, dim=raxis)


def lpdf_ar1(x, tau, lv_process):
    """
    Log probability density of x under the assumption of AR(1) process with decay time tau
    and process log-variance lv_process.
    """
    alpha = torch.exp(-1./tau)
    lv_noise = lv_process + torch.log(1. - alpha**2)
    logp = (  log_normal_pdf(x[:,0], 0., lv_process, raxis=None)
            + log_normal_pdf(x[:,1:], alpha*x[:,:-1], lv_noise, raxis=1))
    return logp   # size: (batch_size, nvar)

def moving_average(x, n=3, axis=-1):
    xt = torch.transpose(x, 0, axis)
    ret = torch.cumsum(xt, dim=0)
    ret[n:] += -ret[:-n]
    return torch.transpose(ret[n-1:]/n, 0, axis)
