
import os.path
import copy

import numpy as np
import scipy.stats as stats
import scipy.special
import scipy.signal as sig
import scipy.ndimage as ndi

import torch

from . import utils


def laplace_smoothing(x, n, alpha):
    return (x+alpha)/(n + alpha*x.size)

def kl_div_bins(x1, x2, nbins=10):
    lo = np.min(x1, axis=0)
    hi = np.max(x1, axis=0)
    ranges = list(zip(lo, hi))
    n = x1.shape[0]

    hist1 = np.histogramdd(x1, bins=nbins, range=ranges)[0].ravel()
    hist2 = np.histogramdd(x2, bins=nbins, range=ranges)[0].ravel()

    p = laplace_smoothing(hist1, n=n, alpha=1e-5)
    q = laplace_smoothing(hist2, n=n, alpha=1e-5)

    kl = np.sum(scipy.special.rel_entr(p, q))

    # Careful: stats.entropy normalizes the distributions, which is
    # not what we want since some/many/all values of x2 may be outside
    # the range of x1
    # kl = stats.entropy(p, q)

    return kl

def spectra_hellinger(x1, x2, sigma=None, nperseg=4096):
    Hs = []

    for i in range(x1.shape[1]):
        f1, pxx1 = sig.welch(x1[:,i], fs=1, nperseg=nperseg)
        f2, pxx2 = sig.welch(x2[:,i], fs=1, nperseg=nperseg)

        # Smooth the spectra
        if sigma is not None:
            p1 = ndi.gaussian_filter1d(pxx1, sigma=sigma)
            p2 = ndi.gaussian_filter1d(pxx2, sigma=sigma)
        else:
            p1, p2 = pxx1, pxx2

        # Normalize
        p1 /= np.sum(p1)
        p2 /= np.sum(p2)

        H = 1./np.sqrt(2.) * np.linalg.norm((np.sqrt(p1) - np.sqrt(p2)))
        Hs.append(H)

    return np.mean(Hs)


def wasserstein_distance(x1, x2, nbins=10, cutoff=2.):
    d = x1.shape[1]
    sd = np.std(x1, axis=0)
    lim = cutoff*sd

    x1 = np.fmin(np.fmax(x1, -lim), lim)
    x2 = np.fmin(np.fmax(x2, -lim), lim)

    dists = []
    for i in range(d):
        hist1, edges = np.histogram(x1[:,i], bins=nbins, range=(-lim[i], lim[i]))
        hist2, _     = np.histogram(x2[:,i], bins=nbins, range=(-lim[i], lim[i]))

        edges = np.array(edges)
        centers = (edges[1:] + edges[:-1])/2.

        dist = stats.wasserstein_distance(centers, centers, hist1, hist2)
        dists.append(dist)

    mdist = np.mean(dists)

    return mdist


def wasserstein_isi(x1, x2, index=0, height=2, prominence=1, outliers_method=None, outliers_lt=0., outliers_ut=2.):

    peaks1 = sig.find_peaks(x1[:,index], height=height, prominence=prominence)[0]
    peaks2 = sig.find_peaks(x2[:,index], height=height, prominence=prominence)[0]
    isi1 = peaks1[1:] - peaks1[:-1]
    isi2 = peaks2[1:] - peaks2[:-1]

    if len(isi1) == 0 or len(isi2) == 0:
        disi = np.nan
    else:
        disi = stats.wasserstein_distance(isi1, isi2)

    # Outliers
    if (outliers_method is None) or (outliers_method == 'none'):
        nout = 0
    elif outliers_method == 'mean':
        mu = np.mean(isi1)
        lo,hi = outliers_lt*mu, outliers_ut*mu
        nout = np.sum(isi2 < lo) + np.sum(isi2 > hi)

    return {'disi': disi, 'isi_nout': nout}


def get_names_longterm(config):
    default = ['kl', 'wasserstein', 'dfreq']
    isi = ['disi', 'isi_nout']

    variant = config['variant']
    if   variant == 'ecg':
        return default + isi
    elif variant == 'neuron':
        return default + isi
    else:
        return default


def compare_longterm_data(variant, x1, x2):
    default = {
            'kl': kl_div_bins(x1, x2, nbins=30),
            'wasserstein': wasserstein_distance(x1, x2, nbins=100, cutoff=3.),
            'dfreq': spectra_hellinger(x1, x2, sigma=2.)
    }

    if variant == 'ecg':
        return default | wasserstein_isi(x1, x2, index=0, height=2, prominence=1,
                                         outliers_method='mean', outliers_lt=0.5, outliers_ut=1.5)
    elif variant == 'neuron':
        return default | wasserstein_isi(x1, x2, index=0, height=1, prominence=0.5)
    else:
        return default


def calculate_prediction(model, x, config, noise):
    x = x[None,:,:]
    _, nt, nvars = x.shape

    variables = config['variables']
    n_embed  = config['n_embed']
    pe_n     = np.array(config['pe_n'])
    pw_c     = np.array(config['pw_c'])
    pw_maxn  = config['pw_maxn']
    max_eval = config['max_eval']
    nsamples = config['n_samples'] if noise else 1 # No need for multiple samples
    
    n_pred = max(max(pe_n), pw_maxn) + 1
    n_eval = min(max_eval, nt - n_pred - n_embed)

    rng = np.random.default_rng()
    ts = rng.permuted(np.r_[n_embed:nt-n_pred])[:n_eval]

    pes = []
    pws = []

    for t in ts:
        x_embed = torch.tensor(x[:,t-n_embed:t,:])
        z0 = model.get_latent_state_last(x_embed)
        
        xsim = model.sample(z0, n_pred, nsamples=nsamples, noise=noise)[:,1:,:].detach().numpy()
        xdat = x[:,t:t+n_pred,:]

        xsim = xsim[:,:,variables]
        xdat = xdat[:,:,variables]

        err = np.linalg.norm(xsim - xdat, axis=2)
        err[:,-1] = max(pw_c) + 1  # To assure prediction window is not zero

        # Prediction error
        pes.append([np.mean(err[:,:index]) for index in pe_n])

        # Prediction window
        pws.append(np.mean(np.argmax(err[:,:,None] > pw_c[None,None,:], axis=1), axis=0))

    pes = np.array(pes)
    pws = np.array(pws)
    
    return pes, pws

def get_names_prediction(config):
    return [f'PE{n}' for n in config['pe_n']] + [f'PW{c:.2f}' for c in config['pw_c']]

def evaluate_prediction(model, x, config, noise):
    pes, pws = calculate_prediction(model, x, config, noise)

    pes = np.mean(pes, axis=0)
    pws = np.median(pws, axis=0)

    res = (   {f'PE{n}':     pe for n,pe in zip(config['pe_n'], pes)} 
            | {f'PW{c:.2f}': pw for c,pw in zip(config['pw_c'], pws)})
    
    return res


def evaluate_longterm(model, x, config, noise):
    n_embed = config['n_embed']
    max_nt = config['max_nt']
    variant = config['variant']
    variables = config['variables']

    x = x[None,:,:]
    nt = min(max_nt, x.shape[1] - n_embed)

    x_embed = torch.tensor(x[:, 0:n_embed, :])
    z0 = model.get_latent_state_last(x_embed)

    xsim = model.sample(z0, nt, noise=noise).detach().numpy()[0,1:][:,variables]
    xdat = x[0,n_embed:n_embed+nt][:,variables]

    res = compare_longterm_data(variant, xdat, xsim)

    return res


def evaluate_model(data_file, model_format, config_file, model_file, params, eval_config, outfile, noise=False):

    default_eval_config = {
        'n_embed': 300,
        'max_nt': 20000,
        'pe_n': [10, 20, 50, 100],
        'pw_c': [0.1, 0.25, 0.5, 1.0],
        'pw_maxn': 200,
        'max_eval': 1000,
        'n_samples': 20,
        'variant': 'default',
        'variables': [0]
    }
    eval_config = default_eval_config | eval_config

    x = utils.load_datafile(data_file).astype(np.float32)

    params = copy.copy(params)
    params['noise'] = noise

    # Write header if needed
    names = get_names_longterm(eval_config) + get_names_prediction(eval_config)
    if not os.path.isfile(outfile):
        with open(outfile, 'w') as fh:
            line = ",".join(list(params.keys()) + names)
            fh.write(line + "\n")

    # Load model
    try:
        if model_format == 'dsrn':
            model = utils.load_model_dsrn(config_file, model_file)
        elif model_format == 'gtf':
            model = utils.load_model_gtf(model_file)
        else:
            raise ValueError(f"Unknown model format {model_format}")
    except Exception as e:
        print("Cannot instantiate the model due to the following exception:")
        print(e)
        model = None

    # Evaluate predictions
    if model is not None:
        try:
            model.eval()
            with torch.no_grad():
                res1 = evaluate_longterm(model, x[0], eval_config, noise)
                res2 = evaluate_prediction(model, x[0], eval_config, noise)
            res = res1 | res2
        except Exception as e:
            print("Cannot evaluate the model due to the following exception:")
            print(e)
            res = None
    else:
        res = None
       
    # Write results
    with open(outfile, 'a') as fh:
        if res is not None:
            line = ",".join([str(v) for v in params.values()] + [str(res[n]) for n in names])
        else:
            line = ",".join([str(v) for v in params.values()] + [str(np.nan) for n in names])
        fh.write(line + "\n")

    
    