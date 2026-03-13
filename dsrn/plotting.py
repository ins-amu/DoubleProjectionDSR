import os
import sys
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.signal as sig
import scipy.stats as stats
import pandas as pd

import torch

from . import utils
from .evaluate_model import wasserstein_distance, spectra_hellinger

matplotlib.use('agg')

color_dat = "#2357cf"
color_sim = "#b8a007"


def plot_samples(model, x, filename):
    direc = os.path.dirname(filename)
    if not os.path.isdir(direc):
        os.makedirs(direc)

    nsamples = min(x.shape[0], 4)
    bs = x.shape[0]
    nt = max(x.shape[1], 100)

    model.eval()

    if model._name == 'dsrn':
        z0 = model.get_latent_state_init(x)[:,0,:]
        xbar = model.sample(z0[:nsamples], nt=nt, nsamples=nsamples).detach()
    elif model._name == 'dkf':
        z0 = model.get_latent_state(x)[:,0,:]
        xbar = model.sample(z0[:nsamples], nt=nt, nsamples=nsamples).detach()
    elif model._name == 'arlstm':
        z0 = model.get_latent_state_last(x[:,-model.init_chunk:,:])
        xbar = model.sample(z0, nt, nsamples=nsamples).detach()
    elif model._name == 'rssm':
        h0 = model.get_latent_state(x)[:,0,:]
        xbar = model.sample(h0, nt, nsamples=nsamples).detach()
    else:
        raise ValueError(f"Unknown model {model._name}")

    xbar = xbar.cpu().numpy()
    x = x.cpu().numpy()

    plt.figure(figsize=(6*model.n_obs,2*nsamples), dpi=150)

    for i in range(model.n_obs):
        for j in range(nsamples):
            plt.subplot2grid((nsamples, model.n_obs), (j,i))
            plt.plot(x[j,:,i], label='data')
            plt.plot(xbar[j,:,i], label='sample')
            if i == 0 and j == 0:
                plt.legend(loc='upper left')
            plt.ylim(-3,3)
            plt.xlim(0, nt)

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_timeseries_lines(ax, x, colorscheme='dat', rng=(0,1)):
    plt.sca(ax)
    nt, n = x.shape

    for i in range(n):
        plt.plot(x[:,i] - 2*i)

    plt.ylim(-2*n, 2)
    plt.xlim(rng[0]*nt, rng[1]*nt)
    plt.grid()


def plot_timeseries_im(ax, x, rng=(0,1)):
    plt.sca(ax)
    nt, n = x.shape
    plt.imshow(x.T, interpolation='none', aspect='auto', vmin=-1.2, vmax=1.2, cmap='viridis')
    plt.xlim(rng[0]*nt, rng[1]*nt)


def plot_fc(ax, x):
    nt, n = x.shape
    plt.sca(ax)

    plt.imshow(np.corrcoef(x.T), interpolation='none', vmin=-1, vmax=1, cmap='bwr_r')
    plt.xticks(np.r_[0:n])
    plt.yticks(np.r_[0:n])

def plot_histogram(ax, x1, x2, rng, nbins=20, ticks=True, log=False, show_distance=False):
    plt.sca(ax)

    plt.hist(x1, bins=np.linspace(rng[0], rng[1], nbins), density=True, color=color_dat)
    plt.hist(x2, bins=np.linspace(rng[0], rng[1], nbins), density=True, alpha=0.5, color=color_sim)
    plt.grid()

    if log:
        plt.yscale('log')

    if not ticks:
        plt.xlabel("")
        ax.set_xticklabels([])

    if show_distance:
        ws = wasserstein_distance(x1[:,None], x2[:,None], nbins=100, cutoff=3.)
        plt.text(x=0.025, y=0.9, s=f'Dwass={ws:.3f}', transform=ax.transAxes, ha='left', va='center', fontsize=8)

    


def plot_freq(ax, x1, x2, log=False, nperseg=1024, lim=None, ticks=True, show_distance=False):
    f1, pxx1 = sig.welch(x1, fs=1, nperseg=nperseg)
    f2, pxx2 = sig.welch(x2, fs=1, nperseg=nperseg)

    plt.plot(f1, pxx1, '-', label='data', color=color_dat)
    plt.plot(f2, pxx2, '-', label='sim', color=color_sim)
    plt.grid()
    plt.xlabel("Frequency [Hz]")

    if log:
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim(bottom=1e-3)
        if lim is not None:
            plt.xlim(lim)

    else:
        plt.xlim((f1[0], f1[-1]) if lim is None else lim)

    if not ticks:
        plt.xlabel("")
        ax.set_xticklabels([])

    if show_distance:
        dh = spectra_hellinger(x1[:,None], x2[:,None], sigma=2.)
        plt.text(x=0.975, y=0.9, s=f'Dfreq={dh:.3f}', transform=ax.transAxes, ha='right', va='center', fontsize=8)


def plot_isi(ax1, ax2, x1, x2, index=0, height=2, prominence=1, lims='ecg'):

    peaks1 = sig.find_peaks(x1[:,index], height=height, prominence=1.)[0]
    peaks2 = sig.find_peaks(x2[:,index], height=height, prominence=1.)[0]

    isi1 = peaks1[1:] - peaks1[:-1]
    isi2 = peaks2[1:] - peaks2[:-1]

    plt.figure(figsize=(14,4))
    plt.plot(x1[:,index], color=color_dat)
    plt.plot(x2[:,index], color=color_sim)
    plt.xlim(13000, 14000)
    plt.scatter(peaks1, x1[peaks1,index], color='r', s=20, ec=color_dat)
    plt.scatter(peaks2, x2[peaks2,index], color='r', s=20, ec=color_sim)

    disi = stats.wasserstein_distance(isi1, [np.mean(isi1)])

    mu = np.mean(isi1)

    if   lims == 'ecg':    lo,hi = 0.5*mu, 1.5*mu
    elif lims == 'neuron': lo,hi = 0, 1.2*np.max(isi1)

    plt.sca(ax1)
    h1 = plt.hist(isi1, bins=np.linspace(lo, hi, 61), density=True, color=color_dat, alpha=0.5)
    h2 = plt.hist(isi2, bins=np.linspace(lo, hi, 61), density=True, color=color_sim, alpha=0.5)

    xisi = np.linspace(lo, hi, 601)

    if len(isi1) > 0:
        k1 = stats.gaussian_kde(isi1)
        plt.plot(xisi, k1(xisi), color=color_dat, lw=2)

    if len(isi2) > 0:
        k2 = stats.gaussian_kde(isi2)
        plt.plot(xisi, k2(xisi), color=color_sim, lw=2)

    plt.xlim(lo,hi)

    nlo1 = np.sum(isi1 < lo)
    nlo2 = np.sum(isi2 < lo)
    nhi1 = np.sum(isi1 > hi)
    nhi2 = np.sum(isi2 > hi)
    plt.text(0.03, 0.9, f"dat: ←{nlo1:<3d} →{nhi1:<3d} ({len(isi1)})", transform=ax1.transAxes, family='monospace')
    plt.text(0.03, 0.8, f"sim: ←{nlo2:<3d} →{nhi2:<3d} ({len(isi2)})", transform=ax1.transAxes, family='monospace')
    plt.ylim(0, 1.2*np.max(h1[0]))

    plt.sca(ax2)

    val1 = x1[peaks1,index]
    val2 = x2[peaks2,index]
    mu = np.mean(val1)
    sd = np.std(val1)

    if lims == 'ecg':      lo,hi = 0.8*mu, 1.2*mu
    elif lims == 'neuron': lo,hi = mu-5*sd, mu+5*sd

    h1 = plt.hist(val1, bins=np.linspace(lo, hi, 61), density=True, color=color_dat, alpha=0.5, orientation='horizontal')
    h2 = plt.hist(val2, bins=np.linspace(lo, hi, 61), density=True, color=color_sim, alpha=0.5, orientation='horizontal')

    xval = np.linspace(lo, hi, 601)

    if len(val1) > 0:
        k1 = stats.gaussian_kde(val1)
        plt.plot(k1(xval), xval, color=color_dat, lw=2)

    if len(val2) > 0:
        k2 = stats.gaussian_kde(val2)    
        plt.plot(k2(xval), xval, color=color_sim, lw=2)

    plt.ylim(lo,hi)
    plt.xlim(0, 1.2*np.max(h1[0]))

    nlo1 = np.sum(val1 < lo)
    nlo2 = np.sum(val2 < lo)
    nhi1 = np.sum(val1 > hi)
    nhi2 = np.sum(val2 > hi)
    plt.text(0.03, 0.9, f"dat:  ↓{nlo1:<3d} ↑{nhi1:<3d}", transform=ax2.transAxes, family='monospace')
    plt.text(0.03, 0.8, f"sim:  ↓{nlo2:<3d} ↑{nhi2:<3d}", transform=ax2.transAxes, family='monospace')



def plot_details_old(datafile, simfile, outfile, dataset):
    xdat = np.load(datafile)['x_test'][0,:,:]
    xsim = np.load(simfile)[:,:]
    # nt, n = xdat.shape
    nt = xdat.shape[0]
    n = min(xdat.shape[1], xsim.shape[1])
    xdat = xdat[:,0:n]
    xsim = xsim[:,0:n]    

    fig = plt.figure(figsize=(16,2*n+4), dpi=200)

    gs = gridspec.GridSpec(ncols=3, nrows=2, left=0.07, right=0.98, top=0.98, bottom=0.5, width_ratios=[2,2,1])

    plot_timeseries_im(plt.subplot(gs[0,0]), xdat, (0,1))
    plt.ylabel("Data\n", fontsize=16)

    plot_timeseries_im(plt.subplot(gs[1,0]), xsim, (0,1))
    plt.ylabel("Sim\n", fontsize=16)

    plot_timeseries_lines(plt.subplot(gs[0,1]), xdat, 'dat', (0.4,0.5))
    plot_timeseries_lines(plt.subplot(gs[1,1]), xsim, 'sim', (0.4,0.5))

    if n > 1:
        plot_fc(plt.subplot(gs[0,2]), xdat)
        plot_fc(plt.subplot(gs[1,2]), xsim)


    gs2 = gridspec.GridSpec(ncols=4, nrows=n, left=0.05, right=0.7, top=0.4, bottom=0.05, wspace=0.2)

    rng = list(zip(np.min(xdat, axis=0), np.max(xdat, axis=0)))
    for i in range(n):
        plot_histogram(plt.subplot(gs2[i,0]), xdat[:,i], xsim[:,i], rng[i], nbins=100, ticks=(i==n-1), show_distance=True)
        plot_histogram(plt.subplot(gs2[i,1]), xdat[:,i], xsim[:,i], rng[i], nbins=100, ticks=(i==n-1), log=True)


    for i in range(n):
        plot_freq(plt.subplot(gs2[i,2]), xdat[:,i], xsim[:,i], log=False, lim=(0., 0.1), ticks=(i==n-1))
        plot_freq(plt.subplot(gs2[i,3]), xdat[:,i], xsim[:,i], log=True, ticks=(i==n-1))

    if dataset in ['ecg', 'neuron']:
        gs3 = gridspec.GridSpec(ncols=2, nrows=1, left=0.74, right=0.99, top=0.4, bottom=0.05, wspace=0.3)
        height, prom = (2, 1) if dataset == 'ecg' else (1, 0.5)
        plot_isi(plt.subplot(gs3[0,0]), plt.subplot(gs3[0,1]), xdat, xsim,
                 height=height, prominence=prom, lims=dataset)

    plt.savefig(outfile)


def _generate_longterm(model, x, config, noise):
    n_embed = config['n_embed']
    max_nt = config['max_nt']
    variables = config['variables']

    x = x[None,:,:]
    nt = min(max_nt, x.shape[1] - n_embed)

    x_embed = torch.tensor(x[:, 0:n_embed, :])
    z0 = model.get_latent_state_last(x_embed)

    xdat = x[0,n_embed:n_embed+nt][:,variables]
    xsim = model.sample(z0, nt, noise=True).detach().numpy()[0,1:][:,variables]
   
    return xdat, xsim



def plot_predictions(ax, x, model, config, noise):
    x = x[None,:,:]

    variables = config['variables']
    n_embed  = config['n_embed']
    n_pred = max(config['pe_n'])
    nsamples = config['n_samples'] if noise else 1

    T = 1000

    plt.sca(ax)

    xdat = x[0,n_embed:n_embed+T]

    plt.plot(np.r_[n_embed:n_embed+T], xdat[:,0])

    for t in np.r_[n_embed:n_embed+T:n_pred+20]:
        x_embed = torch.tensor(x[:,t-n_embed:t,:])

        z0 = model.get_latent_state_last(x_embed, n=nsamples)
        
        xsim = model.sample(z0, n_pred, nsamples=1, noise=noise)[:,1:,:].detach().numpy()

        plt.plot(np.r_[t:t+n_pred], xsim[:,:,0].T, color='k', lw=0.2)

    plt.xlim(n_embed,n_embed+T)
    plt.ylim(np.min(xdat), np.max(xdat))

    

def plot_details(data_file, model_format, config_file, model_file, eval_config, outfile, epoch=None):

    nplot = 2000

    x = utils.load_datafile(data_file).astype(np.float32)[0]

    n = len(eval_config['variables'])
    fig = plt.figure(figsize=(16,2*n+6), dpi=200)

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
        plt.savefig(outfile)
        plt.close()
        return
    
    gs = gridspec.GridSpec(ncols=3, nrows=2, left=0.07, right=0.98, top=0.98, bottom=0.65, width_ratios=[2,2,1])

    # Plot longterm behavior
    xdat, xsim = _generate_longterm(model, x, eval_config, noise=True)
    plot_timeseries_lines(plt.subplot(gs[0,0]), xdat[0:nplot,:], 'dat', (0.0,1.0))
    plot_timeseries_lines(plt.subplot(gs[1,0]), xsim[0:nplot,:], 'sim', (0.0,1.0))
    
    gs2 = gridspec.GridSpec(ncols=4, nrows=n, left=0.05, right=0.7, top=0.55, bottom=0.35, wspace=0.2)

    rng = list(zip(np.min(xdat, axis=0), np.max(xdat, axis=0)))
    for i in range(n):
        plot_histogram(plt.subplot(gs2[i,0]), xdat[:,i], xsim[:,i], rng[i], nbins=100, ticks=(i==n-1), show_distance=True)
        plot_histogram(plt.subplot(gs2[i,1]), xdat[:,i], xsim[:,i], rng[i], nbins=100, ticks=(i==n-1), log=True)

    for i in range(n):
        plot_freq(plt.subplot(gs2[i,2]), xdat[:,i], xsim[:,i], log=False, lim=(0., 0.1), ticks=(i==n-1), show_distance=True)
        plot_freq(plt.subplot(gs2[i,3]), xdat[:,i], xsim[:,i], log=True, ticks=(i==n-1))        


    variant = eval_config['variant']
    if variant in ['ecg']:
        gs3 = gridspec.GridSpec(ncols=2, nrows=1, left=0.74, right=0.99, top=0.55, bottom=0.35, wspace=0.3)
        height, prom = (2, 1) if variant == 'ecg' else (1, 0.5)
        plot_isi(plt.subplot(gs3[0,0]), plt.subplot(gs3[0,1]), xdat, xsim,
                 height=height, prominence=prom, lims=variant)
    
    # Plot predictions
    plot_predictions(plt.subplot(gs[0:2,1:3]), x, model, eval_config, noise=True)

    # Plot convergence    
    if model_format == 'dsrn':
        gs4 = gridspec.GridSpec(ncols=1, nrows=1, left=0.05, right=0.25, top=0.3, bottom=0.05)
        plt.subplot(gs4[0,0])
        logfile = os.path.join(os.path.dirname(model_file), '..', 'log.txt')
        df = pd.read_csv(logfile, index_col=0, delimiter='\s+')

        plt.plot(df.index, df.loss_train, lw=0.5, color='tab:blue')
        plt.plot(df.index, df.loss_test,  lw=0.2, color='tab:orange')
        plt.ylim(np.min(df.loss_train), df.loss_train[500])
        plt.grid()
        plt.axvline(epoch, color='k', ls='--')

    plt.savefig(outfile)
    plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot results')
    parser.add_argument('dataset')
    parser.add_argument('datafile')
    parser.add_argument('simfile')
    parser.add_argument('outfile')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed')
    args = parser.parse_args()

    utils.seed_all(args.seed)
    plot_details_old(args.datafile, args.simfile, args.outfile, args.dataset)