

import argparse
import sys
sys.path.append(".")

import numpy as np
import torch

from dsrn import utils

def generate_data(data_file, config_file, model_file, no_noise=False):

    config = utils.load_config_file(config_file)

    data = np.load(data_file)
    variables = None if ('use_vars' not in config['data']) else config['data']['use_vars']
    dataset_test  = utils.TsDataset(data['x_test'],  chunk_size=None,
                                    subsample=config['data']['subsample'], variables=variables)

    model = utils.load_model_dsrn(config_file, model_file)
    model_name = model._name

    torch.set_grad_enabled(False)

    nsamples = 1
    x = dataset_test[0:nsamples]
    _, nt, _ = x.shape
    ntwarmup = 300
    n_encode = 300

    with_noise = not no_noise
    if model_name == 'dsrn':
        z0 = model.get_latent_state_init(x[:,0:n_encode,:])[:,n_encode//2,:]
        xbar = model.sample(z0, nt=nt+ntwarmup, noise=with_noise, return_latent=False, nsamples=nsamples).detach().cpu().numpy()
    elif model_name == 'dkf':
        z0 = model.get_latent_state((x[:,0:n_encode,:]))[:,n_encode//2,:]
        xbar = model.sample(z0, nt=nt+ntwarmup, nsamples=nsamples, noise=with_noise).detach().cpu().numpy()
    elif model_name == 'arlstm':
        z0 = model.get_latent_state_last(x[:,0:n_encode,:])
        xbar = model.sample(z0, nt, noise=with_noise).detach().cpu().numpy()
    elif model_name == 'rssm':
        z0 = model.get_latent_state(x[:,0:n_encode,:])[:,n_encode//2,:]
        xbar = model.sample(z0, nt=nt+ntwarmup, noise=with_noise, nsamples=nsamples).detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown model {model_name}")

    xbar = xbar[0, ntwarmup:, :]

    torch.set_grad_enabled(True)

    return xbar


def generate_and_save(data_file, config_file, model_file, output_file, no_noise=False):
    xbar = generate_data(data_file, config_file, model_file, no_noise)
    np.save(output_file, xbar)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate data using a trained model')
    parser.add_argument('-d', '--data-path',   type=str, required=True, help='Path to (test) dataset')
    parser.add_argument('-c', '--config-path', type=str, required=True, help='Path to model config')
    parser.add_argument('-m', '--model-path',  type=str, required=True, help='Path to model weights')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='Path to the output')
    parser.add_argument('--no-noise', action='store_true')
    parser.add_argument('-s', '--seed',        type=int, required=False, default=0, help='Seed')
    args = parser.parse_args()

    utils.seed_all(args.seed)
    generate_and_save(args.data_path, args.config_path, args.model_path, args.output_path, args.no_noise)
