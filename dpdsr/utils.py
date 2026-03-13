import yaml
import os

import numpy as np
import torch

from . import models


class TsDataset(torch.utils.data.Dataset):
    """
    Dataset for chunked time series with optional overlap.

    Args:
        X (np.ndarray): Input array of shape (batch, time, features).
        chunk_size (int, optional): Length of each chunk. If None or larger than sequence, no chunking.
        subsample (int, optional): Subsampling factor along the time axis.
        variables (list or np.ndarray, optional): Indices of variables/features to select.
        overlap_ratio (float, optional): Overlap as a ratio of chunk_size (0 <= overlap_ratio < 1). Default: None.
        overlap_steps (int, optional): Overlap as a fixed number of steps. If negative, overlap_steps = chunk_size - abs(overlap_steps). Default: None.
            Only one of overlap_ratio or overlap_steps should be set (the other must be None).

    Raises:
        ValueError: If both overlap_ratio and overlap_steps are set, or if overlap is invalid.
    """
    def __init__(self, X, chunk_size=None, subsample=1, variables=None, overlap_ratio=None, overlap_steps=None):
        super().__init__()
        n = X.shape[1]

        # Validate overlap arguments
        if overlap_ratio is not None and overlap_steps is not None:
            raise ValueError("Specify only one of overlap_ratio or overlap_steps (not both).")
        if overlap_ratio is not None:
            if not (0 <= overlap_ratio < 1):
                raise ValueError("overlap_ratio must be in [0, 1).")
            if chunk_size is None:
                raise ValueError("chunk_size must be set when using overlap_ratio.")
            overlap = int(chunk_size * overlap_ratio)
        elif overlap_steps is not None:
            if chunk_size is None:
                raise ValueError("chunk_size must be set when using overlap_steps.")
            if overlap_steps < 0:
                overlap = chunk_size - abs(overlap_steps)
            else:
                overlap = overlap_steps
        else:
            overlap = 0

        if chunk_size is None or chunk_size > n:
            Xs = X
        else:
            step = chunk_size - overlap
            if step <= 0:
                raise ValueError("overlap must be less than chunk_size.")
            chunks = []
            for start in range(0, n - chunk_size + 1, step):
                chunks.append(X[:, start:start + chunk_size])
            Xs = np.concatenate(chunks, axis=0)

        X = torch.tensor(Xs[:, ::subsample], dtype=torch.float32)
        if variables is not None:
            X = X[:, :, variables]

        self.X = X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]

    def nvar(self):
        return self.X.shape[2]




def replace_variables(config, variables=None):
    if variables is None:
        variables = {}

    if 'variables' in config:
        variables.update(config['variables'])

    for key, value in config.items():
        if key == "variables":
            continue

        if type(value) is dict:
            replace_variables(value, variables)

        if type(value) is str:
            config[key] = value.format(**variables)



def load_config_file(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    replace_variables(config)


    return config

def load_datafile(data_file):
    data = np.load(data_file)

    try:
        # npz format
        x = data['x_test']
    except IndexError:
        # npy format
        x = data[None,:,:]

    return x


def get_dataloader(config, exp_dir='.', part='train', shuffle=False, batch_size=1, device='cpu'):
    data = np.load(os.path.join(exp_dir, config['data']['path']))
    variables = None if ('use_vars' not in config['data']) else config['data']['use_vars']
    
    dataset = TsDataset(data[f'x_{part}'], chunk_size=config['data']['chunk_size'],
                        subsample=config['data']['subsample'],
                        variables=variables, overlap_steps=config['data'].get('overlap', None))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                             generator=torch.Generator(device=device))

    return dataloader



def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


    
def load_model_dsrn(config_file, model_file):
    config = load_config_file(config_file)
    model_type = config['model'].pop('model', 'dsrn')

    if model_type == 'dsrn':
        model = models.DPDSR(**config['model'])
    elif model_type == 'dkf':
        model = models.DKF(**config['model'])
    elif model_type == 'arlstm':
        model = models.ARLSTMModel(**config['model'])
    elif model_type == 'rssm':
        model = models.RSSM(**config['model'])
        model = torch.compile(model)
    else:
        raise ValueError(f"Unknown model {model_type}")
    
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model.eval()

    return model

def load_model_gtf(model_file):
    model = models.ClippedShallowPLRNN.from_file(model_file)
    model.eval()
    
    return model