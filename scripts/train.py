
import argparse
import yaml
import os
import sys

import numpy as np

import torch


sys.path.append(".")
from dsrn import models, plotting, utils



class ArgConstant():
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __call__(self, i):
        return self.value


class ArgLinClip():
    def __init__(self, name, value, tmax, init_value=0., t0=0, dtype='float'):
        self.name = name

        self.a = torch.tensor((value - init_value) / (tmax - t0))
        self.b = torch.tensor((tmax*init_value - t0*value) / (tmax - t0))
        self.minval = torch.tensor(min(init_value, value))
        self.maxval = torch.tensor(max(init_value, value))
        self.dtype = dtype

    def __call__(self, i):
        val = torch.clamp(self.a*i + self.b, self.minval, self.maxval)

        if self.dtype == 'int':
            val = int(val)

        return val


def set_variable_args(conf):
    args = []
    for k, c in conf.items():
        if isinstance(c, dict):
            method = c.pop('method')
            if method == 'constant':
                arg = ArgConstant(k, **c)
            elif method == 'linear':
                arg = ArgLinClip(k, **c)
        else:
            arg = ArgConstant(k,  c)
        
        args.append(arg)

    return args


def model_loss(model, x, args, iter):
    arguments = {arg.name: arg(iter) for arg in args}
    return model.loss(x, **arguments)


def train_model(config, gpu=None):
    utils.seed_all(config['training']['seed'])

    logfile = config['output']['logfile']
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    os.makedirs(os.path.dirname(config['output']['save_direc']), exist_ok=True)

    # Save config
    with open(os.path.join(os.path.dirname(logfile), "config.yaml"), 'w') as fh:
        yaml.dump(config, fh)

    if gpu is not None and torch.cuda.is_available() and (gpu < torch.cuda.device_count()):
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.set_default_device(device)

    model_name = config['model'].pop('model', 'dsrn')

    data = np.load(config['data']['path'])
    batch_size = config['data']['batch_size']

    variables = None if ('use_vars' not in config['data']) else config['data']['use_vars']
    dataset_train = utils.TsDataset(data['x_train'], chunk_size=config['data']['chunk_size'],
                                    subsample=config['data']['subsample'],
                                    variables=variables, overlap_steps=config['data'].get('overlap', None))
    dataset_test  = utils.TsDataset(data['x_test'],  chunk_size=config['data']['chunk_size'],
                                    subsample=config['data']['subsample'],
                                    variables=variables, overlap_steps=config['data'].get('overlap', None))
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                                   generator=torch.Generator(device=device))
    dataloader_test  = torch.utils.data.DataLoader(dataset_test,  batch_size=batch_size, shuffle=True,
                                                   generator=torch.Generator(device=device))

    # Create model    
    if model_name == 'dsrn':
        model = models.DPDSR(**config['model'])
    elif model_name == 'dkf':
        model = models.DKF(**config['model'])
    elif model_name == 'arlstm':
        model = models.ARLSTMModel(**config['model'])
    elif model_name == 'rssm':
        model = models.RSSM(**config['model'])
        model = torch.compile(model)
    else:
        raise ValueError(f"Unknown model {model_name}")

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Optimizer for causal encoder
    if model.causal_encoder is not None:
        opt2 = torch.optim.Adam(model.causal_encoder.parameters(),
                                lr=config['training']['learning_rate'])

    sched = None
    if 'lr_milestones' in config['training']:
        milestones = config['training']['lr_milestones']
        gamma = config['training']['lr_gamma']
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)

        if model.causal_encoder is not None:
            sched2 = torch.optim.lr_scheduler.MultiStepLR(opt2, milestones=milestones, gamma=gamma)

    fh = open(logfile, 'w')
    fh.write(f"Iter   {'loss_train':16s} {'loss_test':16s} {'loss_train_it':16s} {'loss_test_it':16s}"
             f"{'loss_train_cge':16s} {'loss_test_cge':16s}\n")

    args = set_variable_args(config['training'].get('args', {}))
    n_iter = config['training']['n_iter'] + 1
    iters_per_epoch = len(dataloader_train)

    iteration = 0
    x_example = dataset_test[np.random.choice(len(dataset_test), batch_size, replace=False)]
    losses = np.zeros((n_iter+1,4))

    while iteration <= n_iter:
        iter_train = iter(dataloader_train)
        iter_test  = iter(dataloader_test)

        for i in range(iters_per_epoch):
            if iteration > n_iter:
                break

            # Training
            model.train()
            x = next(iter_train)
            x = x.to(device)
            opt.zero_grad()
            loss = model_loss(model, x, args, iteration)
            loss.backward()

            losses[iteration,0] = loss.detach().cpu().numpy()

            if config['training']['gradient_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])

            opt.step()
            if sched:
                sched.step()

            if model.causal_encoder is not None:
                celoss = model.ce_loss(x)
                celoss.backward()
                opt2.step()
                if sched2:
                    sched2.step()
                losses[iteration,2] = celoss.detach().cpu().numpy()

            # Test
            model.eval()
            x = next(iter_test)
            x = x.to(device)
            with torch.no_grad():
                loss_test = model_loss(model, x, args, iteration)
                losses[iteration,1] = loss_test.detach().cpu().numpy()

                if model.causal_encoder is not None:
                    celoss_test = model.ce_loss(x)
                    losses[iteration,3] = celoss_test.detach().cpu().numpy()
            
            # Outputs            
            loss_ma = np.mean(losses[max(0,iteration+1-iters_per_epoch):iteration+1], axis=0)

            fh.write(f"{iteration:6d} {loss_ma[0]:16.8f} {loss_ma[1]:16.8f}" 
                     f" {losses[iteration,0]:16.8f} {losses[iteration,1]:16.8f}"
                     f" {losses[iteration,2]:16.8f} {losses[iteration,3]:16.8f}\n")
            fh.flush()

            if iteration % config['output']['plot_every'] == 0:
                filename = os.path.join(config['output']['plot_direc'], f"samples_{iteration:06d}.png")
                plotting.plot_samples(model, x_example, filename)

            if iteration % config['output']['save_every'] == 0:
                torch.save(model.state_dict(), os.path.join(config['output']['save_direc'], f"model_{iteration:06d}.pth"))

            iteration += 1

    fh.close()


def correct_old_config(config):
    if 'forcing_interval' in config['training']:
        fi = config['training'].pop('forcing_interval')
        config['training']['args'] = {'forcing_interval': fi}



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('configfile')
    parser.add_argument('-g', '--gpu',   type=int, help='If available, use this GPU.')
    parser.add_argument('-c', '--cores', type=int, default=1, help='Number of cores to use.')
    args = parser.parse_args()

    torch.set_num_threads(args.cores)
    torch.set_num_interop_threads(args.cores)

    config = utils.load_config_file(args.configfile)

    correct_old_config(config)

    train_model(config, args.gpu)



