
import sys
import time
import os
import yaml
import json
import copy
import itertools
from subprocess import Popen
import glob

import pandas as pd
import numpy as np
import torch

from dsrn.evaluate_model import evaluate_model
from dsrn.plotting import plot_details

envvars:
    "CONF",
    "GTF_PATH"

GTF_PATH = os.environ['GTF_PATH']
SCRIPT_PATH = "./scripts"
CORES_PER_RUN = {'dsrn': 1, 'gtf': 2}

sys.path.append(".")

def setnested(dic, keys, val):
    if type(keys) == str:
        dic[keys] = val
    elif len(keys) == 1:
        dic[keys[0]] = val
    else:
        setnested(dic[keys[0]], keys[1:], val)

class ExperimentConfig():
    def __init__(self, filename):
        v = {}
        exec(open(filename).read(), {}, v)

        self.tool = v['tool']

        if type(v['default']) == dict:
            self.default = v['default']
        elif type(v['default']) == str:
            self.default = yaml.safe_load(v['default'])
        else:
            raise ValueError(f"Unexpected type '{type(v['default'])}'")
        
        self.nseeds     = v['nseeds']
        self.variants   = v['variants']
        self.evaluation = v['evaluation']
        self.is_nested_eval = all(isinstance(v, dict) for v in self.evaluation.values())

        if isinstance(v['params'], dict):
            self.params = v['params']
        else:
            self.params = {variant: v['params'] for variant in self.variants.keys()}

        dirs_ = os.path.relpath(filename).split(os.sep)[1:]
        dirs_[-1] = os.path.splitext(dirs_[-1])[0]
        self.path = os.path.join(*dirs_)


    def get_variants(self):
        return self.variants.keys()

    def get_pids(self, variant, fmt=None):
        params = self.params[variant]
        if fmt is None:
            return range(len(params))
        else:
            return [("%"+fmt) % i for i in range(len(params))]

    def get_seeds(self):
        return list(range(self.nseeds))

    def get_path(self):
        return self.path

    def get_parameters(self, variant, pid):
        params = self.params[variant]

        pid = int(pid)
        config = copy.deepcopy(self.default)
        p = self.variants[variant] | params[pid]

        for k, v in p.items():
            setnested(config, k, v)

        return config

    def get_eval_config(self, variant):
        if self.is_nested_eval:
            return self.evaluation[variant]
        else:
            return self.evaluation


expconf = ExperimentConfig(os.environ['CONF'])
path = expconf.get_path()
tool = expconf.tool


def create_config_dsrn(config_file, output_direc, experiment_config, variant, pid, seed):
    config = experiment_config.get_parameters(variant, pid)
    config['variables']['variant'] = f"{variant}_{pid:06d}"
    config['training']['seed'] = seed
    config['output']['logfile']    = os.path.join(output_direc, 'log.txt')
    config['output']['plot_direc'] = os.path.join(output_direc, 'img/')
    config['output']['save_direc'] = os.path.join(output_direc, 'models/')
    with open(config_file, 'w') as fh:
        yaml.dump(config, fh, default_flow_style=False)

def create_config_gtf(config_file, experiment_config, variant, pid, seed):    
    config = experiment_config.get_parameters(variant, pid)
    config['run'] = seed
    config['name'] = f"{variant}_{pid:06d}"
    with open(config_file, 'w') as fh:
        json.dump(config, fh)

def train_dsrn(output, wildcards, threads):
    direc = os.path.dirname(output[0])
    config_file = os.path.join(direc, "conf.yaml")
    create_config_dsrn(config_file, direc, expconf, wildcards.variant, int(wildcards.pid), int(wildcards.seed))
    shell(f"python {SCRIPT_PATH}/train.py {config_file} -c{threads}")

def train_gtf(output, wildcards, threads, log):
    direc = os.path.dirname(output[0])
    config_file = os.path.join(direc, "conf.json")
    create_config_gtf(config_file, expconf, wildcards.variant, int(wildcards.pid), int(wildcards.seed))
    shell(f"julia --project={GTF_PATH} -t {threads} {GTF_PATH}/rungtf.jl {config_file} {direc} > {log}")
    
    # Convert to easily readable npz
    for file in sorted(glob.glob(os.path.join(direc, "checkpoints/model_*.bson"))):
        npz_file = os.path.splitext(file)[0] + ".npz"
        shell(f"julia --project={GTF_PATH} -t {threads} {GTF_PATH}/save_shplrnn.jl -m {file} -o {npz_file}")


def eval_dsrn(input, output, wildcards, threads):
    direc = os.path.dirname(input[0])
    files = sorted(glob.glob(os.path.join(direc, "models/model_*.pth")))
    datafile = expconf.get_parameters(wildcards.variant, int(wildcards.pid))['data']['path']
    log = pd.read_csv(os.path.join(direc, 'log.txt'), sep='\s+', index_col=0)
    config_file = os.path.join(direc, "conf.yaml")

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)

    for modelfile in files:
        epoch = int(os.path.basename(modelfile)[6:-4])
        p = {
            'variant': wildcards.variant,
            'pid':  int(wildcards.pid),
            'seed': int(wildcards.seed),
            'epoch': epoch,
            'loss_train': float(log.loc[epoch, 'loss_train']),
            'loss_test':  float(log.loc[epoch, 'loss_test']),
        }

        evaluate_model(datafile, 'dsrn', config_file, modelfile, p, expconf.get_eval_config(wildcards.variant), output[0], True)
        evaluate_model(datafile, 'dsrn', config_file, modelfile, p, expconf.get_eval_config(wildcards.variant), output[0], False)


def eval_gtf(input, output, wildcards, threads):
    direc = os.path.dirname(input[0])
    files = sorted(glob.glob(os.path.join(direc, "checkpoints/model_*.npz")))
    
    trainfile = expconf.get_parameters(wildcards.variant, wildcards.pid)['path_to_data']
    testfile = trainfile.replace(".train.npy", ".test.npy")

    for modelfile in files:
        epoch = int(os.path.splitext(os.path.basename(modelfile))[0][6:])
        p = {
            'variant': wildcards.variant,
            'pid':  int(wildcards.pid),
            'seed': int(wildcards.seed),
            'epoch': epoch,
            'noise': False
        }

        evaluate_model(testfile, 'gtf', None, modelfile, p, expconf.get_eval_config(wildcards.variant), output[0])


def plot_dsrn(input, output, wildcards, threads):
    datafile = expconf.get_parameters(wildcards.variant, wildcards.pid)['data']['path']
    direc = os.path.dirname(input[0])
    config_file = os.path.join(direc, "conf.yaml")
    epoch = int(wildcards.epoch)
    modelfile = os.path.join(direc, f"models/model_{epoch:06d}.pth")
    evalconf = expconf.get_eval_config(wildcards.variant)

    plot_details(datafile, 'dsrn', config_file, modelfile, evalconf, output[0], epoch)


def plot_gtf(input, output, wildcards, threads):
    direc = os.path.dirname(input[0])

    trainfile = expconf.get_parameters(wildcards.variant, wildcards.pid)['path_to_data']
    testfile = trainfile.replace(".train.npy", ".test.npy")
    epoch = int(wildcards.epoch)
    modelfile = os.path.join(direc, f"checkpoints/model_{epoch}.npz")
    evalconf = expconf.get_eval_config(wildcards.variant)

    plot_details(testfile, 'gtf', None, modelfile, evalconf, output[0])    


localrules: all, alltrain, merge, plot

rule all:
    input: f"run/{path}/results.csv"


rule alltrain:
    input: [f"run/{path}/{variant}_{pid:06d}/{seed:03d}/task.done" for variant in expconf.get_variants()  \
            for pid in expconf.get_pids(variant) for seed in expconf.get_seeds()]


rule train:
    input:
    output: touch(f"run/{path}/{{variant}}_{{pid}}/{{seed}}/task.done")
    log: f"run/{path}/{{variant}}_{{pid}}/{{seed}}/log.txt"
    threads: CORES_PER_RUN[tool]
    run:
        if tool == 'dsrn':
            train_dsrn(output, wildcards, threads)
        elif tool == 'gtf':
            train_gtf(output, wildcards, threads, log)


rule eval:
    input:  f"run/{path}/{{variant}}_{{pid}}/{{seed}}/task.done"
    output: f"run/{path}/{{variant}}_{{pid}}/{{seed}}/results.csv"
    threads: CORES_PER_RUN[tool]
    run:
        if tool == 'dsrn':
            eval_dsrn(input, output, wildcards, threads)
        elif tool == 'gtf':
            eval_gtf(input, output, wildcards, threads)        

rule plot:
    input:  f"run/{path}/{{variant}}_{{pid}}/{{seed}}/task.done"
    output: f"run/{path}/{{variant}}_{{pid}}/{{seed}}/img/res_{{epoch}}.png"
    threads: 1
    run:
        if tool == 'dsrn':
            plot_dsrn(input, output, wildcards, threads)
        elif tool == 'gtf':
            plot_gtf(input, output, wildcards, threads)
        

rule merge:
    input: [f"run/{path}/{variant}_{pid:06d}/{seed:03d}/results.csv" \
            for variant in expconf.get_variants()                    \
            for pid     in expconf.get_pids(variant)                 \
            for seed    in expconf.get_seeds()]
    output: f"run/{path}/results.csv"
    run:
        df = pd.read_csv(input[0], sep=',')
        header = list(df.columns.values)

        with open(output[0], 'w') as fh:
            fh.write(",".join(header) + "\n")
            for filename in input:
                df = pd.read_csv(filename, sep=',')
                for row in df.itertuples(index=False, name=None):
                    line = ",".join(str(x) for x in list(row))
                    fh.write(line + "\n")
