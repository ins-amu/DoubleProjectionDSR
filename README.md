

# Dynamical system reconstruction from partial observations using stochastic dynamics

Code and data associated with the paper titled *Dynamical system reconstruction from partial observations using stochastic dynamics*.
 
The repository is structured as follows:

- `conf/` contains the configuration files for the analyzed problems.
- `data/` contains the datasets used in the study.
- `dsrn/` contains the core code.
- `run/`  is the target directory.
- `scripts/` contains the scripts to train and run the models.


An example configuration file for a single model and training configuration is `conf/example.yaml`. The training of a single model can be ran by
```
python scripts/train.py conf/example.yaml -c1
```

The parameter sweeps are described by the the configuration files following naming scheme `conf/<EXPERIMENT>/<DATASET>-<MODEL>.py`. 
They can be run with Snakemake; the Snakemake worflow (training and evaluation) is described by `Snakefile`. To run the parameter sweeps described by a configuration file execute
```
export CONF="<PATH_TO_CONFIG_FILE>"
snakemake all
```
with appropriate Snakemake settings for your computing environment.

