"""
Startup script to run in the beggining of every Jupyter Notebook for the competition
- Import common libraries
- Jupyter Notebook Setup: autoreload, display all, add to sys.path
- Import common functions, classes & constants
- Import competition specific functions / constants
"""

# Start with Flax Imports because it's most brittle
import flax
import jax
import optax

import jax.numpy as jnp
import flax.linen as nn

from flax.traverse_util import flatten_dict, unflatten_dict
from flax.jax_utils import replicate, unreplicate
from flax.core.frozen_dict import unfreeze, freeze
from flax.training.common_utils import shard
from flax.training import train_state

from jax.experimental import maps, PartitionSpec
from jax.experimental.pjit import pjit
from flax.linen import partitioning as nn_partitioning


## Boilerplate Code for TPU Setup ##
import requests
import os

def kaggle_tpu_setup():
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    url = 'http:' + os.environ['TPU_NAME'].split(':')[1] + ':8475/requestversion/tpu_driver_nightly'
    resp = requests.post(url)
    jax.config.FLAGS.jax_xla_backend = 'tpu_driver'
    jax.config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
    # jax.config.update('jax_default_matmul_precision', 'bfloat16')

if 'TPU_NAME' in os.environ:
    print('Attempting to connect with TPU:', os.environ['TPU_NAME'])
    kaggle_tpu_setup()

print('---------- Available Devices ----------')
print(jax.devices())
print('---------------------------------------')


# Commonly Used Libraries
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from functools import partial
from termcolor import colored
from time import time, sleep
from tqdm.auto import tqdm
from pathlib import Path
import pandas as pd
import numpy as np

import dataclasses
import itertools
import warnings
import sklearn
import random
import pickle
import scipy
import yaml
import math
import sys
import os
import re

import scipy.stats
tqdm.pandas()
import wandb

# Huggingface Imports
import transformers 
import datasets

# IPython Imports
from IPython.core.magic import register_line_cell_magic
from IPython import get_ipython, display
from IPython.display import FileLink

from omegaconf import OmegaConf


# Setup Jupyter Notebook
def _setup_jupyter_notebook():
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = 'all'
    ipython = get_ipython()
    ipython.magic('matplotlib inline')
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
_setup_jupyter_notebook()


# Hyperparameters Magic Command
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_yaml(filename):
    with open(filename, 'r') as stream:
        return AttrDict(yaml.safe_load(stream))

@register_line_cell_magic
def hyperparameters(hp_var_name, cell):
    with open('experiment.yaml', 'w') as f:
        f.write(cell)
    HP = OmegaConf.load('experiment.yaml')
    get_ipython().user_ns[hp_var_name] = HP

WORKING_DIR = Path('/kaggle/working')