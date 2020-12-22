"""
When import torchlib, it will make config directory .torchlib under home folder.
It will create torchlib.json if it doesn't exit. It contains
{
    'enable_cuda': True
}
This is used to disable cuda with a GPU machine, especially when training RL models
"""

import errno
import json
import os

import torch

default_config = {
    'enable_cuda': True
}

config_path = os.path.expanduser('~/.torchlib/torchlib.json')

try:
    with open(config_path, 'r') as f:
        config = json.load(f)

except:
    if not os.path.exists(os.path.dirname(config_path)):
        try:
            os.makedirs(os.path.dirname(config_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=4)

    config = default_config

enable_cuda = config['enable_cuda'] and torch.cuda.is_available()
if 'CUDA' in os.environ:
    enable_cuda_env = os.environ['CUDA']
    if enable_cuda_env == 'True':
        enable_cuda = enable_cuda
    elif enable_cuda_env == 'False':
        enable_cuda = False
print('Enable cuda {}'.format(enable_cuda))
