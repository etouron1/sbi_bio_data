import random
import numpy as np
from mlxp.launcher import _import_module
import os 
import torch




# def instantiate(args):
#     if isinstance(args,ConfigDict):
#         args = args.to_dict()
#     name = args.pop(class_name)  
#     instance = instance_from_dict(name, args)
#     return instance

def instantiate(module_name):
    #module, name = os.path.splitext(module_name)
    return _import_module(module_name, None)


def assign_device(device):
    """
    Assigns a device for PyTorch based on the provided device identifier.

    Parameters:
    - device (int): Device identifier. If positive, it represents the GPU device
                   index; if -1, it sets the device to 'cuda'; if -2, it sets
                   the device to 'cpu'.

    Returns:
    - device (str): The assigned device, represented as a string. 
                    'cuda:X' if device > -1 and CUDA is available, where X is 
                    the provided device index. 'cuda' if device is -1.
                    'cpu' if device is -2.
    """
    if device >-1:
        device = (
            'cuda:'+str(device) 
            if torch.cuda.is_available() and device>-1 
            else 'cpu'
        )
    elif device==-1:
        device = 'cuda'
    elif device==-2:
        device = 'cpu'
    return device

def get_dtype(dtype):
    """
    Returns the PyTorch data type based on the provided integer identifier.

    Parameters:
    - dtype (int): Integer identifier representing the desired data type.
                   64 corresponds to torch.double, and 32 corresponds to torch.float.

    Returns:
    - torch.dtype: PyTorch data type corresponding to the provided identifier.

    Raises:
    - NotImplementedError: If the provided identifier is not recognized (not 64 or 32).
    """
    if dtype==64:
        return torch.double
    elif dtype==32:
        return torch.float
    else:
        raise NotImplementedError('Unkown type')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



