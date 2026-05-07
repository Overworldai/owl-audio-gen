import time

import torch
from torch import nn


def freeze(module : nn.Module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze(module : nn.Module):
    for param in module.parameters():
        param.requires_grad = True

class Timer:
    def reset(self):
        self.start_time = time.time()

    def hit(self):
        return time.time() - self.start_time

def int_to_tuple(x):
    """
    Safely turns everything into a tuple, whether it be a list or a fake yaml list
    """
    if x is None:
        return None
    elif isinstance(x, int):
        return (x,x)
    elif isinstance(x, tuple) or isinstance(x, list):
        return x
    else:
        try:
            return tuple(x)
        except:
            return [int(i) for i in x]

def get_tokens(config):
    h,w = int_to_tuple(config.sample_size)
    p_y, p_x = int_to_tuple(config.patch_size)
    return h // p_y * w // p_x

def get_patch_content(config):
    p_y, p_x = int_to_tuple(config.patch_size)
    return p_y * p_x * config.channels

def get_patching_info(config):
    """
    Returns all info related to patching:
    1. p_y
    2. p_x
    3. n_p_y
    4. n_p_x
    5. patch content
    """
    h,w = int_to_tuple(config.sample_size)
    p_y, p_x = int_to_tuple(config.patch_size)
    return p_y, p_x, h // p_y, w // p_x, get_patch_content(config)

def versatile_load(path):
    ckpt = torch.load(path, map_location = 'cpu', weights_only=False)
    if 'ema' not in ckpt and 'model' not in ckpt:
        return ckpt
    elif 'ema' in ckpt:
        ckpt = ckpt['ema']
        key_list = list(ckpt.keys())
        ddp_ckpt = False
        for key in key_list:
            if key.startswith("ema_model.module."):
                ddp_ckpt = True
                break
        if ddp_ckpt:
            prefix = 'ema_model.module.'
        else:
            prefix = 'ema_model.'
    elif 'model' in ckpt:
        ckpt = ckpt['model']
        key_list = list(ckpt.keys())
        ddp_ckpt = False
        for key in key_list:
            if key.startswith("module."):
                ddp_ckpt = True
        if ddp_ckpt:
            prefix = 'module.'
        else:
            prefix = None

    if prefix is None:
        return ckpt
    else:
        ckpt = {k[len(prefix):] : v for (k,v) in ckpt.items() if k.startswith(prefix)}

    return ckpt

def prefix_filter(ckpt, prefix):
    return {k[len(prefix):] : v for (k,v) in ckpt.items() if k.startswith(prefix)}

def find_unused_params(model):
    for name, param in model.named_parameters():
        if '.ema.' in name:
            continue
        if param.grad is None:
            print(f"Parameter with no gradient: {name}")
