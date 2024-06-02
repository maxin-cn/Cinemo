import os
import math
import torch
import logging
import subprocess
import numpy as np
import torch.distributed as dist

# from torch._six import inf
from torch import inf
from PIL import Image
from typing import Union, Iterable
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter   
from typing import Dict
import torch_dct

from diffusers.utils import is_bs4_available, is_ftfy_available

import html
import re
import urllib.parse as ul

if is_bs4_available():
    from bs4 import BeautifulSoup

if is_ftfy_available():
    import ftfy

import torch.fft as fft

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


#################################################################################
#                             Testing  Utils                                    #
#################################################################################

def find_model(model_name):
    """
    Finds a pre-trained model
    """
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        
    if "ema" in checkpoint:  # supports checkpoints from train.py
        print('Using ema ckpt!')
        checkpoint = checkpoint["ema"]
    else:
        checkpoint = checkpoint["model"]
        print("Using model ckpt!")
    return checkpoint

def save_video_grid(video, nrow=None):
    b, t, h, w, c = video.shape
    
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = torch.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype=torch.uint8)
    
    # print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    
    return video_grid

def save_videos_grid_tav(videos: torch.Tensor, path: str, rescale=False, nrow=None, fps=8):
    from einops import rearrange
    import imageio
    import torchvision

    b, _, _, _, _ = videos.shape
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=nrow)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    # os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


#################################################################################
#                             MMCV  Utils                                    #
#################################################################################


def collect_env():
    # Copyright (c) OpenMMLab. All rights reserved.
    from mmcv.utils import collect_env as collect_base_env
    from mmcv.utils import get_git_hash
    """Collect the information of the running environments."""
    
    env_info = collect_base_env()
    env_info['MMClassification'] = get_git_hash()[:7]

    for name, val in env_info.items():
        print(f'{name}: {val}')
    
    print(torch.cuda.get_arch_list())
    print(torch.version.cuda)


#################################################################################
#                              DCT Functions                                    #
#################################################################################  

def dct_low_pass_filter(dct_coefficients, percentage=0.3): # 2d [b c f h w]
    """
    Applies a low pass filter to the given DCT coefficients.

    :param dct_coefficients: 2D tensor of DCT coefficients
    :param percentage: percentage of coefficients to keep (between 0 and 1)
    :return: 2D tensor of DCT coefficients after applying the low pass filter
    """
    # Determine the cutoff indices for both dimensions
    cutoff_x = int(dct_coefficients.shape[-2] * percentage)
    cutoff_y = int(dct_coefficients.shape[-1] * percentage)

    # Create a mask with the same shape as the DCT coefficients
    mask = torch.zeros_like(dct_coefficients)
    # Set the top-left corner of the mask to 1 (the low-frequency area)
    mask[:, :, :, :cutoff_x, :cutoff_y] = 1

    return mask

def normalize(tensor):
    """将Tensor归一化到[0, 1]范围内。"""
    min_val = tensor.min()
    max_val = tensor.max()
    normalized = (tensor - min_val) / (max_val - min_val)
    return normalized

def denormalize(tensor, max_val_target, min_val_target):
    """将Tensor从[0, 1]范围反归一化到目标的[min_val_target, max_val_target]范围。"""
    denormalized = tensor * (max_val_target - min_val_target) + min_val_target
    return denormalized

def exchanged_mixed_dct_freq(noise, base_content, LPF_3d, normalized=False):
    # noise dct
    noise_freq = torch_dct.dct_3d(noise, 'ortho')

    # frequency
    HPF_3d = 1 - LPF_3d
    noise_freq_high = noise_freq * HPF_3d

    # base frame dct
    base_content_freq = torch_dct.dct_3d(base_content, 'ortho')

    # base content low frequency
    base_content_freq_low = base_content_freq * LPF_3d

    # mixed frequency
    mixed_freq = base_content_freq_low + noise_freq_high

    # idct
    mixed_freq = torch_dct.idct_3d(mixed_freq, 'ortho')

    return mixed_freq