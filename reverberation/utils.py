"""
@Author: Conghao Wong
@Date: 2024-12-16 11:00:09
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-09 20:12:49
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from qpid.utils import ROOT_TEMP_DIR, dir_check

TEMP_DIR = dir_check(os.path.join(ROOT_TEMP_DIR, 'Reverberation'))


def show_kernel(k: torch.Tensor,
                name: str,
                partitions: int,
                obs_periods: int,
                pred_periods: int,
                normalize: int | bool = True):

    # Kernel shape: (batch, steps, new_steps)
    _k: np.ndarray = k.cpu().numpy()
    _k = np.mean(_k, axis=-3)   # (steps, new_steps)

    # Normalize on EACH OUTPUT STEP
    if normalize:
        _min = np.min(_k, axis=-2, keepdims=True)
        _max = np.max(_k, axis=-2, keepdims=True)
        _k = (_k - _min)/(_max - _min)
    else:
        _k = _k ** 2

    _k = np.reshape(_k, [obs_periods, partitions, pred_periods])

    # Display kernels on each new step
    title = f'Kernel {name}'
    plt.close(title)

    fig = plt.figure(title)

    for _j in range(pred_periods):
        ax = fig.add_subplot(1, pred_periods, _j + 1)
        ax.imshow(_k[:, :, _j])
        ax.axis('off')

    plt.show()
