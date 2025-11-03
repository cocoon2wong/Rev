"""
@Author: Conghao Wong
@Date: 2024-12-16 11:00:09
@LastEditors: Conghao Wong
@LastEditTime: 2025-08-25 16:00:23
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline

from qpid.utils import ROOT_TEMP_DIR, dir_check

from .__layers import compute_inverse_kernel

TEMP_DIR = dir_check(os.path.join(ROOT_TEMP_DIR, 'Reverberation'))


def show_kernel(k: torch.Tensor | None,
                name: str,
                partitions: int,
                obs_periods: int,
                pred_periods: int,
                partition_label='Partition',
                figs_per_row=4,
                show_axis_labels=True):

    # Do nothing if `k` is not a Tensor
    if not isinstance(k, torch.Tensor):
        return

    # For `batch_size > 1` cases
    _k: np.ndarray = k.cpu().numpy()
    _k = np.mean(_k, axis=0)   # (steps, new_steps)

    # Separate partitions
    # Final kernel shape: (steps, partitions, new_steps)
    _k = np.reshape(_k, [obs_periods, partitions, pred_periods])

    # Display curves on each partition
    title = f'Kernel {name}'
    plt.close(title)
    fig = plt.figure(title)

    for _p in range(partitions):
        rows = int(np.ceil(partitions/figs_per_row))
        cols = min(figs_per_row, partitions)
        ax = fig.add_subplot(rows, cols, _p + 1)

        # Get data of this partition
        _matrix = _k[:, _p, :] ** 2      # (obs, pred)
        _matrix = _matrix / np.sum(_matrix, axis=0, keepdims=True)

        for _o in range(obs_periods):
            _y = _matrix[_o]
            _x = np.arange(len(_y)) + 1 + obs_periods

            # Draw as reverberation curves
            _x_interp = np.linspace(_x[0], _x[-1], 100)
            _y_interp = make_interp_spline(_x, _y)(_x_interp)

            ax.plot(_x_interp, _y_interp, label=f'Step {_o+1}')
            ax.plot(_x, _y, 'x', color='black')

            # Save meta data (txt)
            _path = os.path.join(TEMP_DIR, f'meta_{title}_p{_p+1}_o{_o+1}.txt')
            np.savetxt(_path, _y)

        # Draw the baseline
        ax.plot(_x, (1/obs_periods) * np.ones_like(_x),
                color='grey', linestyle='--', label='AVG')

        ax.set_ylim(np.min(_matrix) - 0.1, np.max(_matrix) + 0.1)

        if show_axis_labels:
            ax.legend()
            ax.set_xlabel('Future steps (t)')
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if partitions > 1:
            ax.set_title(f'{partition_label} {_p+1}')

    plt.show()


def vis_kernels(R: torch.Tensor | None,
                G: torch.Tensor | None,
                name: str,
                setting: int = 1,
                partitions: int = 1,
                selected_partition: int = 1):

    from .utils import show_kernel

    if (setting >= 1) and (isinstance(R, torch.Tensor)):
        # Shape of R: (batch, past_steps, future_steps)
        # or (batch, past_steps * partitions, future_steps)
        [steps, T_f] = R.shape[-2:]
        T_h = steps // partitions
        show_kernel(R, f'{name}: Reverberation Kernel',
                    partitions, T_h, T_f,
                    show_axis_labels=False if partitions > 1 else True)

    if ((setting >= 2) and (isinstance(R, torch.Tensor))
            and (isinstance(G, torch.Tensor))):
        # Shape of G: (batch, past_steps, K)
        # NOTE: G kernels will be displayed along with R kernels
        # It only accepts `partitions == 1` cases when visualizing

        postfix = ''

        if partitions > 1:
            if ((p := selected_partition) > partitions):
                return

            # Current R shape: (batch, past_steps, future_steps)
            [steps, T_f] = R.shape[-2:]
            K = G.shape[-1]
            T_h = steps // partitions

            # Reshape and select one social partition
            R = torch.reshape(R, [-1, T_h, partitions, T_f])
            G = torch.reshape(G, [-1, T_h, partitions, K])
            R = R[..., p-1, :]
            G = G[..., p-1, :]

            # Title prefix
            postfix = f' (on Social Partition {p})'

        T_f = R.shape[-1]
        T_h = G.shape[-2]
        K_g = G.shape[-1]

        # Multiple G on R
        _R = R[:, :, None, :]   # (batch, past_steps, 1, future_steps)
        _G = G[:, :, :, None]   # (batch, past_steps, K, 1)
        _G = _R * _G            # (batch, past_steps, K, future_steps)
        _G = torch.reshape(_G, [-1, T_h * K_g, T_f])

        show_kernel(_G, f'{name}: Generating Kernel (on R)' + postfix,
                    K_g, T_h, T_f,
                    partition_label='Generation',
                    figs_per_row=10,
                    show_axis_labels=False)

    if setting >= 3:
        if isinstance(R, torch.Tensor):
            R_inv = compute_inverse_kernel(R)
            show_kernel(R_inv, f'{name}: Inverse Reverberation Kernel',
                        partitions, T_f, T_h)

        if isinstance(G, torch.Tensor):
            G_inv = compute_inverse_kernel(G)
            show_kernel(G_inv, f'{name}: Inverse Generating Kernel',
                        partitions, K_g, T_h)
