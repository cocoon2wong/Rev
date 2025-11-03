"""
@Author: Conghao Wong
@Date: 2025-04-15 19:03:22
@LastEditors: Conghao Wong
@LastEditTime: 2025-04-15 20:10:09
@Github: https://cocoon2wong.github.io
@Copyright 2025 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers


class ReverberationTransform(torch.nn.Module):
    """
    Reverberation Transform Layer
    ---
    The reverberation transform layer, which applies the proposed reverberation
    transform on the given sequential representation.

    NOTE: This layer does not contain the trainable reverberation kernels. Please
    train them outside from this class (layer).
    """

    def __init__(self, historical_steps: int,
                 future_steps: int,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.T_h = historical_steps
        self.T_f = future_steps

        self.outer = layers.OuterLayer(self.T_h, self.T_h)

    def forward(self, f: torch.Tensor,
                R: torch.Tensor | torch.nn.Module,
                G: torch.Tensor | torch.nn.Module) -> torch.Tensor:
        """
        The reverberation transform.
        `f` is the representation of a sequential input, with a shape of
        `(..., steps, d)`.

        If any of `R` or `G` is an instance of `torch.nn.Module`, it will
        directly apply that layer on the input `f`, without applying the
        original reverberation transform.
        When `R` or `G` are `torch.nn.Module`s, their input-output shapes
        should satisfy:
        - `G`: `(..., d, T_h, T_h)` -> `(..., d, K_g, T_h)`;
        - `R`: `(..., d, K_g, T_h)` -> `(..., K_g, T_f, d)`.
        """

        # Outer product
        f_t = torch.transpose(f, -1, -2)            # (..., d, T_h)
        f_o = self.outer(f_t, f_t)                  # (..., d, T_h, T_h)

        # Apply the generating kernel
        if isinstance(G, torch.Tensor):
            f = f_o @ G[..., None, :, :]            # (batch, d, T_h, K_g)
            f = torch.transpose(f, -1, -2)          # (batch, d, K_g, T_h)
        elif isinstance(G, torch.nn.Module):
            f = G(f_o)
        else:
            raise ValueError('Illegal value received (Generating Kernel)!')

        if isinstance(R, torch.Tensor):
            # `f` should now has the shape `(batch, d, K_g, T_h)`
            f = f @ R[..., None, :, :]              # (batch, d, K_g, T_f)
            f = torch.transpose(f, -1, -3)          # (batch, T_f, K_g, d)
            f = torch.transpose(f, -2, -3)          # (batch, K_g, T_f, d)
        elif isinstance(R, torch.nn.Module):
            f = R(f)
        else:
            raise ValueError('Illegal value received (Reverberation Kernel)!')

        return f


class MultiStyleGeneratingLayer(torch.nn.Module):
    """
    Generate stochastic predictions using an MSN-like approach.
    NOTE: This layer is used to conduct ablation variations.
    """

    def __init__(self, feature_dim: int, style_channels: int,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.d = feature_dim
        self.K_c = style_channels

        self.ms_fc = layers.Dense(self.d, self.K_c, torch.nn.Tanh)
        self.ms_conv = layers.GraphConv(self.d, self.d)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Input shape: `(..., d, steps, steps)`;
        Return shape: `(..., d, K_g, steps)`.
        """
        f_kernel = torch.transpose(f, -1, -3)       # (..., steps, steps, d)
        kernel = self.ms_fc(f_kernel)               # (..., steps, steps, K_g)
        kernel = torch.transpose(kernel, -1, -2)    # (..., steps, K_g, steps)
        f = self.ms_conv(f_kernel, kernel)          # (..., steps, K_g, d)
        f = torch.transpose(f, -1, -3)              # (..., d, K_g, steps)
        return f


class LinearMappingLayer(torch.nn.Module):
    """
    Prediction trajectories based on the observed features, using direct FC
    layers to build connections from the past to the future.
    NOTE: This layer is used to conduct ablation variations.
    """

    def __init__(self, feature_dim: int,
                 historical_steps: int,
                 future_steps: int,
                 *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)

        self.d = feature_dim
        self.T_h = historical_steps
        self.T_f = future_steps

        self.fc1 = layers.Dense(self.d, self.d, torch.nn.Tanh)
        self.fc2 = layers.Dense(self.T_h, self.T_f)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Input shape: `(..., d, K_g, T_h)`;
        Return shape: `(..., K_g, T_f, d)`.
        """
        f = torch.transpose(f, -1, -3)          # (..., T_h, K_g, d)
        f = torch.transpose(f, -2, -3)          # (..., K_g, T_h, d)
        f = self.fc1(f)                         # (..., K_g, T_h, d)
        f = torch.transpose(f, -1, -2)          # (..., K_g, d, T_h)
        f = self.fc2(f)                         # (..., K_g, d, T_f)
        f = torch.transpose(f, -1, -2)          # (..., K_g, T_f, d)
        return f
