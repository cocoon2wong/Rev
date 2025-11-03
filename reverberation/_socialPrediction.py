"""
@Author: Conghao Wong
@Date: 2024-12-16 14:56:33
@LastEditors: Conghao Wong
@LastEditTime: 2025-04-22 10:45:56
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers, transformer

from .__layers import KernelLayer
from ._revberationTransfrom import (LinearMappingLayer,
                                    MultiStyleGeneratingLayer,
                                    ReverberationTransform)


class SocialPrediction(torch.nn.Module):
    """
    Social Prediction Layer
    ---
    Forecast the *social-caused* future trajectories according to the observed
    trajectories of both ego agents and their neighbors.
    Similar to the `NonInteractivePrediction`, two reverberation kernels will
    be computed to weighted sum historical features to *wiring* past information
    into the (rehearsal) future:

    - **Social-Generating kernel**: Weighted sum features in different styles to
      achieve the random/characterized/multi-style social behavior prediction;
    - **Social-reverberation kernel**: Evaluate how much contribution that each
      historical frame (step) has made when planning future trajectories and 
      social behaviors on each specific future frame (step).
    """

    def __init__(self, input_ego_feature_dim: int,
                 input_social_feature_dim: int,
                 output_feature_dim: int,
                 angle_partitions: int,
                 noise_depth: int,
                 traj_generations: int,
                 transform_layer: layers.transfroms._BaseTransformLayer,
                 inverse_transform_layer: layers.transfroms._BaseTransformLayer,
                 disable_G: bool = False,
                 disable_R: bool = False,
                 *args, **kwargs) -> None:

        super().__init__()

        # Variables and Settings
        self.d_ego = input_ego_feature_dim
        self.d_soc = input_social_feature_dim
        self.d = output_feature_dim
        self.d_noise = noise_depth
        self.K_g = traj_generations
        self.p = angle_partitions

        # Ablation settings
        self.disable_G = disable_G
        self.disable_R = disable_R

        # Layers
        # Transform layers (for trajectories)
        self.Tlayer = transform_layer
        self.iTlayer = inverse_transform_layer

        # Shapes
        self.T_h, self.M_h = self.Tlayer.Tshape
        self.T_f, self.M_f = self.iTlayer.Tshape
        self.steps = self.T_h * self.p

        # Fusion layer (ego features and social features)
        self.concat_fc = layers.Dense(self.d_ego + self.d_soc,
                                      self.d//2,
                                      torch.nn.ReLU)

        # Noise encoding
        self.ie = layers.TrajEncoding(self.d_noise, self.d//2, torch.nn.Tanh)

        # Transformer as the feature extractor
        self.T = transformer.Transformer(
            num_layers=2,
            d_model=self.d,
            num_heads=8,
            dff=512,
            input_vocab_size=self.M_h,
            target_vocab_size=self.M_h,
            pe_input=self.steps,
            pe_target=self.steps,
            include_top=False,
        )

        # Reverberation-transform-related layers
        # Main transform layer
        self.rev = ReverberationTransform(
            historical_steps=self.steps,
            future_steps=self.T_f,
        )

        # Kernel layers
        if self.disable_G:
            # Use the MSK-like way to generate stochastic predictions
            # See "MSN: Multi-Style Network for Trajectory Prediction"
            self.G_layer = MultiStyleGeneratingLayer(
                feature_dim=self.d,
                style_channels=self.K_g,
            )
        else:
            # The generating kernel
            self.k1 = KernelLayer(self.d, self.d, self.K_g)

        if self.disable_R:
            # Forecast trajectories using direct FC layers
            self.R_layer = LinearMappingLayer(
                feature_dim=self.d,
                historical_steps=self.steps,
                future_steps=self.T_f,
            )
        else:
            # The reverberation kernel
            self.k2 = KernelLayer(self.d, self.d, self.T_f)

        # Final output layer
        self.decoder = layers.Dense(self.d, self.M_f)

    def forward(self, x_ego_diff: torch.Tensor,
                f_ego_diff: torch.Tensor,
                f_social: torch.Tensor,
                repeats: int = 1,
                training=None, mask=None, *args, **kwargs):

        # Pad features to keep the compatible tensor shape
        # Original shape of `f_ego_diff` is `(batch, T_h, d_ego)`
        f_diff_pad = torch.repeat_interleave(f_ego_diff, self.p, -2)

        # Original shape of `f_social` is `(batch, T_h, partitions, d_soc)`
        f_social_pad = torch.flatten(f_social, -3, -2)

        # Concat and fuse social features with trajectory features
        # It serves as keys and queries in attention layers
        # -> `(batch, steps, d)`, where `steps = T_h * partitions`
        f = torch.concat([f_diff_pad, f_social_pad], dim=-1)
        f = self.concat_fc(f)

        # Target value for queries
        # `x_ego_diff = x_ego - linear_fit`
        # -> (batch, T_h * partitions, M)
        traj_targets = self.Tlayer(x_ego_diff)
        traj_targets = torch.repeat_interleave(traj_targets, self.p, -2)

        all_predictions = []
        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d/2)
            z = torch.normal(mean=0, std=1,
                             size=list(f.shape[:-1]) + [self.d_noise])
            re_f_z = self.ie(z.to(f.device))

            # -> (batch, steps, d)
            re_f_final = torch.concat([f, re_f_z], dim=-1)

            # Transformer backbone -> (batch, steps, d)
            f_tran, _ = self.T(inputs=re_f_final,
                               targets=traj_targets,
                               training=training)

            # Reverberation kernels and transform
            G = self.k1(f_tran) if not self.disable_G else self.G_layer
            R = self.k2(f_tran) if not self.disable_R else self.R_layer
            f_rev = self.rev(f_tran, R, G)          # (batch, K_g, T_f, d)

            # Decode predictions
            y = self.decoder(f_rev)                 # (batch, K_g, T_f, M)
            y = self.iTlayer(y)                     # (batch, K_g, t_f, m)

            all_predictions.append(y)

        # Stack all outputs -> (batch, K, t_f, m)
        return torch.concat(all_predictions, dim=-3), G, R
