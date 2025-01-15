"""
@Author: Conghao Wong
@Date: 2024-12-16 14:56:33
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-15 15:18:38
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import torch

from qpid.model import layers, transformer

from .__layers import KernelLayer


class SocialReverberationLayer(torch.nn.Module):
    """
    Social-Reverberation Layer
    ---
    Forecast the *social-caused* future trajectories according to the observed
    trajectories of both ego agents and their neighbors.
    Similar to the `SelfReververationLayer`, two reverberation kernels will be
    computed to weighted sum historical features to *wiring* past information
    into the future:

    - **Social-Generation kernel**: Weighted sum features in different styles to
      achieve the random/characterized/multi-style social behavior prediction;
    - **Social-reverberation kernel**: Evaluate how much contribution that each
      historical frame (step) has made when planning future trajectories and 
      social behaviors on each specific future frame (step).

    *NOTE* that the layer's behaviors may change according to the arg `lite`.
    """

    def __init__(self, input_ego_feature_dim: int,
                 input_re_feature_dim: int,
                 output_feature_dim: int,
                 angle_partitions: int,
                 noise_depth: int,
                 traj_channels: int,
                 transform_layer: layers.transfroms._BaseTransformLayer,
                 inverse_transform_layer: layers.transfroms._BaseTransformLayer,
                 enable_lite_mode: int | bool = False,
                 *args, **kwargs) -> None:

        super().__init__()

        # Variables and Settings
        self.d_i_ego = input_ego_feature_dim
        self.d_i_re = input_re_feature_dim
        self.d = output_feature_dim
        self.d_noise = noise_depth

        self.traj_channels = traj_channels
        self.p = angle_partitions
        self.lite = enable_lite_mode

        # Layers
        # Transform layers
        self.Tlayer = transform_layer
        self.iTlayer = inverse_transform_layer

        # Shapes
        self.Tsteps_en, self.Tchannels_en = self.Tlayer.Tshape
        self.Tsteps_de, self.Tchannels_de = self.iTlayer.Tshape

        if not self.lite:
            self.steps = self.Tsteps_en * self.p
        else:
            self.steps = max(self.Tsteps_en, self.p)

        # Fusion layer (ego features and resonance features)
        self.concat_fc = layers.Dense(self.d_i_ego + self.d_i_re,
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
            input_vocab_size=self.Tchannels_en,
            target_vocab_size=self.Tchannels_en,
            pe_input=self.steps,
            pe_target=self.steps,
            include_top=False,
        )

        # FC layers for computing reverberation kernels
        if not self.lite:
            self.k1 = KernelLayer(self.d, self.d, self.traj_channels)
            self.k2 = KernelLayer(self.d, self.d, self.Tsteps_de)

        else:
            self.k1 = layers.Dense(self.d, self.traj_channels, torch.nn.Tanh)
            self.k2 = layers.Dense(self.d, self.Tsteps_de, torch.nn.Tanh)

        self.outer = layers.OuterLayer(self.steps, self.steps)
        self.decoder = layers.Dense(self.d, self.Tchannels_de)

    def forward(self, x_ego_diff: torch.Tensor,
                f_ego_diff: torch.Tensor,
                re_matrix: torch.Tensor,
                repeats: int = 1,
                training=None, mask=None, *args, **kwargs):

        # Pad features to keep the compatible tensor shape
        if not self.lite:
            f_diff_pad = torch.repeat_interleave(f_ego_diff, self.p, -2)
            f_re_pad = torch.flatten(re_matrix, -3, -2)
        else:
            f_diff_pad = pad(f_ego_diff, self.steps)
            f_re_pad = pad(re_matrix, self.steps)

        # Concat and fuse resonance matrices with trajectory features
        # -> (batch, steps, d) (when `lite` is enabled)
        # -> (batch, steps*partitions, d) (when `lite` is disabled)
        f_behavior = torch.concat([f_diff_pad, f_re_pad], dim=-1)
        f_behavior = self.concat_fc(f_behavior)

        # Target value for queries
        traj_targets = self.Tlayer(x_ego_diff)
        if not self.lite:
            traj_targets = torch.repeat_interleave(traj_targets, self.p, -2)
        else:
            traj_targets = pad(traj_targets, self.steps)

        all_predictions = []
        for _ in range(repeats):
            # Assign random ids and embedding -> (batch, steps, d)
            z = torch.normal(mean=0, std=1,
                             size=list(f_behavior.shape[:-1]) + [self.d_noise])
            re_f_z = self.ie(z.to(x_ego_diff.device))

            # (batch, steps, 2*d)
            re_f_final = torch.concat([f_behavior, re_f_z], dim=-1)

            # Transformer outputs' shape is (batch, steps, d)
            f_tran, _ = self.T(inputs=re_f_final,
                               targets=traj_targets,
                               training=training)

            # Outer product -> (batch, d, steps, steps)
            f_tran_t = torch.transpose(f_tran, -1, -2)
            f_o = self.outer(f_tran_t, f_tran_t)

            # Compute reverberation kernels
            k1 = self.k1(f_tran)        # (batch, steps, Kc)
            k2 = self.k2(f_tran)        # (batch, steps, Tsteps_de)

            # Apply k1
            f1 = f_o @ k1[..., None, :, :]    # (batch, d, steps, Kc)

            # Apply k2
            f2 = torch.transpose(f1, -1, -2) @ k2[..., None, :, :]

            # Decode predictions
            f2 = torch.permute(f2, [0, 2, 3, 1])        # (b, Kc, Tsteps_de, d)

            y = self.iTlayer(self.decoder(f2))          # (b, Kc, pred, dim)
            all_predictions.append(y)

        all_predictions = torch.concat(all_predictions, dim=-3)
        return all_predictions, k1, k2


def pad(input: torch.Tensor, max_steps: int):
    """
    Zero-padding the input tensor (whose shape must be `(batch, steps, dim)`).
    It will pad the input tensor on the `steps` axis if `steps < max_steps`.
    """
    steps = input.shape[-2]
    if steps < max_steps:
        paddings = [0, 0, 0, max_steps - steps, 0, 0]
        return torch.nn.functional.pad(input, paddings)
    else:
        return input
