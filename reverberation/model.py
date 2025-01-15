"""
@Author: Conghao Wong
@Date: 2024-12-05 15:17:31
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-15 15:22:43
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from typing import Any

import torch

from qpid.constant import INPUT_TYPES
from qpid.model import Model, layers
from qpid.training import Structure
from qpid.utils import INIT_POSITION

from .__args import ReverberationArgs
from .__layers import LinearDiffEncoding
from ._resonanceLayer import ResonanceLayer
from ._selfReverberation import SelfReverberationLayer
from ._socialReverberation import SocialReverberationLayer


class ReverberationModel(Model):
    def __init__(self, structure=None, *args, **kwargs):
        super().__init__(structure, *args, **kwargs)

        self.as_final_stage_model = True

        # Init args
        self.args._set_default('K', 1)
        self.args._set_default('K_train', 1)
        self.args._set('output_pred_steps', 'all')
        self.rev_args = self.args.register_subargs(ReverberationArgs, 'rev')

        # Set model inputs
        # Types of agents are only used in complex scenes
        # For other datasets, keep it disabled (through the arg)
        if not self.rev_args.encode_agent_types:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ)
        else:
            self.set_inputs(INPUT_TYPES.OBSERVED_TRAJ,
                            INPUT_TYPES.NEIGHBOR_TRAJ,
                            INPUT_TYPES.AGENT_TYPES)

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(self.rev_args.T)
        self.tlayer = t_type((self.args.obs_frames, self.dim))
        self.itlayer = it_type((self.args.pred_frames, self.dim))

        # Common settings for all layers (subnetworks)
        settings: dict[str, Any] = dict(
            noise_depth=self.args.noise_depth,
            traj_channels=self.rev_args.Kc,
            angle_partitions=self.rev_args.partitions,
            transform_layer=self.tlayer,
            inverse_transform_layer=self.itlayer,
            enable_lite_mode=self.rev_args.lite,
            encode_agent_types=self.rev_args.encode_agent_types,
        )

        # Linear difference encoding
        self.linear = LinearDiffEncoding(
            obs_frames=self.args.obs_frames,
            pred_frames=self.args.pred_frames,
            output_units=self.d//2,
            **settings,
        )

        if self.rev_args.compute_self_bias:
            # Self-reverberation layer
            self.self_rev = SelfReverberationLayer(
                input_feature_dim=self.d//2,
                output_feature_dim=self.d,
                **settings,
            )

        if self.rev_args.compute_re_bias:
            # Resonance feature
            self.resonance = ResonanceLayer(
                hidden_feature_dim=self.d,
                output_feature_dim=self.d//2,
                **settings,
            )

            # Re-reverberation layer
            self.re_rev = SocialReverberationLayer(
                input_ego_feature_dim=self.d//2,
                input_re_feature_dim=self.d//2,
                output_feature_dim=self.d,
                **settings,
            )

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # -------------
        # Unpack inputs
        # -------------
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)
        x_nei = self.get_input(inputs, INPUT_TYPES.NEIGHBOR_TRAJ)

        # Create empty neighbors (mainly used in qualitative analyses)
        if self.rev_args.no_interaction and not training:
            x_nei = INIT_POSITION * torch.ones_like(x_nei)

        # Agent types (labels) will be encoded only in complex scenes
        if self.rev_args.encode_agent_types:
            agent_types = self.get_input(inputs, INPUT_TYPES.AGENT_TYPES)
        else:
            agent_types = None

        # Times of multiple generations
        repeats = self.args.K_train if training else self.args.K

        # -----------
        # Linear Base
        # -----------
        # Encode difference features (for ego agents)
        f_ego_diff, linear_fit, linear_base = self.linear(x_ego, agent_types)
        x_ego_diff = x_ego - linear_fit

        # The linear base
        if self.rev_args.compute_linear_base and not self.rev_args.no_linear_base:
            linear_base = linear_base[..., None, :, :]
        else:
            linear_base = 0

        # -----------------------
        # Self-Reverberation-Bias
        # -----------------------
        if self.rev_args.compute_self_bias and not self.rev_args.no_self_bias:
            self_rev_bias, self_k1, self_k2 = self.self_rev(
                f_ego_diff, x_ego_diff, repeats, training)
        else:
            self_rev_bias, self_k1, self_k2 = [0, None, None]

        # -------------------------
        # Social-Reverberation-Bias
        # -------------------------
        if self.rev_args.compute_re_bias and not self.rev_args.no_re_bias:
            re_matrix, f_re = self.resonance(self.picker.get_center(x_ego)[..., :2],
                                             self.picker.get_center(x_nei)[..., :2])

            re_rev_bias, re_k1, re_k2 = self.re_rev(
                x_ego_diff, f_ego_diff, re_matrix, repeats, training)
        else:
            re_rev_bias, re_k1, re_k2 = [0, None, None]

        # ----------------------------------
        # Reverberation-Kernel-Visualization
        # ----------------------------------
        if (self.rev_args.draw_kernels and
                None not in [self_k1, self_k2, re_k1, re_k2]):
            from .utils import show_kernel

            # Self-Reverberation kernels
            show_kernel(self_k1, 'self-1', 1, self.Tsteps_en, self.rev_args.Kc)
            show_kernel(self_k2, 'self-2', 1, self.Tsteps_en, self.Tsteps_de)

            # Social-Reverberation kernels
            show_kernel(re_k1, 'social-1', self.rev_args.partitions,
                        self.Tsteps_en, self.rev_args.Kc)
            show_kernel(re_k2, 'social-2', self.rev_args.partitions,
                        self.Tsteps_en, self.Tsteps_de)

        return linear_base + self_rev_bias + re_rev_bias


class Reverberation(Structure):
    MODEL_TYPE = ReverberationModel
