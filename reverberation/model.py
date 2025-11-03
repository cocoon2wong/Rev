"""
@Author: Conghao Wong
@Date: 2024-12-05 15:17:31
@LastEditors: Conghao Wong
@LastEditTime: 2025-09-28 09:59:26
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
from ._nonInteractivePrediction import NonInteractivePrediction
from ._resonanceLayer import ResonanceLayer
from ._socialPrediction import SocialPrediction


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
        inputs = [INPUT_TYPES.OBSERVED_TRAJ,
                  INPUT_TYPES.NEIGHBOR_TRAJ]

        # Types of agents are only used in complex scenes
        # For other datasets, keep it disabled (through the arg)
        if self.rev_args.encode_agent_types:
            inputs.append(INPUT_TYPES.AGENT_TYPES)

        # It will not prepare neighbors' trajectories when disabling social parts
        if not self.rev_args.compute_social:
            inputs.remove(INPUT_TYPES.NEIGHBOR_TRAJ)

        self.set_inputs(*inputs)

        # Layers
        # Transform layers
        t_type, it_type = layers.get_transform_layers(self.rev_args.T)
        self.tlayer = t_type((self.args.obs_frames, self.dim))
        self.itlayer = it_type((self.args.pred_frames, self.dim))

        # Common settings for all layers (subnetworks)
        settings: dict[str, Any] = dict(
            noise_depth=self.args.noise_depth,
            traj_generations=self.rev_args.Kc,
            angle_partitions=self.rev_args.partitions,
            transform_layer=self.tlayer,
            inverse_transform_layer=self.itlayer,
            encode_agent_types=self.rev_args.encode_agent_types,
            disable_G=self.rev_args.disable_G,
            disable_R=self.rev_args.disable_R,
        )

        # Linear difference encoding
        self.linear = LinearDiffEncoding(
            obs_frames=self.args.obs_frames,
            pred_frames=self.args.pred_frames,
            output_units=self.d//2,
            **settings,
        )

        if self.rev_args.compute_noninteractive:
            # Non-interactive prediction layer
            self.self_rev = NonInteractivePrediction(
                input_feature_dim=self.d//2,
                output_feature_dim=self.d,
                **settings,
            )

        if self.rev_args.compute_social:
            # Resonance feature (social-interaction-modeling)
            self.resonance = ResonanceLayer(
                hidden_feature_dim=self.d,
                output_feature_dim=self.d//2,
                **settings,
            )

            # Social prediction layer
            self.re_rev = SocialPrediction(
                input_ego_feature_dim=self.d//2,
                input_social_feature_dim=self.d//2,
                output_feature_dim=self.d,
                **settings,
            )

    def forward(self, inputs, training=None, mask=None, *args, **kwargs):
        # -------------
        # Unpack inputs
        # -------------
        x_ego = self.get_input(inputs, INPUT_TYPES.OBSERVED_TRAJ)

        if self.rev_args.compute_social:
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

        # -----------------
        # Linear Trajectory
        # -----------------
        # Linear prediction (least squares) && Encode difference features (for ego agents)
        f_ego_diff, linear_fit, linear_pred = self.linear(x_ego, agent_types)
        x_ego_diff = x_ego - linear_fit

        # The linear trajectory
        if self.rev_args.compute_linear and not self.rev_args.test_with_linear:
            linear_pred = linear_pred[..., None, :, :]
        else:
            linear_pred = 0

        # --------------------------
        # Non-interactive Prediction
        # --------------------------
        if self.rev_args.compute_noninteractive and not self.rev_args.test_with_noninteractive:
            non_interactive_pred, G_non, R_non = self.self_rev(
                f_ego_diff, x_ego_diff, repeats, training)
        else:
            non_interactive_pred, G_non, R_non = [0, None, None]

        # -----------------
        # Social Prediction
        # -----------------
        if self.rev_args.compute_social and not self.rev_args.test_with_social:
            f_social, f_re = self.resonance(self.picker.get_center(x_ego)[..., :2],
                                            self.picker.get_center(x_nei)[..., :2])

            social_pred, G_soc, R_soc = self.re_rev(
                x_ego_diff, f_ego_diff, f_social, repeats, training)
        else:
            social_pred, G_soc, R_soc = [0, None, None]

        # ----------------------------------
        # Reverberation-Kernel Visualization
        # ----------------------------------
        if (d := self.rev_args.draw_kernels):
            from .utils import vis_kernels
            vis_kernels(R_non, G_non, 'Non-Interactive', d)
            vis_kernels(R_soc, G_soc, 'Social', d,
                        self.rev_args.partitions,
                        self.rev_args.select_social_partition)

        # -> (batch, K_g, pred, dim)
        y = linear_pred + non_interactive_pred + social_pred

        # Select a generation (only for ablation studies)
        if (s := self.rev_args.select_generating_channel) != -1:
            y = y[..., s, None, :, :]

        return y


class Reverberation(Structure):
    MODEL_TYPE = ReverberationModel
