"""
@Author: Conghao Wong
@Date: 2024-12-05 15:14:02
@LastEditors: Conghao Wong
@LastEditTime: 2025-09-28 10:01:17
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class ReverberationArgs(EmptyArgs):

    @property
    def Kc(self) -> int:
        """
        The number of generations when making predictions.
        It is also the channels of the generating kernel in the proposed
        reverberation transform.
        """
        return self._arg('Kc', 20, argtype=STATIC,
                         other_names=['Kg', 'K_g'],
                         desc_in_model_summary='Generating channels')

    @property
    def select_generating_channel(self) -> int:
        """
        (Ablation Only) Select one of the generating channel as the direct
        output of the prediction network.
        Value range: 0 <= n < K_g.
        NOTE: This MAY lead to significant performance degradation as only
        one channel is reserved for prediction. This arg is only used for
        conducting ablation analyses or discussions.
        """
        return self._arg('select_generating_channel', -1, argtype=TEMPORARY)

    @property
    def partitions(self) -> int:
        """
        The number of partitions when computing the angle-based feature.
        It is only used when modeling social interactions.
        """
        return self._arg('partitions', -1, argtype=STATIC,
                         desc_in_model_summary='Number of Angle-based Partitions')

    @property
    def T(self) -> str:
        """
        Transform type used to compute trajectory spectrums.

        It could be:
        - `none`: no transformations;
        - `haar`: haar wavelet transform;
        - `db2`: DB2 wavelet transform.
        """
        return self._arg('T', 'haar', argtype=STATIC, short_name='T',
                         desc_in_model_summary='Transform type')

    @property
    def no_interaction(self) -> int:
        """
        (bool) Whether to forecast trajectories by considering social
        interactions. It will compute all social-interaction-related components
        on the set of empty neighbors if this args is set to `1`.
        """
        return self._arg('no_interaction', 0, argtype=TEMPORARY)

    @property
    def encode_agent_types(self) -> int:
        """
        (bool) Choose whether to encode the type name of each agent.
        It is mainly used in multi-type-agent prediction scenes, providing
        a unique type-coding for each type of agents when encoding their
        trajectories.
        """
        return self._arg('encode_agent_types', 0, argtype=STATIC)

    @property
    def compute_linear(self) -> int:
        """
        (bool) Choose whether to learn to forecast the linear trajectory during
        training.
        """
        return self._arg('compute_linear', 1, argtype=STATIC,
                         other_names=['compute_linear_base'],
                         desc_in_model_summary=('Training configs (Rev Model)',
                                                'Train with linear trajectory'))

    @property
    def compute_noninteractive(self) -> int:
        """
        (bool) Choose whether to learn to forecast the non-interactive
        trajectory during training.
        """
        return self._arg('compute_noninteractive', 1, argtype=STATIC,
                         other_names=['learn_self_bias',
                                      'compute_self_bias',
                                      'compute_non'],
                         desc_in_model_summary=('Training configs (Rev Model)',
                                                'Learn non-interactive latency during training'))

    @property
    def compute_social(self) -> int:
        """
        (bool) Choose whether to learn to forecast the social trajectory
        during training.
        """
        return self._arg('compute_social', 1, argtype=STATIC,
                         other_names=['learn_re_bias',
                                      'compute_re_bias'],
                         desc_in_model_summary=('Training configs (Rev Model)',
                                                'Learn social latency during training'))

    @property
    def disable_G(self) -> int:
        """
        (bool) Choose whether to disable the generating kernels when applying
        the reverberation transform. An MSN-like generating approach will be
        used if this arg is set to `1`.
        """
        return self._arg('disable_G', 0, argtype=STATIC,
                         desc_in_model_summary=('Reverberation Transform',
                                                'Disable the generating kernel'))

    @property
    def disable_R(self) -> int:
        """
        (bool) Choose whether to disable the reverberation kernels when
        applying the reverberation transform. Flatten and fc layers will be
        used if this arg is set to `1`.
        """
        return self._arg('disable_R', 0, argtype=STATIC,
                         desc_in_model_summary=('Reverberation Transform',
                                                'Disable the reverberation kernel'))

    @property
    def test_with_linear(self) -> int:
        """
        (bool) Choose whether to ignore the linear base when forecasting.
        It only works when testing.
        """
        return self._arg('test_with_linear', 0, argtype=TEMPORARY)

    @property
    def test_with_noninteractive(self) -> int:
        """
        (bool) Choose whether to ignore the self-bias when forecasting.
        It only works when testing.
        """
        return self._arg('test_with_noninteractive', 0, argtype=TEMPORARY,
                         other_names=['test_with_non'])

    @property
    def test_with_social(self) -> int:
        """
        (bool) Choose whether to ignore the resonance-bias when forecasting.
        It only works when testing.
        """
        return self._arg('test_with_social', 0, argtype=TEMPORARY,
                         other_names=['test_with_soc'])

    @property
    def draw_kernels(self) -> int:
        """
        Choose whether or in which ways to draw and show visualized kernels
        when testing. It accepts an int value, including `[0, 1, 2, 3]`:
        - `0`: Do nothing;
        - `1`: Only visualize the reverberation kernel;
        - `2`: Visualize both reverberation and generating kernels;
        - `3`: Visualize both kernels and their inverse kernels.

        This arg is typically used in the playground mode. 
        """
        return self._arg('draw_kernels', 0, argtype=TEMPORARY)

    @property
    def select_social_partition(self) -> int:
        """
        Choose which social partition will be displayed when visualizing social
        generating kernels.
        The indices of social partitions start from `1`, rather than `0`.
        It only works when the arg `draw_kernels` is set to `2` or `3`.
        NOTE: This value should be no more than the number of total partitions.
        """
        return self._arg('select_social_partition', 1, argtype=TEMPORARY)

    def _init_all_args(self):
        super()._init_all_args()

        if self.T == 'fft':
            self.log(f'Transform `{self.T}` is not supported!',
                     level='error', raiseError=ValueError)

        if self.partitions <= 0:
            self.log(f'Illegal partition settings ({self.partitions})! ' +
                     'Please add the arg `--partitions` to set the number of ' +
                     'angle-based partitions.',
                     level='error', raiseError=ValueError)
