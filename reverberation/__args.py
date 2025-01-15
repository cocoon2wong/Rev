"""
@Author: Conghao Wong
@Date: 2024-12-05 15:14:02
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-15 15:43:36
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

from qpid.args import DYNAMIC, STATIC, TEMPORARY, EmptyArgs


class ReverberationArgs(EmptyArgs):

    @property
    def Kc(self) -> int:
        """
        The number of style channels when making predictions.
        """
        return self._arg('Kc', 20, argtype=STATIC,
                         desc_in_model_summary='Output channels')

    @property
    def partitions(self) -> int:
        """
        The number of partitions when computing the angle-based feature.
        """
        return self._arg('partitions', -1, argtype=STATIC,
                         desc_in_model_summary='Number of Angle-based Partitions')

    @property
    def T(self) -> str:
        """
        Transform type used to compute trajectory spectrums.

        It could be:
        - `none`: no transformations
        - `haar`: haar wavelet transform
        - `db2`: DB2 wavelet transform
        """
        return self._arg('T', 'haar', argtype=STATIC, short_name='T',
                         desc_in_model_summary='Transform type')

    @property
    def no_interaction(self) -> int:
        """
        Whether to forecast trajectories by considering social interactions.
        It will compute all social-interaction-related components on the set
        of empty neighbors if this args is set to `1`.
        """
        return self._arg('no_interaction', 0, argtype=TEMPORARY)

    @property
    def encode_agent_types(self) -> int:
        """
        Choose whether to encode the type name of each agent.
        It is mainly used in multi-type-agent prediction scenes, providing
        a unique type-coding for each type of agents when encoding their
        trajectories.
        """
        return self._arg('encode_agent_types', 0, argtype=STATIC)

    @property
    def compute_linear_base(self) -> int:
        """
        Whether to learn to forecast the linear base during training.
        """
        return self._arg('compute_linear_base', 1, argtype=STATIC,
                         desc_in_model_summary='Train with linear base')

    @property
    def compute_self_bias(self) -> int:
        """
        Whether to learn to forecast the self-bias during training.
        """
        return self._arg('compute_self_bias', 1, argtype=STATIC,
                         other_names=['learn_self_bias'],
                         desc_in_model_summary='Learn self-bias')

    @property
    def compute_re_bias(self) -> int:
        """
        Whether to learn to forecast the re-bias during training.
        """
        return self._arg('compute_re_bias', 1, argtype=STATIC,
                         other_names=['learn_re_bias'],
                         desc_in_model_summary='Learn re-bias')

    @property
    def lite(self) -> int:
        """
        It controls whether to implement the full reverberation kernel on all
        historical steps and angle-based partitions or the simplified shared-
        steps. Simultaneously, the model will compute all angle-based social
        partitions on a flattened feature rather than all observation frames,
        which may further reduce the computation consumptions. This arg is
        typically used to obtain a model variation with faster computation
        and smaller model size, reducing prediction performance as a compromise.
        """
        return self._arg('lite', 0, argtype=STATIC,
                         desc_in_model_summary='Lite mode')

    @property
    def no_linear_base(self) -> int:
        """
        Ignoring the linear base term when forecasting.
        It only works when testing.
        """
        return self._arg('no_linear_base', 0, argtype=TEMPORARY)

    @property
    def no_self_bias(self) -> int:
        """
        Ignoring the self-bias term when forecasting.
        It only works when testing.
        """
        return self._arg('no_self_bias', 0, argtype=TEMPORARY)

    @property
    def no_re_bias(self) -> int:
        """
        Ignoring the resonance-bias term when forecasting.
        It only works when testing.
        """
        return self._arg('no_re_bias', 0, argtype=TEMPORARY)

    @property
    def draw_kernels(self) -> int:
        """
        Choose whether to draw and show visualized kernels when testing.
        It is typically used in the playground mode.
        """
        return self._arg('draw_kernels', 0, argtype=TEMPORARY)

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
