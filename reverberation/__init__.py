"""
@Author: Conghao Wong
@Date: 2024-12-05 15:12:33
@LastEditors: Conghao Wong
@LastEditTime: 2025-01-15 15:37:16
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import qpid

from .__args import ReverberationArgs
from .model import Reverberation, ReverberationModel

# Register new args and models
qpid.register(rev=[Reverberation, ReverberationModel])
qpid.register_args(ReverberationArgs, 'Reverberation Args')
