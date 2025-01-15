"""
@Author: Conghao Wong
@Date: 2024-12-05 15:02:40
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-05 20:54:56
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import sys

import qpid
import reverberation

try:
    import resonance
except:
    pass

if __name__ == '__main__':
    qpid.entrance(sys.argv)
