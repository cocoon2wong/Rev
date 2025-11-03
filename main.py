"""
@Author: Conghao Wong
@Date: 2024-12-05 15:02:40
@LastEditors: Conghao Wong
@LastEditTime: 2025-09-28 10:25:43
@Github: https://cocoon2wong.github.io
@Copyright 2024 Conghao Wong, All Rights Reserved.
"""

import sys

import qpid
import reverberation

"""
This repo is compatible with our previous models.
Put their model folders (not the whole repo) into this folder to enable them.

File structures:

(root)
    |___qpid
    |___dataset_original
    |___dataset_configs
    |___dataset_processed
    |___main.py
    |___reverberation         # <- this repo
    |___resonance             # <- optional, our previous model
    |___socialCircle          # <- optional, our previous model 
    |___...
"""

try:
    import socialCircle
except:
    pass

try:
    import resonance
except:
    pass

if __name__ == '__main__':
    qpid.entrance(sys.argv)
