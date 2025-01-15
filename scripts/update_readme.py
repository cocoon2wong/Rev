"""
@Author: Conghao Wong
@Date: 2021-08-05 15:26:57
@LastEditors: Conghao Wong
@LastEditTime: 2024-12-27 10:44:33
@Description: file content
@Github: https://github.com/cocoon2wong
@Copyright 2022 Conghao Wong, All Rights Reserved.
"""

import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import qpid
import reverberation
from playground import PlaygroundArgs
from qpid.mods.vis import VisArgs

TARGET_FILE = './README.md'
SECTION_HEAD = """
---

## Args Used

Please specify your customized args when training or testing your model in the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 -SHORT_ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages are listed below.

About the `argtype`:

- Args with argtype=`static` can not be changed once after training.
  When testing the model, the program will not parse these args to overwrite the saved values.
- Args with argtype=`dynamic` can be changed anytime.
  The program will try to first parse inputs from the terminal and then try to load from the saved JSON file.
- Args with argtype=`temporary` will not be saved into JSON files.
  The program will parse these args from the terminal at each time.
"""


if __name__ == '__main__':
    qpid.register_args(PlaygroundArgs, 'Playground Args')
    qpid.register_args(VisArgs, 'Visualization Args')
    qpid.help.update_readme([SECTION_HEAD] + qpid.print_help_info(),
                            TARGET_FILE)
