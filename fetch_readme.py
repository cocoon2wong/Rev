"""
@Author: Conghao Wong
@Date: 2024-12-04 09:44:45
@LastEditors: Conghao Wong
@LastEditTime: 2026-09-28 12:32:05
@Github: https://cocoon2wong.github.io
@Copyright 2026 Conghao Wong, All Rights Reserved.
"""

import re
import requests
import shutil

GITHUB_USERNAME = 'cocoon2wong'
GITHUB_REPONAME = 'Rev'
GITHUB_READMEFILE = 'README.md'

SOURCE_FILE = '__pages/guidelines.md'
TARGET_FILE = '__pages/README.md.downloaded'

START_LINE = '## Getting Started'

DOWNLOAD_SOURCE = f'https://github.com/{GITHUB_USERNAME}/{GITHUB_REPONAME}/raw/refs/heads/main/{GITHUB_READMEFILE}'


if __name__ == '__main__':
    # Backup old files
    shutil.copy(SOURCE_FILE, SOURCE_FILE + '.backup')

    # Fetch new file
    with open(TARGET_FILE, 'wb') as f:
        _content = requests.get(DOWNLOAD_SOURCE)
        f.write(_content.content)

    with open(TARGET_FILE, 'r') as f:
        new_lines = f.readlines()

    # Check lines
    for i, line in enumerate(new_lines):
        if line.startswith(START_LINE):
            break

    # Write new file
    with open(SOURCE_FILE, 'a+') as f:
        f.writelines(new_lines[i:])

    # Anti-crawler: replace email @xxx.com with [at-mark}xxx.com
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    content = re.sub(
        r'(?<=[a-zA-Z0-9_.+-])@([a-zA-Z0-9.-]+\.com)\b',
        r'[at-mark}\1',
        content,
    )

    # Replace image links: figs/xxx -> /assets/figs/xxx
    content = re.sub(r'(?<=[("\'])figs/', '/Rev/assets/figs/', content)

    with open(SOURCE_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

