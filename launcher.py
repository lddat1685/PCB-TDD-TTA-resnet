# -*- coding: utf-8 -*-
"""Portable entry point for the TDD-Net PCB defect UI.

This file is run by the bundled CPython interpreter (./python/python.exe), not
by the venv's own python.exe, because the latter hard-codes an absolute path to
the machine that built it. Instead we add .venv/Lib/site-packages to sys.path
directly, which needs no reinstall and no absolute paths.

Usage (from anywhere):
    start_ui.bat            # Windows
    ./start_ui.sh           # Git Bash / WSL
    ./python/python.exe launcher.py
"""

from __future__ import print_function

import os
import site
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(HERE, '.venv', 'Lib', 'site-packages')


def _fail(msg):
    print('')
    print('=' * 72)
    print('KHONG THE KHOI DONG / CANNOT START')
    print('=' * 72)
    print(msg)
    print('')
    print('Xem README.md de biet cach sua.')
    sys.exit(1)


if not os.path.isdir(os.path.join(HERE, 'python')):
    _fail('Thieu thu muc ./python (Python 3.9 runtime).\n'
          'Missing ./python (portable Python 3.9 runtime).')
if not os.path.isdir(SITE):
    _fail('Thieu thu muc .venv/Lib/site-packages.\n'
          'Missing .venv/Lib/site-packages (the Python packages).')

site.addsitedir(SITE)
sys.path.insert(0, HERE)

print('Python  :', sys.version.split()[0])
print('Runtime :', sys.executable)
print('Packages:', SITE)
print('')

try:
    import tensorflow as tf  # noqa: F401
except Exception as exc:
    _fail('Khong import duoc TensorFlow: %s: %s\n'
          'Neu la loi ve DLL (VCRUNTIME140.dll / MSVCP140.dll), cai them\n'
          'Microsoft Visual C++ Redistributable 2015-2022 (x64):\n'
          'https://aka.ms/vs/17/release/vc_redist.x64.exe'
          % (type(exc).__name__, exc))

sys.argv = [os.path.join(HERE, 'ui', 'app.py')]
exec(compile(open(sys.argv[0], 'rb').read(), sys.argv[0], 'exec'),
     {'__name__': '__main__', '__file__': sys.argv[0]})
