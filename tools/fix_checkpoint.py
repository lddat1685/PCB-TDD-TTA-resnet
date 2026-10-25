# -*- coding: utf-8 -*-
"""Tao lai file `checkpoint` voi duong dan tuyet doi cua may hien tai.

Vi sao can: TF nap checkpoint qua `tf.train.latest_checkpoint()` doc file
`output/trained_weights/FPN_Res101_0117_OHEM/checkpoint`. File nay chua duong
dan tuyet doi cua may cu -> sang may moi phai viet lai. Script nay tu tim
ROOT tu vi tri file (khong phu thuoc CWD), chi dung thu vien chuan.

Chay:  python\python.exe tools\fix_checkpoint.py
"""
from __future__ import print_function
import io
import os
import sys


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(root, 'output', 'trained_weights', 'FPN_Res101_0117_OHEM')
    need = ['pcb_30300model.ckpt.index',
            'pcb_30300model.ckpt.meta',
            'pcb_30300model.ckpt.data-00000-of-00001']
    missing = [f for f in need if not os.path.isfile(os.path.join(d, f))]
    if missing:
        print('[LOI] thieu file checkpoint: %s' % (missing,))
        print('Thu muc: %s' % d)
        return 1
    p = d.replace('\\', '/') + '/pcb_30300model.ckpt'
    with io.open(os.path.join(d, 'checkpoint'), 'w', encoding='utf-8') as f:
        f.write(u'model_checkpoint_path: "%s"\n' % p)
        f.write(u'all_model_checkpoint_paths: "%s"\n' % p)
    try:
        print('[OK] checkpoint -> ' + p)
    except UnicodeEncodeError:
        print('[OK] checkpoint viet xong (duong dan co dau tieng Viet).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
