# -*- coding: utf-8 -*-
"""Kiem tra model dang su dung (chi doc, khong sua logic).

Chay (tu thu muc repo):
  python\\python.exe tools\\check_model.py
"""
from __future__ import absolute_import, division, print_function
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(HERE, '.venv', 'Lib', 'site-packages')
import site as _site
_site.addsitedir(SITE)
sys.path.insert(0, HERE)
os.chdir(os.path.join(HERE, 'tools'))
import tf1_compat  # noqa: F401
import tensorflow as tf
from libs.configs import cfgs
from libs.label_name_dict.label_dict import LABEl_NAME_MAP
from ui.app import Detector

print('=' * 72)
print('MODEL DANG SU DUNG — TDD-Net PCB')
print('=' * 72)
print('VERSION        :', cfgs.VERSION)
print('NET_NAME       :', cfgs.NET_NAME, '(ResNet-101 backbone + FPN + FastRCNN)')
print('DATASET        :', cfgs.DATASET_NAME, '| CLASS_NUM =', cfgs.CLASS_NUM)
print('Classes        :', {i: LABEl_NAME_MAP[i] for i in sorted(LABEl_NAME_MAP)})
print('IMG_SHORT_SIDE :', cfgs.IMG_SHORT_SIDE_LEN, '| MAX_LENGTH =', cfgs.IMG_MAX_LENGTH)
print('PIXEL_MEAN     :', cfgs.PIXEL_MEAN)
print('Anchors        : BASE', cfgs.BASE_ANCHOR_SIZE_LIST, 'x SCALES', cfgs.ANCHOR_SCALES, 'x RATIOS', cfgs.ANCHOR_RATIOS)
print('Levels         :', cfgs.LEVLES, '| strides', cfgs.ANCHOR_STRIDE_LIST)
print('RPN            : NMS', cfgs.RPN_NMS_IOU_THRESHOLD, '| test topK', cfgs.RPN_TOP_K_NMS_TEST, '->', cfgs.RPN_MAXIMUM_PROPOSAL_TEST)
print('FastRCNN       : NMS', cfgs.FAST_RCNN_NMS_IOU_THRESHOLD, '| ROI', cfgs.ROI_SIZE, '| KEEP_PROB', cfgs.KEEP_PROB)
print('Threshold mac dinh:', cfgs.SHOW_SCORE_THRSHOLD)
print('-' * 72)
print('Dang nap graph + checkpoint (mat ~40s)...')
det = Detector()
print('Checkpoint file:', det.ckpt_path)
print('Checkpoint loaded:', det.ckpt_loaded)
print('So tensor (variables):', det.num_vars)
import os as _os
d = _os.path.join(HERE, 'output', 'trained_weights', cfgs.VERSION)
total = 0
for f in sorted(_os.listdir(d)):
    p = _os.path.join(d, f)
    sz = _os.path.getsize(p)
    total += sz
    print('  %-32s %10.1f MB' % (f, sz / 1048576.0))
print('Tong dung luong checkpoint: %.1f MB' % (total / 1048576.0))
print('Device hien tai:', 'GPU' if det.gpu_available() else 'CPU (GTX1650 thieu cuDNN -> chay CPU)')
print('TF version:', tf.__version__)
print('=' * 72)
print('KET LUAN: model = TDD-Net FPN_Res101_0117_OHEM, ResNet-101 + FPN P2-P6 + RPN share-head + FastRCNN FC1024x2.')
print('Khong phai LLM -> khong co "token" nhu ChatGPT. Bang tracking ben duoi dung')
print('"forward pass / tile / ms / box / P-R-F1" lam don vi tuong duong de so phase.')
