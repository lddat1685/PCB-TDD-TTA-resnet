# -*- coding: utf-8 -*-
"""CLI inference SAHI cho TDD-Net (file MOI, khong sua inference.py goc).

Chay:  (tu thu muc tools/)
  ../python/python.exe inference_sahi.py --data_dir ../input --save_dir ../output/sahi_results

Moi anh: cat slice -> chay Detector goc (cung graph/checkpoint) -> NMS gop.
Anh luu ra co ve box tren he anh GOC.
"""
from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tf1_compat  # noqa: F401  phai truoc tensorflow

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))

from ui.app import Detector  # noqa: E402  tai su dung graph/checkpoint goc
from libs.sahi.sahi_detector import sahi_detect  # noqa: E402
from libs.label_name_dict.label_dict import LABEL_NAME_MAP  # noqa: E402
from libs.configs import cfgs  # noqa: E402


CLASS_DRAW_COLOR = {
    1: (77, 77, 255),
    2: (20, 173, 250),
    3: (26, 193, 82),
    4: (194, 194, 19),
    5: (222, 82, 146),
    6: (255, 24, 24),
}


def draw_on_original(bgr_img, dets):
    out = bgr_img.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(round(v)) for v in d['box']]
        color = CLASS_DRAW_COLOR.get(int(d['label']), (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        txt = '%s:%.2f' % (d.get('name', str(d['label'])), d['score'])
        cv2.putText(out, txt, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description='SAHI inference cho TDD-Net (khong sua baseline)')
    ap.add_argument('--data_dir', default='../input')
    ap.add_argument('--save_dir', default='../output/sahi_results')
    ap.add_argument('--threshold', type=float, default=float(cfgs.SHOW_SCORE_THRSHOLD))
    ap.add_argument('--slice', type=int, default=512, help='kich thuoc slice vuong')
    ap.add_argument('--overlap', type=float, default=0.2)
    ap.add_argument('--nms_iou', type=float, default=0.5)
    ap.add_argument('--no_full', action='store_true', help='khong chay them anh full')
    args = ap.parse_args()

    print('Dang nap model goc (mat ~40s)...')
    det = Detector()
    print('Model ready. checkpoint:', det.ckpt_path)

    imgs = [os.path.join(args.data_dir, f) for f in sorted(os.listdir(args.data_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
    assert imgs, 'khong thay anh trong %s' % args.data_dir
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    for i, p in enumerate(imgs):
        rgb = np.array(Image.open(p).convert('RGB'))
        t0 = time.time()
        r = sahi_detect(det, rgb, threshold=args.threshold,
                        slice_h=args.slice, slice_w=args.slice,
                        overlap=args.overlap, nms_iou=args.nms_iou,
                        include_full_image=(not args.no_full))
        wall = (time.time() - t0) * 1000.0
        bgr = cv2.imread(p)
        vis = draw_on_original(bgr, r['detections'])
        out_p = os.path.join(args.save_dir, os.path.basename(p))
        cv2.imwrite(out_p, vis)
        summ = {}
        for d in r['detections']:
            summ[d['name']] = summ.get(d['name'], 0) + 1
        print('[%d/%d] %s: %d box %s | tiles=%d raw=%d slice_ms=%d wall=%dms -> %s'
              % (i + 1, len(imgs), os.path.basename(p), len(r['detections']),
                 summ, r['num_slices'], r['raw_count'], r['slice_ms'],
                 int(wall), out_p))


if __name__ == '__main__':
    main()
