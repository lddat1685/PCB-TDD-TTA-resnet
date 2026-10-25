# -*- coding: utf-8 -*-
"""Benchmark Baseline vs SAHI cho TDD-Net (file MOI, khong sua logic goc).

So sanh cong bang o he toa do ANH GOC:
  baseline: Detector.run -> map box resized ve anh goc
  sahi    : cat slice -> Detector goc -> map ve anh goc -> NMS

Neu co XML ground-truth cung ten file trong --annotation_dir thi tinh
TP/FP/FN (IoU>=0.5, greedy theo score), Precision/Recall/F1.
Neu khong co GT thi chi so sanh so box / score / thoi gian (proxy recall).

Chay (tu thu muc tools/):
  ../python/python.exe benchmark_sahi.py --data_dir ../input --annotation_dir test_annotation

Ket qua in ra console dang markdown + luu csv/md vao ../output/sahi_eval/
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import sys
import time
import xml.etree.ElementTree as ET

# --- nap .venv giong launcher.py (portable, khong can cai dat) ---
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(HERE, '.venv', 'Lib', 'site-packages')
import site as _site
_site.addsitedir(SITE)
sys.path.insert(0, HERE)
_ORIG_CWD = os.getcwd()
os.chdir(os.path.join(HERE, 'tools'))

import tf1_compat  # noqa: F401,E402  phai truoc tensorflow
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ui.app import Detector  # noqa: E402  tai su dung graph/checkpoint goc
from libs.sahi.sahi_detector import sahi_detect, map_baseline_to_original  # noqa: E402
from libs.configs import cfgs  # noqa: E402
from libs.label_name_dict.label_dict import NAME_LABEL_MAP, LABEl_NAME_MAP  # noqa: E402

CLASS_IDS = [i for i in sorted(LABEl_NAME_MAP) if i != 0]
CLASS_NAMES = [LABEl_NAME_MAP[i] for i in CLASS_IDS]


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size_el = root.find('size')
    w = int(size_el.find('width').text) if size_el is not None else None
    h = int(size_el.find('height').text) if size_el is not None else None
    objs = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bb = obj.find('bndbox')
        objs.append({
            'name': name,
            'label': int(NAME_LABEL_MAP.get(name, -1)),
            'box': [float(bb.find('xmin').text), float(bb.find('ymin').text),
                    float(bb.find('xmax').text), float(bb.find('ymax').text)],
        })
    return objs, w, h


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def match_dets_to_gt(dets, gts, iou_thr=0.5):
    """Greedy theo score. Tra ve (tp, fp, fn) + chi tiet tung cap khop."""
    gt_matched = [False] * len(gts)
    tp = 0; fp = 0
    for d in sorted(dets, key=lambda x: -x['score']):
        best = -1; best_iou = 0.0
        for gi, g in enumerate(gts):
            if gt_matched[gi]:
                continue
            if int(g['label']) != int(d['label']):
                continue
            v = iou(d['box'], g['box'])
            if v > best_iou:
                best_iou = v; best = gi
        if best >= 0 and best_iou >= iou_thr:
            gt_matched[best] = True
            tp += 1
        else:
            fp += 1
    fn = gt_matched.count(False)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser(description='Benchmark Baseline vs SAHI (TDD-Net)')
    ap.add_argument('--data_dir', default='../input')
    ap.add_argument('--annotation_dir', default='test_annotation')
    ap.add_argument('--threshold', type=float, default=float(cfgs.SHOW_SCORE_THRSHOLD))
    ap.add_argument('--slice', type=int, default=512)
    ap.add_argument('--overlap', type=float, default=0.2)
    ap.add_argument('--nms_iou', type=float, default=0.5)
    ap.add_argument('--iou_thr', type=float, default=0.5, help='nguong IoU tinh TP')
    ap.add_argument('--out_dir', default='../output/sahi_eval')
    args = ap.parse_args()

    print('=' * 72)
    print('Benchmark Baseline vs SAHI — TDD-Net (6 lop loi PCB)')
    print('  threshold=%.2f slice=%d overlap=%.2f nms_iou=%.2f iou_thr=%.2f'
          % (args.threshold, args.slice, args.overlap, args.nms_iou, args.iou_thr))
    print('=' * 72)
    print('Dang nap model goc (mat ~40s)...')
    det = Detector()
    print('Model ready. checkpoint:', det.ckpt_path)

    imgs = sorted([f for f in os.listdir(args.data_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))])
    # them anh demo o thu muc goc neu data_dir khong co
    if not imgs:
        print('KHONG thay anh trong', args.data_dir)
        return 1
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    rows = []
    tot = {'base_tp': 0, 'base_fp': 0, 'base_fn': 0,
           'sahi_tp': 0, 'sahi_fp': 0, 'sahi_fn': 0,
           'base_ms': 0, 'sahi_ms': 0, 'base_n': 0, 'sahi_n': 0, 'gt_n': 0}
    per_class = {c: {'gt': 0, 'base_tp': 0, 'sahi_tp': 0, 'base_fp': 0, 'sahi_fp': 0}
                 for c in CLASS_NAMES}

    for fname in imgs:
        p = os.path.join(args.data_dir, fname)
        rgb = np.array(Image.open(p).convert('RGB'))
        H, W = rgb.shape[0], rgb.shape[1]

        # ---- baseline (he goc) ----
        b_res = det.run(rgb, args.threshold)
        b_dets = map_baseline_to_original(b_res, W, H)
        b_ms = b_res['elapsed_ms']

        # ---- sahi (he goc) ----
        s_res = sahi_detect(det, rgb, threshold=args.threshold,
                            slice_h=args.slice, slice_w=args.slice,
                            overlap=args.overlap, nms_iou=args.nms_iou,
                            include_full_image=True)
        s_dets = s_res['detections']
        s_ms = s_res['total_ms']

        # ---- ground truth (neu co) ----
        stem = os.path.splitext(fname)[0]
        xml_p = os.path.join(args.annotation_dir, stem + '.xml')
        gt = []
        has_gt = os.path.isfile(xml_p)
        if has_gt:
            gt_raw, gw, gh = parse_voc_xml(xml_p)
            sx = float(W) / float(gw) if gw else 1.0
            sy = float(H) / float(gh) if gh else 1.0
            for g in gt_raw:
                x1, y1, x2, y2 = g['box']
                gt.append({'name': g['name'], 'label': g['label'],
                           'box': [x1 * sx, y1 * sy, x2 * sx, y2 * sy]})
            for g in gt:
                if g['name'] in per_class:
                    per_class[g['name']]['gt'] += 1

        if has_gt:
            btp, bfp, bfn = match_dets_to_gt(b_dets, gt, args.iou_thr)
            stp, sfp, sfn = match_dets_to_gt(s_dets, gt, args.iou_thr)
            bp, br, bf = prf(btp, bfp, bfn)
            sp, sr, sf = prf(stp, sfp, sfn)
            tot['base_tp'] += btp; tot['base_fp'] += bfp; tot['base_fn'] += bfn
            tot['sahi_tp'] += stp; tot['sahi_fp'] += sfp; tot['sahi_fn'] += sfn
            tot['gt_n'] += len(gt)
            # per-class tp (don gian: dem theo tung lop rieng)
            for c in CLASS_NAMES:
                gtc = [g for g in gt if g['name'] == c]
                bdt = [d for d in b_dets if d.get('name') == c]
                sdt = [d for d in s_dets if d.get('name') == c]
                btp_c, bfp_c, _ = match_dets_to_gt(bdt, gtc, args.iou_thr)
                stp_c, sfp_c, _ = match_dets_to_gt(sdt, gtc, args.iou_thr)
                per_class[c]['base_tp'] += btp_c; per_class[c]['base_fp'] += bfp_c
                per_class[c]['sahi_tp'] += stp_c; per_class[c]['sahi_fp'] += sfp_c
        else:
            btp = bfp = bfn = stp = sfp = sfn = None
            bp = br = bf = sp = sr = sf = None

        tot['base_ms'] += b_ms; tot['sahi_ms'] += s_ms
        tot['base_n'] += len(b_dets); tot['sahi_n'] += len(s_dets)

        b_scores = [d['score'] for d in b_dets]
        s_scores = [d['score'] for d in s_dets]
        rows.append({
            'image': fname, 'size': '%dx%d' % (W, H),
            'gt': len(gt) if has_gt else '-',
            'base_n': len(b_dets), 'sahi_n': len(s_dets),
            'base_ms': b_ms, 'sahi_ms': s_ms,
            'tiles': s_res['num_slices'], 'raw': s_res['raw_count'],
            'base_mean': (sum(b_scores) / len(b_scores)) if b_scores else 0.0,
            'sahi_mean': (sum(s_scores) / len(s_scores)) if s_scores else 0.0,
            'btp': btp, 'bfp': bfp, 'bfn': bfn, 'bp': bp, 'br': br, 'bf': bf,
            'stp': stp, 'sfp': sfp, 'sfn': sfn, 'sp': sp, 'sr': sr, 'sf': sf,
            'has_gt': has_gt,
        })
        if has_gt:
            print('%-22s %s GT=%d | base: n=%d TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f %dms | sahi: n=%d TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f %dms(tiles=%d raw=%d)'
                  % (fname, '%dx%d' % (W, H), len(gt),
                     len(b_dets), btp, bfp, bfn, bp, br, bf, b_ms,
                     len(s_dets), stp, sfp, sfn, sp, sr, sf, s_ms,
                     s_res['num_slices'], s_res['raw_count']))
        else:
            print('%-22s %s (khong co GT) | base: n=%d mean=%.3f %dms | sahi: n=%d mean=%.3f %dms (tiles=%d raw=%d)'
                  % (fname, '%dx%d' % (W, H),
                     len(b_dets), (sum(b_scores) / len(b_scores)) if b_scores else 0.0, b_ms,
                     len(s_dets), (sum(s_scores) / len(s_scores)) if s_scores else 0.0, s_ms,
                     s_res['num_slices'], s_res['raw_count']))

    n_img = len(rows)
    print('')
    print('-' * 72)
    print('TONG HOP (%d anh):' % n_img)
    print('  Baseline: %d box, %.1f ms/anh | SAHI: %d box, %.1f ms/anh (cham hon x%.2f)'
          % (tot['base_n'], tot['base_ms'] / max(1, n_img),
             tot['sahi_n'], tot['sahi_ms'] / max(1, n_img),
             (tot['sahi_ms'] / max(1, tot['base_ms']))))
    if tot['gt_n'] > 0:
        bp, br, bf = prf(tot['base_tp'], tot['base_fp'], tot['base_fn'])
        sp, sr, sf = prf(tot['sahi_tp'], tot['sahi_fp'], tot['sahi_fn'])
        print('  GT tong: %d box. Baseline micro: P=%.3f R=%.3f F1=%.3f (TP=%d FP=%d FN=%d)'
              % (tot['gt_n'], bp, br, bf, tot['base_tp'], tot['base_fp'], tot['base_fn']))
        print('  GT tong: %d box. SAHI     micro: P=%.3f R=%.3f F1=%.3f (TP=%d FP=%d FN=%d)'
              % (tot['gt_n'], sp, sr, sf, tot['sahi_tp'], tot['sahi_fp'], tot['sahi_fn']))
        print('  Delta Recall (SAHI-base): %+.3f | Delta Precision: %+.3f | Delta F1: %+.3f'
              % (sr - br, sp - bp, sf - bf))

    # ---- viet csv + md ----
    csv_p = os.path.join(args.out_dir, 'benchmark.csv')
    with open(csv_p, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image', 'size', 'gt', 'base_n', 'sahi_n', 'base_ms', 'sahi_ms',
                    'tiles', 'raw', 'base_mean', 'sahi_mean',
                    'base_TP', 'base_FP', 'base_FN', 'base_P', 'base_R', 'base_F1',
                    'sahi_TP', 'sahi_FP', 'sahi_FN', 'sahi_P', 'sahi_R', 'sahi_F1'])
        for r in rows:
            w.writerow([r['image'], r['size'], r['gt'], r['base_n'], r['sahi_n'],
                        r['base_ms'], r['sahi_ms'], r['tiles'], r['raw'],
                        '%.4f' % r['base_mean'], '%.4f' % r['sahi_mean'],
                        r['btp'], r['bfp'], r['bfn'], r['bp'], r['br'], r['bf'],
                        r['stp'], r['sfp'], r['sfn'], r['sp'], r['sr'], r['sf']])
    md_p = os.path.join(args.out_dir, 'benchmark.md')
    with open(md_p, 'w') as f:
        f.write('# Benchmark Baseline vs SAHI — TDD-Net\n\n')
        f.write('threshold=%.2f slice=%d overlap=%.2f nms_iou=%.2f iou_thr=%.2f\n\n'
                % (args.threshold, args.slice, args.overlap, args.nms_iou, args.iou_thr))
        f.write('| anh | kich thuoc | GT | baseline (n / ms) | SAHI (n / ms) | tiles/raw | P base->sahi | R base->sahi | F1 base->sahi |\n')
        f.write('|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            if r['has_gt']:
                f.write('| %s | %s | %s | %d / %d | %d / %d | %d/%d | %.3f->%.3f | %.3f->%.3f | %.3f->%.3f |\n'
                        % (r['image'], r['size'], r['gt'], r['base_n'], r['base_ms'],
                           r['sahi_n'], r['sahi_ms'], r['tiles'], r['raw'],
                           r['bp'], r['sp'], r['br'], r['sr'], r['bf'], r['sf']))
            else:
                f.write('| %s | %s | - | %d / %d | %d / %d | %d/%d | - | - | - |\n'
                        % (r['image'], r['size'], r['base_n'], r['base_ms'],
                           r['sahi_n'], r['sahi_ms'], r['tiles'], r['raw']))
        f.write('\n## Theo lop loi (chi nhung anh co GT)\n\n')
        f.write('| lop | GT | base TP | sahi TP | base FP | sahi FP |\n')
        f.write('|---|---|---|---|---|---|\n')
        for c in CLASS_NAMES:
            d = per_class[c]
            f.write('| %s | %d | %d | %d | %d | %d |\n'
                    % (c, d['gt'], d['base_tp'], d['sahi_tp'], d['base_fp'], d['sahi_fp']))
    print('Da luu: %s , %s' % (csv_p, md_p))
    return 0


if __name__ == '__main__':
    sys.exit(main())
