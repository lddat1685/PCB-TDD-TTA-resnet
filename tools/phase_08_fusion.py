# -*- coding: utf-8 -*-
"""Phase 08 — fusion khong train: TTA flip (phase07) + class-wise threshold (phase06).

Ly do: fine-tune gradient that (phase 08 goc) can bo PCB du + tfrecord that + cuDNN,
hien chua co (xem bao cao cuoi file + tin nhan kem theo). Trong luc cho du lieu,
phase08 nay ket hop 2 kien truc on dinh nhat hien co, KHONG sua logic goc,
KHONG train lai, chi goi Detector.run() giong phase_loop.

Chay (tu thu muc repo):
  python\\python.exe tools\\phase_08_fusion.py

Xuat: output/phases/phase_08_fusion/{metrics.json,metrics.csv,REPORT.md}
      + append output/phases/usage_log.csv + regen output/phases/PHASES.md
"""
from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(HERE, '.venv', 'Lib', 'site-packages')
import site as _site
_site.addsitedir(SITE)
sys.path.insert(0, HERE)
os.chdir(os.path.join(HERE, 'tools'))

import tf1_compat  # noqa: F401,E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ui.app import Detector  # noqa: E402
from libs.sahi.slicing import nms_per_class  # noqa: E402
from libs.configs import cfgs  # noqa: E402
from libs.label_name_dict.label_dict import NAME_LABEL_MAP  # noqa: E402

PID = 'phase_08_fusion'
PNAME = 'Fusion TTA flip + class-wise (khong train)'
PARAMS = {
    'base_threshold': 0.6,
    'threshold': 0.6,
    'nms_iou': 0.5,
    'tta_flip': True,
    'sahi': False,
    'class_thresholds': {
        'missing_hole': 0.6, 'short': 0.6, 'mouse_bite': 0.6,
        'spur': 0.65, 'open_circuit': 0.8, 'spurious_copper': 0.8,
    },
}
DELTA = ('Fusion phase06+phase07: chay baseline 2 lan (goc + lat ngang), '
         'flip box ve, loc class-wise (nhom manh 0.6 / spur 0.65 / nhom yeu 0.8), '
         'gop NMS theo lop 0.5. Khong slice, khong train.')

ARCH_CORE = [
    ("Backbone", "ResNet-101 (`resnet_v1_101`): conv1 7x7 s2 + maxpool; block1 3u/d64, block2 4u/d128, block3 23u/d256, block4 3u/d512 s1. BN dong bang (is_training=False, trainable=False)"),
    ("Feature tap", "C2/C3/C4/C5 tai bottleneck cuoi moi block (`libs/networks/resnet.py:resnet_base`)"),
    ("FPN", "P5=conv1x1/256(C5); P6=maxpool(P5,s2); P4..P2=fusion_two_layer (upsample + conv1x1 + 0.5*sum) + conv3x3/256 fuse. Levels P2-P6, stride 4/8/16/32/64"),
    ("Anchor", "BASE [15,25,40,60,80] x SCALES [2,3,4] x RATIOS [2,3,4,5] = 12 anchor/diem (`cfgs.py`)"),
    ("RPN head", "Share heads: conv3x3/512 + conv1x1 cls(24) + conv1x1 reg(48). NMS 0.7, test topK 6000->1000 (`build_whole_network.py`)"),
    ("ROI", "assign_levels theo dien tich 4+log2(sqrt(wh)/224); ROI-warp; concat 4 level"),
    ("FastRCNN head", "`restnet_head`: flatten + FC1024 + FC1024, cls_fc(7) + reg_fc(28). Dropout KEEP_PROB=1.0 (tat)"),
    ("Post", "decode + clip + NMS theo lop IoU 0.3, toi da 100 box/lop; loc score>=threshold (graph tensor)"),
    ("Preprocess", "short_side_resize ve canh ngan 600, max 3000; tru PIXEL_MEAN [21.25, 85.936, 28.729]"),
    ("Checkpoint", "FPN_Res101_0117_OHEM/pcb_30000model.ckpt (tai su dung, khong sua)"),
]


def map_to_original(res, W, H):
    rw, rh = res['size']
    sx, sy = W / float(rw), H / float(rh)
    out = []
    for d in res['detections']:
        x1, y1, x2, y2 = d['box']
        out.append({'box': [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
                    'score': d['score'], 'label': d['label'],
                    'name': d.get('name', ''), 'color': d.get('color', '#888888')})
    return out


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    s = root.find('size')
    w = int(s.find('width').text)
    h = int(s.find('height').text)
    objs = []
    for o in root.findall('object'):
        bb = o.find('bndbox')
        objs.append({'name': o.find('name').text,
                     'label': int(NAME_LABEL_MAP.get(o.find('name').text, -1)),
                     'box': [float(bb.find('xmin').text), float(bb.find('ymin').text),
                             float(bb.find('xmax').text), float(bb.find('ymax').text)]})
    return objs, w, h


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def match_dets(dets, gts, thr=0.5):
    gm = [False] * len(gts); tp = 0; fp = 0
    for d in sorted(dets, key=lambda x: -x['score']):
        best = -1; bi = 0.0
        for gi, g in enumerate(gts):
            if gm[gi]:
                continue
            if int(g['label']) != int(d['label']):
                continue
            v = iou(d['box'], g['box'])
            if v > bi:
                bi = v; best = gi
        if best >= 0 and bi >= thr:
            gm[best] = True; tp += 1
        else:
            fp += 1
    return tp, fp, gm.count(False)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return p, r, 2 * p * r / (p + r) if (p + r) else 0.0


def collect_images(data_dirs):
    files = []
    for d in data_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                files.append(os.path.join(d, f))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iou_thr', type=float, default=0.5)
    ap.add_argument('--out', default='../output/phases')
    ap.add_argument('--annotation_dir', default='test_annotation')
    args = ap.parse_args()

    print('=' * 72)
    print('PHASE 08 — %s' % PNAME)
    print('params:', json.dumps(PARAMS, ensure_ascii=False))
    print('=' * 72)
    print('Dang nap model goc (mat ~40s)...')
    det = Detector()
    print('Model ready:', det.ckpt_path)

    img_files = collect_images(['../input', '../tools/demos', '..'])
    seen = {}; uniq = []
    for p in img_files:
        b = os.path.basename(p)
        if b not in seen:
            seen[b] = p; uniq.append(p)
    img_files = uniq
    print('Anh test (%d): %s' % (len(img_files), [os.path.basename(p) for p in img_files]))

    cthr = PARAMS['class_thresholds']
    base_thr = PARAMS['base_threshold']
    nms_iou = PARAMS['nms_iou']

    rows = []
    t_tp = t_fp = t_fn = t_gt = 0
    t_ms = 0; t_n = 0; t_tiles = 0; t_raw = 0
    for fp in img_files:
        fname = os.path.basename(fp)
        rgb = np.array(Image.open(fp).convert('RGB'))
        H, W = rgb.shape[0], rgb.shape[1]
        r1 = det.run(rgb, base_thr)
        d1 = map_to_original(r1, W, H)
        rgb_flip = np.ascontiguousarray(rgb[:, ::-1, :])
        r2 = det.run(rgb_flip, base_thr)
        d2 = map_to_original(r2, W, H)
        d2b = []
        for d in d2:
            x1, y1, x2, y2 = d['box']
            d2b.append({'box': [W - x2, y1, W - x1, y2],
                        'score': d['score'], 'label': d['label'],
                        'name': d.get('name', ''), 'color': d.get('color', '#888888')})
        merged = d1 + d2b
        filt = [d for d in merged if d['score'] >= float(cthr.get(d.get('name'), base_thr))]
        dets = nms_per_class(filt, iou_threshold=nms_iou)
        dets.sort(key=lambda d: -d['score'])
        ms = r1['elapsed_ms'] + r2['elapsed_ms']
        tiles = 2; raw = len(d1) + len(d2)

        scores = [d['score'] for d in dets]
        mean_s = sum(scores) / len(scores) if scores else 0.0
        cnt = Counter([d['name'] for d in dets])
        xml_p = os.path.join(args.annotation_dir, os.path.splitext(fname)[0] + '.xml')
        has_gt = os.path.isfile(xml_p)
        if has_gt:
            gt_raw, gw, gh = parse_voc_xml(xml_p)
            sx = W / float(gw); sy = H / float(gh)
            gt = [{'name': g['name'], 'label': g['label'],
                   'box': [g['box'][0] * sx, g['box'][1] * sy, g['box'][2] * sx, g['box'][3] * sy]}
                  for g in gt_raw]
            tp, fp_, fn_ = match_dets(dets, gt, args.iou_thr)
            p, r, f = prf(tp, fp_, fn_)
            t_tp += tp; t_fp += fp_; t_fn += fn_; t_gt += len(gt)
        else:
            tp = fp_ = fn_ = None; p = r = f = None
        t_ms += ms; t_n += len(dets); t_tiles += tiles; t_raw += raw
        rows.append({'image': fname, 'size': '%dx%d' % (W, H), 'gt': len(gt) if has_gt else '-',
                     'n': len(dets), 'ms': ms, 'tiles': tiles, 'raw': raw,
                     'mean': round(mean_s, 4), 'counts': dict(cnt),
                     'tp': tp, 'fp': fp_, 'fn': fn_, 'p': p, 'r': r, 'f': f, 'has_gt': has_gt,
                     'dets': [{'name': d['name'], 'score': round(d['score'], 4),
                               'box': [round(v, 1) for v in d['box']]} for d in dets]})
        if has_gt:
            print('  %-22s %s GT=%d n=%d TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f %dms t=%d raw=%d %s'
                  % (fname, '%dx%d' % (W, H), len(gt), len(dets), tp, fp_, fn_, p, r, f, ms, tiles, raw, dict(cnt)))
        else:
            print('  %-22s %s n=%d mean=%.3f %dms t=%d raw=%d %s'
                  % (fname, '%dx%d' % (W, H), len(dets), mean_s, ms, tiles, raw, dict(cnt)))

    n_img = len(rows)
    mp, mr, mf = prf(t_tp, t_fp, t_fn) if t_gt else (None, None, None)
    rec = {'id': PID, 'name': PNAME, 'params': PARAMS, 'delta': DELTA,
           'n_img': n_img, 'total_box': t_n, 'avg_ms': round(t_ms / max(1, n_img), 1),
           'avg_tiles': round(t_tiles / max(1, n_img), 2),
           'gt_total': t_gt, 'tp': t_tp, 'fp': t_fp, 'fn': t_fn, 'p': mp, 'r': mr, 'f': mf}

    pdir = os.path.join(args.out, PID)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, 'metrics.csv'), 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['image', 'size', 'gt', 'n', 'ms', 'tiles', 'raw', 'mean', 'counts',
                    'TP', 'FP', 'FN', 'P', 'R', 'F1'])
        for r in rows:
            w.writerow([r['image'], r['size'], r['gt'], r['n'], r['ms'], r['tiles'], r['raw'],
                        r['mean'], json.dumps(r['counts'], ensure_ascii=False),
                        r['tp'], r['fp'], r['fn'], r['p'], r['r'], r['f']])
    with open(os.path.join(pdir, 'metrics.json'), 'w') as jf:
        json.dump({'phase': rec, 'rows': rows}, jf, indent=2, ensure_ascii=False)

    import datetime as _dt
    _logp = os.path.join(args.out, 'usage_log.csv')
    _need_head = not os.path.isfile(_logp)
    with open(_logp, 'a', newline='') as lf:
        _w = csv.writer(lf)
        if _need_head:
            _w.writerow(['timestamp', 'phase_id', 'phase_name', 'n_img',
                         'forwards_tiles', 'raw_dets', 'ms_total', 'ms_per_img',
                         'boxes', 'mean_ms_per_box', 'GT', 'TP', 'FP', 'FN',
                         'P', 'R', 'F1', 'params'])
        _w.writerow([_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                     PID, PNAME, n_img, t_tiles, t_raw, t_ms,
                     round(t_ms / max(1, n_img), 1), t_n,
                     round(t_ms / max(1, t_n), 1), t_gt, t_tp, t_fp, t_fn,
                     mp, mr, mf, json.dumps(PARAMS, ensure_ascii=False)])

    with open(os.path.join(pdir, 'REPORT.md'), 'w') as md:
        md.write('# %s — %s\n\n' % (PID, PNAME))
        md.write('Tham so: `%s`\n\nY tuong: %s\n\n' % (json.dumps(PARAMS, ensure_ascii=False), DELTA))
        md.write('Checkpoint: `%s` (tai su dung, khong sua)\n\n' % det.ckpt_path)
        md.write('## 1. Cong nghe loi + layer\n\n')
        md.write('| Nhom | Chi tiet |\n|---|---|\n')
        for k, v in ARCH_CORE:
            md.write('| %s | %s |\n' % (k, v))
        md.write('| Phase 08 fusion | TTA flip x2 (goc+lat ngang, flip box `x1=W-x2`) + loc class-wise `%s`, NMS gop %.2f, loc score>=%.2f. Khong slice, khong train |\n'
                 % (json.dumps(PARAMS['class_thresholds'], ensure_ascii=False), nms_iou, base_thr))
        md.write('\n## 2. Ket qua tung anh (IoU>=%.2f, he toa do anh goc)\n\n' % args.iou_thr)
        md.write('| anh | kich thuoc | GT | n box | ms | tiles/raw | mean score | phan bo lop | P | R | F1 |\n')
        md.write('|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            if r['has_gt']:
                md.write('| %s | %s | %s | %d | %d | %d/%d | %.3f | %s | %.3f | %.3f | %.3f |\n'
                         % (r['image'], r['size'], r['gt'], r['n'], r['ms'], r['tiles'], r['raw'],
                            r['mean'], json.dumps(r['counts'], ensure_ascii=False), r['p'], r['r'], r['f']))
            else:
                md.write('| %s | %s | - | %d | %d | %d/%d | %.3f | %s | - | - | - |\n'
                         % (r['image'], r['size'], r['n'], r['ms'], r['tiles'], r['raw'],
                            r['mean'], json.dumps(r['counts'], ensure_ascii=False)))
        md.write('\n## 3. Tong hop phase\n\n')
        md.write('- So anh: %d, tong box: %d, TB %.1f ms/anh, TB %.2f tile/anh\n'
                 % (rec['n_img'], rec['total_box'], rec['avg_ms'], rec['avg_tiles']))
        if rec['gt_total']:
            md.write('- Micro tren anh co GT (tong GT=%d): TP=%d FP=%d FN=%d -> P=%.3f R=%.3f F1=%.3f\n'
                     % (rec['gt_total'], rec['tp'], rec['fp'], rec['fn'], rec['p'], rec['r'], rec['f']))
        md.write('\n## 4. Nhan xet / quyet dinh\n\n_Phieu nay do nguoi chay dien sau khi xem bang._\n')

    # ---- regen PHASES.md tu TAT CA metrics.json (gom ca phase cu) ----
    allp = []
    for pid in sorted(os.listdir(args.out)):
        mp_ = os.path.join(args.out, pid, 'metrics.json')
        if os.path.isfile(mp_):
            with open(mp_) as f:
                allp.append(json.load(f))
    allp.sort(key=lambda d: d['phase']['id'])
    with open(os.path.join(args.out, 'PHASES.md'), 'w') as md:
        md.write('# Vong lap Phase — TDD-Net PCB\n\n')
        md.write('Checkpoint goc giu nguyen moi phase. IoU TP>=%.2f. He toa do anh goc.\n\n' % args.iou_thr)
        md.write('| phase | ten | tham so | box | ms/anh | tile/anh | P | R | F1 |\n')
        md.write('|---|---|---|---|---|---|---|---|---|\n')
        for d in allp:
            r = d['phase']
            md.write('| %s | %s | `%s` | %d | %.1f | %.2f | %s | %s | %s |\n'
                     % (r['id'], r['name'], json.dumps(r['params'], ensure_ascii=False),
                        r['total_box'], r['avg_ms'], r['avg_tiles'],
                        ('%.3f' % r['p']) if r['p'] is not None else '-',
                        ('%.3f' % r['r']) if r['r'] is not None else '-',
                        ('%.3f' % r['f']) if r['f'] is not None else '-'))
        md.write('\n## Bang chi tiet tung anh (tat ca phase)\n\n')
        md.write('| anh | size | GT | ' + ' | '.join([d['phase']['id'] + ' (n/ms)' for d in allp]) + ' |\n')
        md.write('|---|---|---|' + '|'.join(['---'] * len(allp)) + '|\n')
        for i in range(len(allp[0]['rows'])):
            cells = ['%d/%d' % (d['rows'][i]['n'], d['rows'][i]['ms']) for d in allp]
            r0 = allp[0]['rows'][i]
            md.write('| %s | %s | %s | %s |\n' % (r0['image'], r0['size'], r0['gt'], ' | '.join(cells)))

    print('=' * 72)
    print('  -> Tong: %d box, %.1f ms/anh' % (t_n, t_ms / max(1, n_img))
          + (', GT=%d P=%.3f R=%.3f F1=%.3f' % (t_gt, mp, mr, mf) if t_gt else ' (khong GT)'))
    print('XONG phase 08. PHASES.md da regen gom %d phase.' % len(allp))
    return 0


if __name__ == '__main__':
    sys.exit(main())
