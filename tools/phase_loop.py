# -*- coding: utf-8 -*-
"""Vong lap danh gia theo phase cho TDD-Net (file MOI, khong sua logic goc).

Moi phase = 1 cau hinh inference tren cung checkpoint goc.
- Phase 01: baseline TDD-Net (FPN_Res101_0117_OHEM), threshold 0.6
- Phase 02: + SAHI mac dinh (slice 512, overlap 0.2, nms 0.5, thr 0.6, co full)
- Phase 03: SAHI precision (thr 0.7, nms 0.4) de tri FP
- Phase 04: SAHI slice 640 overlap 0.25 (it tile hon, nhanh hon)
- Phase 05: SAHI bo anh full (ablation: chi slice)

Chay (tu thu muc repo):
  python\\python.exe tools\\phase_loop.py

Ket qua: output/phases/<phase_id>/REPORT.md + metrics.json/csv
         output/phases/PHASES.md (bang so sanh + de xuat phase tiep theo)
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
from libs.sahi.sahi_detector import sahi_detect, map_baseline_to_original  # noqa: E402
from libs.sahi.slicing import nms_per_class  # noqa: E402
from libs.configs import cfgs  # noqa: E402
from libs.label_name_dict.label_dict import NAME_LABEL_MAP, LABEl_NAME_MAP  # noqa: E402

CLASS_IDS = [i for i in sorted(LABEl_NAME_MAP) if i != 0]
CLASS_NAMES = [LABEl_NAME_MAP[i] for i in CLASS_IDS]

# ---------------------------------------------------------------- kien truc loi (dung chung, trich tu code)
ARCH_CORE = [
    ("Backbone", "ResNet-101 (`resnet_v1_101`): conv1 7x7 s2 + maxpool; block1 3u/d64, block2 4u/d128, block3 23u/d256, block4 3u/d512 s1. BN dong bang (is_training=False, trainable=False)"),
    ("Feature tap", "C2/C3/C4/C5 tai bottleneck cuoi moi block (`libs/networks/resnet.py:resnet_base`)"),
    ("FPN", "P5=conv1x1/256(C5); P6=maxpool(P5,s2); P4..P2=fusion_two_layer (upsample + conv1x1 + 0.5*sum) + conv3x3/256 fuse. Levels P2-P6, stride 4/8/16/32/64"),
    ("Anchor", "BASE [15,25,40,60,80] x SCALES [2,3,4] x RATIOS [2,3,4,5] = 12 anchor/diem (`cfgs.py`)"),
    ("RPN head", "Share heads: conv3x3/512 + conv1x1 cls(24) + conv1x1 reg(48). NMS 0.7, test topK 6000->1000 (`build_whole_network.py`)"),
    ("ROI", "assign_levels theo dien tich 4+log2(sqrt(wh)/224); ROI-warp; concat 4 level"),
    ("FastRCNN head", "`restnet_head`: flatten + FC1024 + FC1024, cls_fc(7) + reg_fc(28). Dropout KEEP_PROB=1.0 (tat)"),
    ("Post", "decode + clip + NMS theo lop IoU 0.3, toi da 100 box/lop; loc score>=threshold (graph tensor)"),
    ("Preprocess", "short_side_resize ve canh ngan 600, max 3000; tru PIXEL_MEAN [21.25,85.936,28.729]"),
    ("Checkpoint", "FPN_Res101_0117_OHEM/pcb_30000model.ckpt (680 tensor, ResNet101+FPN+FastRCNN, 6 loi + background)"),
]

PHASES = [
    {"id": "phase_01_baseline",
     "name": "Baseline TDD-Net",
     "kind": "baseline",
     "params": {"threshold": 0.6},
     "delta": "Khong cat slice. 1 forward/anh. Lam moc doi chieu."},
    {"id": "phase_02_sahi_default",
     "name": "SAHI mac dinh",
     "kind": "sahi",
     "params": {"threshold": 0.6, "slice": 512, "overlap": 0.2, "nms_iou": 0.5, "full": True},
     "delta": "+ SAHI: slice 512x512 overlap 0.2 + anh full, NMS gop 0.5. Tang recall loi tiny."},
    {"id": "phase_03_sahi_precision",
     "name": "SAHI precision (tri FP)",
     "kind": "sahi",
     "params": {"threshold": 0.7, "slice": 512, "overlap": 0.2, "nms_iou": 0.4, "full": True},
     "delta": "Nang threshold 0.6->0.7, NMS 0.5->0.4 de cat FP open_circuit/spurious_copper."},
    {"id": "phase_04_sahi_fast",
     "name": "SAHI nhanh (slice 640)",
     "kind": "sahi",
     "params": {"threshold": 0.6, "slice": 640, "overlap": 0.25, "nms_iou": 0.5, "full": True},
     "delta": "Slice 640 overlap 0.25 -> it tile hon (~4-6 thay vi 9-10), giam thoi gian."},
    {"id": "phase_05_sahi_nofull",
     "name": "SAHI ablation (bo anh full)",
     "kind": "sahi",
     "params": {"threshold": 0.6, "slice": 512, "overlap": 0.2, "nms_iou": 0.5, "full": False},
     "delta": "Bo nhanh full-image de do rieng dong gop cua slice (kiem tra trung lap)."},
    {"id": "phase_06_classwise",
     "name": "SAHI class-wise threshold (tri FP co chon loc)",
     "kind": "sahi_classwise",
     "params": {"base_threshold": 0.6, "slice": 640, "overlap": 0.25, "nms_iou": 0.3,
                "full": True,
                "class_thresholds": {"missing_hole": 0.6, "short": 0.6, "mouse_bite": 0.6,
                                     "spur": 0.65, "open_circuit": 0.8, "spurious_copper": 0.8}},
     "delta": "Giu recall nhom manh (missing_hole/short/mouse_bite 0.6), siet FP nhom yeu (open_circuit/spurious_copper 0.8, spur 0.65). NMS 0.3. Nen slice640 nhanh."},
    {"id": "phase_07_tta_flip",
     "name": "TTA flip ngang + baseline (khong train)",
     "kind": "tta_flip",
     "params": {"threshold": 0.6, "nms_iou": 0.5},
     "delta": "Chay baseline 2 lan (goc + lat ngang), flip box ve, gop NMS 0.5. Tang on dinh khong ton train."},
]


def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    s = root.find('size')
    w = int(s.find('width').text); h = int(s.find('height').text)
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
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


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
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--iou_thr', type=float, default=0.5)
    ap.add_argument('--out', default='../output/phases')
    ap.add_argument('--annotation_dir', default='test_annotation')
    args = ap.parse_args()
    for ph in PHASES:
        if ph['kind'] == 'baseline':
            ph['params']['threshold'] = args.threshold

    print('=' * 72)
    print('VONG LAP PHASE — TDD-Net PCB (checkpoint goc, khong sua baseline)')
    print('=' * 72)
    print('Dang nap model goc (mat ~40s)...')
    det = Detector()
    print('Model ready:', det.ckpt_path)

    img_files = collect_images(['../input', '../tools/demos', '..'])
    # loc trung ten (demos va input trung nhau)
    seen = {}; uniq = []
    for p in img_files:
        b = os.path.basename(p)
        if b not in seen:
            seen[b] = p; uniq.append(p)
    img_files = uniq
    print('Anh test (%d): %s' % (len(img_files), [os.path.basename(p) for p in img_files]))

    out_root = args.out
    if not os.path.isdir(out_root):
        os.makedirs(out_root)

    summary = []
    for ph in PHASES:
        pid = ph['id']; pdir = os.path.join(out_root, pid)
        if not os.path.isdir(pdir):
            os.makedirs(pdir)
        print('-' * 72)
        print('%s — %s %s' % (pid, ph['name'], ph['params']))
        rows = []
        t_tp = t_fp = t_fn = t_gt = 0
        t_ms = 0; t_n = 0
        t_tiles = 0; t_raw = 0
        for fp in img_files:
            fname = os.path.basename(fp)
            rgb = np.array(Image.open(fp).convert('RGB'))
            H, W = rgb.shape[0], rgb.shape[1]
            if ph['kind'] == 'baseline':
                th = ph['params']['threshold']
                t0 = time.time()
                res = det.run(rgb, th)
                ms = res['elapsed_ms']
                dets = map_baseline_to_original(res, W, H)
                tiles = 1; raw = len(dets)
            elif ph['kind'] == 'sahi_classwise':
                pr = ph['params']
                cthr = pr.get('class_thresholds', {})
                base_thr = pr.get('base_threshold',
                                  min(cthr.values()) if cthr else 0.6)
                t0 = time.time()
                s = sahi_detect(det, rgb, threshold=base_thr, slice_h=pr['slice'],
                                slice_w=pr['slice'], overlap=pr['overlap'],
                                nms_iou=pr['nms_iou'], include_full_image=pr['full'])
                # loc theo tung lop (tri FP co chon loc)
                filt = [d for d in s['detections']
                        if d['score'] >= float(cthr.get(d.get('name'), base_thr))]
                dets = nms_per_class(filt, iou_threshold=pr['nms_iou'])
                dets.sort(key=lambda d: -d['score'])
                ms = s['total_ms']
                tiles = s['num_slices']; raw = s['raw_count']
            elif ph['kind'] == 'tta_flip':
                pr = ph['params']
                th = pr['threshold']
                t0 = time.time()
                r1 = det.run(rgb, th)
                d1 = map_baseline_to_original(r1, W, H)
                rgb_flip = np.ascontiguousarray(rgb[:, ::-1, :])
                r2 = det.run(rgb_flip, th)
                d2 = map_baseline_to_original(r2, W, H)
                # flip box ve he goc: x' = W - x2
                d2b = []
                for d in d2:
                    x1, y1, x2, y2 = d['box']
                    d2b.append({'box': [W - x2, y1, W - x1, y2],
                                'score': d['score'], 'label': d['label'],
                                'name': d.get('name', ''), 'color': d.get('color', '#888888')})
                dets = nms_per_class(d1 + d2b, iou_threshold=pr['nms_iou'])
                dets.sort(key=lambda d: -d['score'])
                ms = r1['elapsed_ms'] + r2['elapsed_ms']
                tiles = 2; raw = len(d1) + len(d2)
            else:
                pr = ph['params']
                s = sahi_detect(det, rgb, threshold=pr['threshold'], slice_h=pr['slice'],
                                slice_w=pr['slice'], overlap=pr['overlap'],
                                nms_iou=pr['nms_iou'], include_full_image=pr['full'])
                dets = s['detections']; ms = s['total_ms']
                tiles = s['num_slices']; raw = s['raw_count']
            scores = [d['score'] for d in dets]
            mean_s = sum(scores) / len(scores) if scores else 0.0
            cnt = Counter([d['name'] for d in dets])
            # GT
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
        rec = {'id': pid, 'name': ph['name'], 'params': ph['params'], 'delta': ph['delta'],
               'n_img': n_img, 'total_box': t_n, 'avg_ms': round(t_ms / max(1, n_img), 1),
               'avg_tiles': round(t_tiles / max(1, n_img), 2),
               'gt_total': t_gt, 'tp': t_tp, 'fp': t_fp, 'fn': t_fn, 'p': mp, 'r': mr, 'f': mf}
        summary.append((ph, rec, rows))
        # ---- luu csv/json ----
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
        # ---- usage_log.csv (cong don theo thoi gian: chi phi + chi so) ----
        import datetime as _dt
        _logp = os.path.join(out_root, 'usage_log.csv')
        _need_head = not os.path.isfile(_logp)
        with open(_logp, 'a', newline='') as lf:
            _w = csv.writer(lf)
            if _need_head:
                _w.writerow(['timestamp', 'phase_id', 'phase_name', 'n_img',
                             'forwards_tiles', 'raw_dets', 'ms_total', 'ms_per_img',
                             'boxes', 'mean_ms_per_box', 'GT', 'TP', 'FP', 'FN',
                             'P', 'R', 'F1', 'params'])
            _w.writerow([_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         pid, ph['name'], n_img, t_tiles, t_raw, t_ms,
                         round(t_ms / max(1, n_img), 1), t_n,
                         round(t_ms / max(1, t_n), 1), t_gt, t_tp, t_fp, t_fn,
                         mp, mr, mf, json.dumps(ph['params'], ensure_ascii=False)])
        # ---- REPORT.md ----
        with open(os.path.join(pdir, 'REPORT.md'), 'w') as md:
            md.write('# %s — %s\n\n' % (pid, ph['name']))
            md.write('Tham so: `%s`\n\nY tuong: %s\n\n' % (json.dumps(ph['params'], ensure_ascii=False), ph['delta']))
            md.write('Checkpoint: `%s` (tai su dung, khong sua)\n\n' % det.ckpt_path)
            md.write('## 1. Cong nghe loi + layer (Phase 1 lam chuan, cac phase sau giu nguyen)\n\n')
            md.write('| Nhom | Chi tiet |\n|---|---|\n')
            for k, v in ARCH_CORE:
                md.write('| %s | %s |\n' % (k, v))
            if ph['kind'] == 'sahi':
                pr = ph['params']
                md.write('| SAHI (them moi) | `libs/sahi/slicing.py:get_slices` slice %dx%d overlap %.2f; '
                         '`sahi_detector.sahi_detect` chay lai Detector goc tren tung slice + full=%s, '
                         'map box ve anh goc, `merge_sahi_detections` NMS theo lop IoU %.2f, loc score>=%.2f |\n'
                         % (pr['slice'], pr['slice'], pr['overlap'], pr['full'], pr['nms_iou'], pr['threshold']))
            elif ph['kind'] == 'sahi_classwise':
                pr = ph['params']
                md.write('| Phase 06 class-wise | nen slice%d (giong phase_04) + loc theo tung lop `%s`, NMS gop %.2f. '
                         'Giu recall nhom manh, siet FP nhom yeu |\n'
                         % (pr['slice'], json.dumps(pr.get('class_thresholds', {}), ensure_ascii=False),
                            pr['nms_iou']))
            elif ph['kind'] == 'tta_flip':
                pr = ph['params']
                md.write('| Phase 07 TTA flip | chay baseline 2 lan (goc + lat ngang), flip box ve, '
                         '`nms_per_class` gop NMS IoU %.2f, loc score>=%.2f. Khong train, khong slice |\n'
                         % (pr['nms_iou'], pr['threshold']))
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
            else:
                md.write('- Khong co anh GT trong dot nay (chi so proxy: so box/score/thoi gian).\n')
                md.write('- De co P/R/F1 day du 6 lop: them XML VOC cho cac anh input roi chay lai.\n')
            md.write('\n## 4. Nhan xet / quyet dinh\n\n_Phieu nay do nguoi chay dien sau khi xem bang._\n')
        print('  -> Tong: %d box, %.1f ms/anh' % (t_n, t_ms / max(1, n_img))
              + (', GT=%d P=%.3f R=%.3f F1=%.3f' % (t_gt, mp, mr, mf) if t_gt else ' (khong GT)'))

    # ---- PHASES.md tong ----
    with open(os.path.join(out_root, 'PHASES.md'), 'w') as md:
        md.write('# Vong lap Phase — TDD-Net PCB\n\n')
        md.write('Checkpoint goc giu nguyen moi phase. IoU TP>=%.2f. He toa do anh goc.\n\n' % args.iou_thr)
        md.write('| phase | ten | tham so | box | ms/anh | tile/anh | P | R | F1 |\n')
        md.write('|---|---|---|---|---|---|---|---|---|\n')
        for ph, rec, rows in summary:
            md.write('| %s | %s | `%s` | %d | %.1f | %.2f | %s | %s | %s |\n'
                     % (rec['id'], rec['name'], json.dumps(rec['params'], ensure_ascii=False),
                        rec['total_box'], rec['avg_ms'], rec['avg_tiles'],
                        ('%.3f' % rec['p']) if rec['p'] is not None else '-',
                        ('%.3f' % rec['r']) if rec['r'] is not None else '-',
                        ('%.3f' % rec['f']) if rec['f'] is not None else '-'))
        md.write('\n## Bang chi tiet tung anh (tat ca phase)\n\n')
        md.write('| anh | size | GT | ' + ' | '.join([s[1]['id'] + ' (n/ms)' for s in summary]) + ' |\n')
        md.write('|---|---|---|' + '|'.join(['---'] * len(summary)) + '|\n')
        for i in range(len(summary[0][2])):
            cells = []
            for ph, rec, rows in summary:
                r = rows[i]
                cells.append('%d/%d' % (r['n'], r['ms']))
            r0 = summary[0][2][i]
            md.write('| %s | %s | %s | %s |\n' % (r0['image'], r0['size'], r0['gt'], ' | '.join(cells)))
        # de xuat tu dong
        md.write('\n## De xuat tu dong (doc bang tren)\n\n')
        base = summary[0][1]
        best_f = max([s[1] for s in summary if s[1]['f'] is not None], key=lambda x: x['f'], default=None)
        fastest_sahi = min([s[1] for s in summary if s[1]['id'].startswith('phase_') and s[1]['id'] != 'phase_01_baseline'],
                           key=lambda x: x['avg_ms'], default=None)
        most_box = max(summary, key=lambda s: s[1]['total_box'])[1]
        if best_f:
            md.write('- On dinh nhat hien tai: `%s` (F1=%.3f).\n' % (best_f['id'], best_f['f']))
        else:
            md.write('- On dinh nhat hien tai: chua co GT de chon theo F1.\n')
        if fastest_sahi:
            md.write('- SAHI nhanh nhat: `%s` (%.1f ms/anh).\n' % (fastest_sahi['id'], fastest_sahi['avg_ms']))
        md.write('- Nhieu box nhat: `%s` (%d box) — coi chung FP, can GT de xac nhan.\n' % (most_box['id'], most_box['total_box']))
        md.write('\n## Ke hoach vong lap tiep theo\n\n')
        md.write('- Phase 06/07 da co trong vong nay (class-wise threshold + TTA flip). Xem bang tren.\n')
        md.write('- Phase 08+ (fine-tune, can du lieu): can bo PCB train 9920 anh + test 2508 anh + compiler Cython (nms.pyx). '
                 'Mo `FAST_RCNN_MINIBATCH_SIZE=-1` (OHEM 128) nhu ten VERSION, unfreeze block4, LR 1e-3 decay [10000,20000], '
                 'MAX_ITER 30000, anchor giu [15,25,40,60,80]. Chay `tools/train.py` roi danh gia lai bang vong nay.\n')
    print('=' * 72)
    print('XONG. Ket qua: %s/<phase_id>/REPORT.md + PHASES.md' % out_root)
    return 0


if __name__ == '__main__':
    sys.exit(main())
