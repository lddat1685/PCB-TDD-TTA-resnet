# -*- coding: utf-8 -*-
"""Convert bo PCB YOLO ngoai -> TFRecord chuan TDD-Net (file MOI, khong sua logic goc).

Bo ngoai: D:/FPTU/Ki9_FPT_FA26/DO_AN/data/pcb-defect-dataset (YOLO, 600x600 RGB)
  data.yaml names: 0:mouse_bite 1:spur 2:missing_hole 3:short 4:open_circuit 5:spurious_copper
  train 8534 / val 1066 / test 1068 (images + labels)
  Quy luat mapping da kiem chung:
    - label *_600.txt -> anh cung ten *.jpg (khop truc tiep)
    - label *_256.txt -> anh *_600.jpg (doi hau to, khop 100%, vi toa do YOLO
      da chuan hoa 0-1 nen dung duoc cho anh 600x600)
    - tien to l_ : khop chinh xac ca 2 dau, khong phai cat bo

TDD NAME_LABEL_MAP (libs/label_name_dict/label_dict.py, DATASET_NAME='pcb'):
  missing_hole:1 mouse_bite:2 open_circuit:3 short:4 spur:5 spurious_copper:6
  => yolo_id -> tdd_id = [2,5,1,4,3,6]

Dinh dang TFRecord giong convert_data_to_tfrecord.py goc:
  img_name(bytes) img_height/width(int64) img(bytes raw uint8 RGB)
  gtboxes_and_label(bytes raw int32 [N,5] x1,y1,x2,y2,label) num_objects(int64)

Chay (tu thu muc repo):
  python\\python.exe tools\\convert_yolo_to_tfrecord.py [--splits train test] [--max N]
Xuat: data/tfrecord/pcb_train.tfrecord + pcb_test.tfrecord (+ stats json)
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob as _glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(HERE, '.venv', 'Lib', 'site-packages')
import site as _site
_site.addsitedir(SITE)
sys.path.insert(0, HERE)

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import tf1_compat  # noqa: E402,F401  phai truoc tensorflow
import tensorflow as tf  # noqa: E402

YOLO_NAMES = ['mouse_bite', 'spur', 'missing_hole', 'short', 'open_circuit',
              'spurious_copper']
TDD_IDS = {'missing_hole': 1, 'mouse_bite': 2, 'open_circuit': 3,
           'short': 4, 'spur': 5, 'spurious_copper': 6}
YOLO2TDD = [TDD_IDS[n] for n in YOLO_NAMES]  # [2,5,1,4,3,6]

def _autodetect_root():
    # pattern ASCII-only de tranh van de encode dau tieng Viet tren console
    cands = _glob.glob('D:/FPTU/Ki9_FPT_FA26/*/data/pcb-defect-dataset')
    cands = [c for c in cands if os.path.isdir(c)]
    if cands:
        return cands[0]
    return 'D:/FPTU/Ki9_FPT_FA26/DO_AN/data/pcb-defect-dataset'


DEFAULT_ROOT = _autodetect_root()


def _imread_unicode(path):
    # cv2.imread that bai voi duong dan co dau tieng Viet -> doc raw + imdecode
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size == 0:
        return None
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img[:, :, ::-1]  # BGR->RGB


def _int64(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v)]))


def _bytes(v):
    if isinstance(v, str):
        v = v.encode('utf-8')
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))


def resolve_image(imdir, imageset, label_base):
    """Tra ve ten file anh cho 1 label base. Nhan (img_name|None, cach)."""
    if label_base in imageset:
        return label_base + '.jpg', 'exact'
    if label_base.endswith('_256'):
        cand = label_base.rsplit('_', 1)[0] + '_600'
        if cand in imageset:
            return cand + '.jpg', '256->600'
    return None, 'unmatched'


def yolo_to_abs(lines, W, H):
    boxes = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        p = ln.split()
        cid = int(float(p[0]))
        cx, cy, w, h = float(p[1]), float(p[2]), float(p[3]), float(p[4])
        x1 = max(0, int(round((cx - w / 2.0) * W)))
        y1 = max(0, int(round((cy - h / 2.0) * H)))
        x2 = min(W, int(round((cx + w / 2.0) * W)))
        y2 = min(H, int(round((cy + h / 2.0) * H)))
        if x2 > x1 and y2 > y1:
            boxes.append([x1, y1, x2, y2, YOLO2TDD[cid]])
    return np.array(boxes, dtype=np.int32).reshape(-1, 5)


def convert_split(droot, split, out_path, max_n=None):
    imdir = os.path.join(droot, split, 'images')
    lbdir = os.path.join(droot, split, 'labels')
    lab_files = sorted(f for f in os.listdir(lbdir) if f.endswith('.txt'))
    imageset = set(os.path.splitext(f)[0] for f in os.listdir(imdir)
                   if f.lower().endswith('.jpg'))
    if max_n:
        lab_files = lab_files[:max_n]

    # backup placeholder cu (<1KB la file text gia) thay vi ghi de mat dau vet
    if os.path.isfile(out_path) and os.path.getsize(out_path) < 1024:
        bak = out_path + '.orig_placeholder_bak'
        if not os.path.isfile(bak):
            shutil.copy2(out_path, bak)
            print('backup placeholder: %s (%d B)' % (bak, os.path.getsize(bak)))

    writer = tf.io.TFRecordWriter(out_path)
    stats = {'split': split, 'n_label': len(lab_files), 'n_ok': 0,
             'n_unmatched': 0, 'n_bad_img': 0, 'via_exact': 0, 'via_256': 0,
             'boxes': 0, 'skipped_box': 0, 'per_class': {n: 0 for n in YOLO_NAMES}}
    for i, lf in enumerate(lab_files):
        base = os.path.splitext(lf)[0]
        img_name, how = resolve_image(imdir, imageset, base)
        if img_name is None:
            stats['n_unmatched'] += 1
            continue
        img = _imread_unicode(os.path.join(imdir, img_name))
        if img is None:
            stats['n_bad_img'] += 1
            continue
        H, W = img.shape[0], img.shape[1]
        with open(os.path.join(lbdir, lf), encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        gt = yolo_to_abs(lines, W, H)
        stats['skipped_box'] += len([l for l in lines if l.strip()]) - len(gt)
        for row in gt:
            stats['per_class'][YOLO_NAMES[YOLO2TDD.index(int(row[4]))]] += 1
        feat = tf.train.Features(feature={
            'img_name': _bytes(img_name),
            'img_height': _int64(H),
            'img_width': _int64(W),
            'img': _bytes(img.tobytes()),
            'gtboxes_and_label': _bytes(gt.tobytes()),
            'num_objects': _int64(len(gt)),
        })
        writer.write(tf.train.Example(features=feat).SerializeToString())
        stats['n_ok'] += 1
        stats['boxes'] += len(gt)
        stats['via_exact' if how == 'exact' else 'via_256'] += 1
        if (i + 1) % 1000 == 0:
            print('  %s: %d/%d ok=%d boxes=%d' % (split, i + 1, len(lab_files),
                                                  stats['n_ok'], stats['boxes']))
    writer.close()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--splits', nargs='+', default=['train', 'test'])
    ap.add_argument('--out_dir', default=os.path.join(HERE, 'data', 'tfrecord'))
    ap.add_argument('--max', type=int, default=None, help='gioi han so label de chay thu')
    args = ap.parse_args()

    droot = args.root
    if not os.path.isdir(droot):
        # thu duong dan co dau tieng Viet (crappy console encode truoc do)
        alt = 'D:/FPTU/Ki9_FPT_FA26/\u0110\u1ed2_AN/data/pcb-defect-dataset'
        if os.path.isdir(alt):
            droot = alt
    print('dataset root:', droot)
    print('yaml names:', YOLO_NAMES, '-> TDD ids:', YOLO2TDD)

    os.makedirs(args.out_dir, exist_ok=True)
    all_stats = {}
    for split in args.splits:
        outp = os.path.join(args.out_dir, 'pcb_%s.tfrecord' % split)
        print('convert %s -> %s' % (split, outp))
        st = convert_split(droot, split, outp, args.max)
        all_stats[split] = st
        print('  xong: ok=%d/%d unmatched=%d bad_img=%d boxes=%d (exact=%d 256->600=%d)' %
              (st['n_ok'], st['n_label'], st['n_unmatched'], st['n_bad_img'],
               st['boxes'], st['via_exact'], st['via_256']))
        print('  per_class:', st['per_class'])
        print('  file size: %.1f MB' % (os.path.getsize(outp) / 1048576.0))

    sp = os.path.join(args.out_dir, 'convert_stats.json')
    with open(sp, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print('stats ->', sp)
    return 0


if __name__ == '__main__':
    sys.exit(main())
