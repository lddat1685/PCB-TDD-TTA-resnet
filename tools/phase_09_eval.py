# -*- coding: utf-8 -*-
"""Phase 09 eval — danh gia checkpoint fine-tune (file MOI, khong sua logic goc).

Dung Detector goc (tu dong nap latest = pcb_30300model sau train) chay:
  (a) baseline thr 0.6 tren 7 anh demo — so cong bang voi phase_01
  (b) fusion TTA+class-wise (params giong phase_08) — so voi phase_08
  (c) sample N anh tu pcb_test.tfrecord that (co GT day du) — P/R/F1 that

Chay (tu thu muc repo): python\\python.exe tools\\phase_09_eval.py [--n_test 100]
Xuat: output/phases/phase_09_finetuned/{metrics.json,metrics.csv,REPORT.md}
"""
from __future__ import absolute_import, division, print_function
import argparse, csv, glob as _glob, json, os, sys, time
import xml.etree.ElementTree as ET
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(HERE, '.venv', 'Lib', 'site-packages')
import site as _site
_site.addsitedir(SITE)
sys.path.insert(0, HERE)
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.chdir(os.path.join(HERE, 'tools'))

import tf1_compat  # noqa: E402,F401
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import tensorflow as tf  # noqa: E402

from ui.app import Detector  # noqa: E402
from libs.sahi.slicing import nms_per_class  # noqa: E402
from libs.label_name_dict.label_dict import NAME_LABEL_MAP  # noqa: E402

PID = 'phase_09_finetuned'
PNAME = 'Finetune OHEM 300 step tren PCB that (pcb_30300)'
CTHR = {'missing_hole': 0.6, 'short': 0.6, 'mouse_bite': 0.6,
        'spur': 0.65, 'open_circuit': 0.8, 'spurious_copper': 0.8}


def map_to_original(res, W, H):
    rw, rh = res['size']; sx, sy = W / float(rw), H / float(rh)
    return [{'box': [d['box'][0]*sx, d['box'][1]*sy, d['box'][2]*sx, d['box'][3]*sy],
             'score': d['score'], 'label': d['label'], 'name': d.get('name',''),
             'color': d.get('color','#888888')} for d in res['detections']]


def parse_voc_xml(p):
    t = ET.parse(p); r = t.getroot(); s = r.find('size')
    w, h = int(s.find('width').text), int(s.find('height').text)
    objs = []
    for o in r.findall('object'):
        bb = o.find('bndbox')
        objs.append({'name': o.find('name').text,
                     'label': int(NAME_LABEL_MAP.get(o.find('name').text, -1)),
                     'box': [float(bb.find('xmin').text), float(bb.find('ymin').text),
                             float(bb.find('xmax').text), float(bb.find('ymax').text)]})
    return objs, w, h


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    ab = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def match_dets(dets, gts, thr=0.5):
    gm = [False]*len(gts); tp = fp = 0
    for d in sorted(dets, key=lambda x: -x['score']):
        best, bi = -1, 0.0
        for gi, g in enumerate(gts):
            if gm[gi] or int(g['label']) != int(d['label']):
                continue
            v = iou(d['box'], g['box'])
            if v > bi:
                bi, best = v, gi
        if best >= 0 and bi >= thr:
            gm[best] = True; tp += 1
        else:
            fp += 1
    return tp, fp, gm.count(False)


def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return p, r, 2*p*r/(p+r) if p+r else 0.0


def run_fusion(det, rgb, W, H):
    r1 = det.run(rgb, 0.6); d1 = map_to_original(r1, W, H)
    rf = np.ascontiguousarray(rgb[:, ::-1, :])
    r2 = det.run(rf, 0.6); d2 = map_to_original(r2, W, H)
    d2b = [{'box': [W-d['box'][2], d['box'][1], W-d['box'][0], d['box'][3]],
            'score': d['score'], 'label': d['label'],
            'name': d.get('name',''), 'color': d.get('color','#888888')} for d in d2]
    merged = [d for d in d1+d2b if d['score'] >= float(CTHR.get(d.get('name'), 0.6))]
    dets = nms_per_class(merged, iou_threshold=0.5)
    dets.sort(key=lambda d: -d['score'])
    return dets, r1['elapsed_ms']+r2['elapsed_ms'], 2, len(d1)+len(d2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_test', type=int, default=100)
    ap.add_argument('--iou_thr', type=float, default=0.5)
    args = ap.parse_args()
    print('Dang nap model (latest checkpoint, ~40s)...')
    det = Detector()
    print('Model:', det.ckpt_path)

    files = []
    for d in ['../input', '../tools/demos', '..']:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff')):
                    files.append(os.path.join(d, f))
    seen, uniq = {}, []
    for p in files:
        b = os.path.basename(p)
        if b not in seen:
            seen[b] = p; uniq.append(p)

    rows = []; T = {'tp':0,'fp':0,'fn':0,'gt':0,'box':0,'ms':0,'tiles':0,'raw':0}
    for fp in uniq:
        fn = os.path.basename(fp)
        rgb = np.array(Image.open(fp).convert('RGB')); H, W = rgb.shape[:2]
        # (a) baseline
        t0 = time.time()
        r = det.run(rgb, 0.6); dets = map_to_original(r, W, H)
        ms_a = r['elapsed_ms']
        # (b) fusion
        dets_f, ms_f, tiles_f, raw_f = run_fusion(det, rgb, W, H)
        xp = os.path.join('test_annotation', os.path.splitext(fn)[0]+'.xml')
        has_gt = os.path.isfile(xp)
        if has_gt:
            gt_raw, gw, gh = parse_voc_xml(xp)
            sx, sy = W/float(gw), H/float(gh)
            gt = [{'name':g['name'],'label':g['label'],
                   'box':[g['box'][0]*sx,g['box'][1]*sy,g['box'][2]*sx,g['box'][3]*sy]} for g in gt_raw]
            tp,fp_,fn_ = match_dets(dets_f, gt, args.iou_thr)
            p,rr,f = prf(tp,fp_,fn_)
            T['tp']+=tp; T['fp']+=fp_; T['fn']+=fn_; T['gt']+=len(gt)
            extra = dict(tp=tp,fp=fp_,fn=fn_,p=p,r=rr,f=f,has_gt=True)
        else:
            extra = dict(tp=None,fp=None,fn=None,p=None,r=None,f=None,has_gt=False)
        T['box']+=len(dets_f); T['ms']+=ms_f; T['tiles']+=tiles_f; T['raw']+=raw_f
        cnt = dict(Counter(d['name'] for d in dets_f))
        mean_s = round(sum(d['score'] for d in dets_f)/len(dets_f),4) if dets_f else 0.0
        rows.append(dict(image=fn,size='%dx%d'%(W,H),gt=len(gt) if has_gt else '-',
                         n=len(dets_f),ms=ms_f,tiles=tiles_f,raw=raw_f,mean=mean_s,
                         counts=cnt,n_base=len(dets),ms_base=ms_a,**extra))
        print('  %-22s %s base=%d fusion=%d %dms %s %s' % (
            fn,'%dx%d'%(W,H),len(dets),len(dets_f),ms_f,dict(cnt),
            ('P=%.3f R=%.3f F1=%.3f'%(p,rr,f)) if has_gt else ''))

    mp,mr,mf = prf(T['tp'],T['fp'],T['fn']) if T['gt'] else (None,None,None)
    rec = dict(id=PID,name=PNAME,
               params=dict(ckpt=os.path.basename(det.ckpt_path or ''),
                           train='OHEM 300 step lr=0.0003 tu pcb_30000',
                           fusion=dict(base_thr=0.6,nms_iou=0.5,tta_flip=True,
                                       class_thresholds=CTHR)),
               delta='Cung pipeline fusion nhu phase_08 nhung checkpoint la pcb_30300 (fine-tune that)',
               n_img=len(rows),total_box=T['box'],
               avg_ms=round(T['ms']/max(1,len(rows)),1),
               avg_tiles=round(T['tiles']/max(1,len(rows)),2),
               gt_total=T['gt'],tp=T['tp'],fp=T['fp'],fn=T['fn'],p=mp,r=mr,f=mf)
    pdir = os.path.join('../output/phases', PID); os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir,'metrics.csv'),'w',newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['image','size','gt','n','ms','tiles','raw','mean','counts','n_base','TP','FP','FN','P','R','F1'])
        for r in rows:
            w.writerow([r['image'],r['size'],r['gt'],r['n'],r['ms'],r['tiles'],r['raw'],
                        r['mean'],json.dumps(r['counts'],ensure_ascii=False),r['n_base'],
                        r['tp'],r['fp'],r['fn'],r['p'],r['r'],r['f']])
    with open(os.path.join(pdir,'metrics.json'),'w') as jf:
        json.dump({'phase':rec,'rows':rows},jf,indent=2,ensure_ascii=False)
    with open(os.path.join(pdir,'REPORT.md'),'w') as md:
        md.write('# %s — %s\n\n' % (PID, PNAME))
        md.write('Checkpoint: `%s`\n\n' % det.ckpt_path)
        md.write('So voi phase_08 (fusion, ckpt cu): box %d, P/R/F1 = %s/%s/%s\n\n' % (
            rec['total_box'],
            ('%.3f'%mp) if mp is not None else '-', ('%.3f'%mr) if mr is not None else '-',
            ('%.3f'%mf) if mf is not None else '-'))
        md.write('| anh | size | GT | n_base | n_fusion | ms | mean | counts | P | R | F1 |\n')
        md.write('|---|---|---|---|---|---|---|---|---|---|---|\n')
        for r in rows:
            if r['has_gt']:
                md.write('| %s | %s | %s | %d | %d | %d | %.3f | %s | %.3f | %.3f | %.3f |\n' % (
                    r['image'],r['size'],r['gt'],r['n_base'],r['n'],r['ms'],r['mean'],
                    json.dumps(r['counts'],ensure_ascii=False),r['p'],r['r'],r['f']))
            else:
                md.write('| %s | %s | - | %d | %d | %d | %.3f | %s | - | - | - |\n' % (
                    r['image'],r['size'],r['n_base'],r['n'],r['ms'],r['mean'],
                    json.dumps(r['counts'],ensure_ascii=False)))
    print('='*72)
    print('XONG eval phase 09 (7 anh demo): box=%d %s' % (
        T['box'], ('P=%.3f R=%.3f F1=%.3f'%(mp,mr,mf)) if T['gt'] else ''))
    print('Tiep: danh gia tren %d anh test that...' % args.n_test)

    # ---- (c) test that tu tfrecord ----
    import glob as _g
    tfiles = sorted(_g.glob(os.path.abspath('../data/tfrecord/pcb_test.tfrecord')))
    ds = tf.data.TFRecordDataset(tfiles)
    def _parse(s):
        f = tf.io.parse_single_example(s, features={
            'img_name': tf.io.FixedLenFeature([], tf.string),
            'img_height': tf.io.FixedLenFeature([], tf.int64),
            'img_width': tf.io.FixedLenFeature([], tf.int64),
            'img': tf.io.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.io.FixedLenFeature([], tf.string),
            'num_objects': tf.io.FixedLenFeature([], tf.int64)})
        im = tf.io.decode_raw(f['img'], tf.uint8)
        h = tf.cast(f['img_height'], tf.int32); w = tf.cast(f['img_width'], tf.int32)
        im = tf.reshape(im, tf.stack([h, w, 3]))
        gt = tf.reshape(tf.io.decode_raw(f['gtboxes_and_label'], tf.int32), [-1, 5])
        return f['img_name'], im, gt
    ds = ds.map(_parse).take(args.n_test)
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    nxt = it.get_next()
    gtp=gtt=0; gfp=gfn=0; gn=0; gms=0
    with tf.Session() as sess:
        i = 0
        while True:
            try:
                name_b, im_np, gt_np = sess.run(nxt)
            except tf.errors.OutOfRangeError:
                break
            i += 1
            rgb = np.array(im_np, dtype=np.uint8); Hh, Ww = rgb.shape[:2]
            dets_f, ms_f, _, _ = run_fusion(det, rgb, Ww, Hh)
            gt = [{'label': int(row[4]),
                   'box': [float(row[0]),float(row[1]),float(row[2]),float(row[3])]}
                  for row in gt_np]
            tp,fp_,fn_ = match_dets(dets_f, gt, args.iou_thr)
            gtp+=tp; gfp+=fp_; gfn+=fn_; gtt+=len(gt); gn+=len(dets_f); gms+=ms_f
            if i % 20 == 0:
                p0,r0,f0 = prf(gtp,gfp,gfn)
                print('  test %d/%d: box=%d GT=%d TP=%d FP=%d FN=%d P=%.3f R=%.3f F1=%.3f' % (
                    i,args.n_test,gn,gtt,gtp,gfp,gfn,p0,r0,f0))
    gp,gr,gf = prf(gtp,gfp,gfn)
    print('='*72)
    print('TEST THAT %d anh: box=%d GT=%d TP=%d FP=%d FN=%d -> P=%.3f R=%.3f F1=%.3f, %.1f ms/anh' % (
        args.n_test,gn,gtt,gtp,gfp,gfn,gp,gr,gf,gms/max(1,i)))
    res = dict(n_test=i,boxes=gn,gt=gtt,tp=gtp,fp=gfp,fn=gfn,p=gp,r=gr,f=gf,
               ms_per_img=round(gms/max(1,i),1))
    with open(os.path.join(pdir,'testset_eval.json'),'w') as jf:
        json.dump(res,jf,indent=2,ensure_ascii=False)
    # append vao usage_log + regen PROGRESS
    import datetime as _dt
    lp = os.path.join('../output/phases','usage_log.csv')
    with open(lp,'a',newline='') as lf:
        w = csv.writer(lf)
        w.writerow([_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),PID,PNAME,len(rows),
                    T['tiles'],T['raw'],T['ms'],round(T['ms']/max(1,len(rows)),1),T['box'],
                    round(T['ms']/max(1,T['box']),1),T['gt'],T['tp'],T['fp'],T['fn'],mp,mr,mf,
                    json.dumps(rec['params'],ensure_ascii=False)])
    with open(os.path.join('../output/phases',PID,'PROGRESS.md') if False else os.path.join(pdir,'PROGRESS.md'),'w',encoding='utf-8') as f:
        f.write('# Phase 09 — hoan thanh\n\nTrain: 300/300 step (100%%), ckpt pcb_30300.\n')
        f.write('Demo 7 anh (fusion): box=%d P/R/F1=%s/%s/%s.\n' % (
            T['box'],('%.3f'%mp) if mp is not None else '-',
            ('%.3f'%mr) if mr is not None else '-',('%.3f'%mf) if mf is not None else '-'))
        f.write('Test that %d anh: box=%d GT=%d P/R/F1=%.3f/%.3f/%.3f, %.1f ms/anh.\n' % (
            i,gn,gtt,gp,gr,gf,gms/max(1,i)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
