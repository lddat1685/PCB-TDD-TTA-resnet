# -*- coding: utf-8 -*-
"""Local web UI for TDD-Net (PCB defect detection).

Run:  .venv/Scripts/python.exe ui/app.py
Then open http://127.0.0.1:5000

The detection graph is built once at startup; the score threshold is a graph
tensor so moving the slider does not require rebuilding the network.
"""

from __future__ import absolute_import, division, print_function

import base64
import csv
import io
import json
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# libs/configs/cfgs.py computes ROOT_PATH as abspath('../'), i.e. relative to the
# process CWD, and expects to be run from <repo>/tools. Pin it here so the UI can
# be launched from anywhere.
os.chdir(os.path.join(ROOT, 'tools'))

import tf1_compat  # noqa: E402,F401  must precede tensorflow imports

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from PIL import Image  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402

from data.io.image_preprocess import short_side_resize_for_inference_data  # noqa: E402
from libs.configs import cfgs  # noqa: E402
from libs.label_name_dict.label_dict import LABEl_NAME_MAP  # noqa: E402
from libs.networks import build_whole_network  # noqa: E402

app = Flask(__name__, template_folder=os.path.join(ROOT, 'ui', 'templates'))
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024

CLASS_COLORS = {
    1: '#ff4d4f',  # missing_hole
    2: '#faad14',  # mouse_bite
    3: '#52c41a',  # open_circuit
    4: '#13c2c2',  # short
    5: '#9254de',  # spur
    6: '#1890ff',  # spurious_copper
}


class Detector(object):
    """Holds the TF graph + session; one instance for the process."""

    def __init__(self):
        self.lock = threading.Lock()

        self.img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
        self.thresh_plac = tf.placeholder(dtype=tf.float32, shape=[])

        img_f = tf.cast(self.img_plac, tf.float32)
        resized = short_side_resize_for_inference_data(
            img_tensor=img_f,
            target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
            length_limitation=cfgs.IMG_MAX_LENGTH)
        self.resized_u8 = tf.cast(tf.clip_by_value(resized, 0.0, 255.0), tf.uint8)
        net_in = tf.expand_dims(resized - tf.constant(cfgs.PIXEL_MEAN), axis=0)

        det_net = build_whole_network.DetectionNetwork(
            base_network_name=cfgs.NET_NAME, is_training=False)
        boxes, scores, categories = det_net.build_whole_detection_network(
            input_img_batch=net_in, gtboxes_batch=None)

        keep = scores >= self.thresh_plac
        self.boxes = tf.boolean_mask(boxes, keep)
        self.scores = tf.boolean_mask(scores, keep)
        self.categories = tf.boolean_mask(categories, keep)

        self.init_op = tf.group(tf.global_variables_initializer(),
                                tf.local_variables_initializer())
        self.restorer, self.ckpt_path = det_net.get_restorer()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.intra_op_parallelism_threads = 0

        self.sess = tf.Session(config=config)
        self.sess.run(self.init_op)

        self.ckpt_loaded = False
        if self.restorer is not None:
            self.restorer.restore(self.sess, self.ckpt_path)
            self.ckpt_loaded = True

        self.num_vars = len(tf.global_variables())
        self.classes = [
            {'id': int(i), 'name': LABEl_NAME_MAP[i], 'color': CLASS_COLORS.get(i, '#888888')}
            for i in sorted(LABEl_NAME_MAP) if i != 0
        ]

    def gpu_available(self):
        return len(tf.config.list_physical_devices('GPU')) > 0

    def run(self, rgb_array, threshold):
        with self.lock:
            t0 = time.time()
            (resized, boxes, scores, cats) = self.sess.run(
                [self.resized_u8, self.boxes, self.scores, self.categories],
                feed_dict={self.img_plac: rgb_array,
                           self.thresh_plac: float(threshold)})
            elapsed = time.time() - t0

        dets = []
        for b, s, c in zip(boxes, scores, cats):
            c = int(c)
            dets.append({
                'box': [float(round(v, 1)) for v in b],
                'score': float(round(float(s), 4)),
                'label': c,
                'name': LABEl_NAME_MAP.get(c, 'class_%d' % c),
                'color': CLASS_COLORS.get(c, '#888888'),
            })
        dets.sort(key=lambda d: -d['score'])

        summary = {}
        for d in dets:
            summary[d['name']] = summary.get(d['name'], 0) + 1

        return {
            'detections': dets,
            'summary': summary,
            'elapsed_ms': int(elapsed * 1000),
            'image': _png_b64(resized),
            'size': [int(resized.shape[1]), int(resized.shape[0])],
        }


def _png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


DETECTOR = None
STARTUP_ERR = None


def get_detector():
    global DETECTOR, STARTUP_ERR
    if DETECTOR is None and STARTUP_ERR is None:
        try:
            DETECTOR = Detector()
        except Exception as exc:  # surface the real reason in /api/health
            STARTUP_ERR = '%s: %s' % (type(exc).__name__, exc)
            raise
    if STARTUP_ERR:
        raise RuntimeError(STARTUP_ERR)
    return DETECTOR


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/health')
def health():
    try:
        det = get_detector()
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 500
    return jsonify({
        'ok': True,
        'checkpoint': os.path.basename(det.ckpt_path or ''),
        'checkpoint_loaded': det.ckpt_loaded,
        'variables': det.num_vars,
        'device': 'GPU' if det.gpu_available() else 'CPU',
        'classes': det.classes,
        'img_short_side': cfgs.IMG_SHORT_SIDE_LEN,
        'max_length': cfgs.IMG_MAX_LENGTH,
        'default_threshold': cfgs.SHOW_SCORE_THRSHOLD,
    })


@app.route('/api/model')
def model_info():
    """Thong tin model dang su dung (khong nap graph, chi doc cfgs + checkpoint)."""
    ckpt_dir = os.path.join(ROOT, 'output', 'trained_weights', cfgs.VERSION)
    files, total = [], 0
    if os.path.isdir(ckpt_dir):
        for f in sorted(os.listdir(ckpt_dir)):
            sz = os.path.getsize(os.path.join(ckpt_dir, f))
            total += sz
            files.append({'name': f, 'mb': round(sz / 1048576.0, 1)})
    import tensorflow as _tf
    return jsonify({
        'ok': True,
        'version': cfgs.VERSION,
        'net': cfgs.NET_NAME,
        'backbone': 'ResNet-101 + FPN P2-P6 + RPN share-head + FastRCNN FC1024x2',
        'dataset': cfgs.DATASET_NAME,
        'class_num': cfgs.CLASS_NUM,
        'classes': {int(i): LABEl_NAME_MAP[i] for i in sorted(LABEl_NAME_MAP)},
        'img_short_side': cfgs.IMG_SHORT_SIDE_LEN,
        'max_length': cfgs.IMG_MAX_LENGTH,
        'pixel_mean': list(cfgs.PIXEL_MEAN),
        'anchors': {'base': list(cfgs.BASE_ANCHOR_SIZE_LIST),
                    'scales': list(cfgs.ANCHOR_SCALES),
                    'ratios': list(cfgs.ANCHOR_RATIOS),
                    'levels': list(cfgs.LEVLES),
                    'strides': list(cfgs.ANCHOR_STRIDE_LIST)},
        'rpn': {'nms': cfgs.RPN_NMS_IOU_THRESHOLD,
                'topk': cfgs.RPN_TOP_K_NMS_TEST,
                'max_proposal': cfgs.RPN_MAXIMUM_PROPOSAL_TEST},
        'fastrcnn': {'nms': cfgs.FAST_RCNN_NMS_IOU_THRESHOLD,
                     'roi': cfgs.ROI_SIZE, 'keep_prob': cfgs.KEEP_PROB},
        'default_threshold': cfgs.SHOW_SCORE_THRSHOLD,
        'checkpoint_dir': ckpt_dir,
        'checkpoint_files': files,
        'checkpoint_mb': round(total / 1048576.0, 1),
        'device_note': 'CPU (GTX1650 thieu cuDNN) am GPU=%s' % len(_tf.config.list_physical_devices('GPU')),
        'tf_version': _tf.__version__,
        'is_llm': False,
        'cost_unit': '1 forward pass Detector = 1 don vi (tuong duong token)',
    })


def _load_tracking():
    """Doc output/phases/*/metrics.json + usage_log.csv (file MOI, khong cham graph)."""
    out = os.path.join(ROOT, 'output', 'phases')
    phases = []
    if os.path.isdir(out):
        for pid in sorted(os.listdir(out)):
            mp = os.path.join(out, pid, 'metrics.json')
            if os.path.isfile(mp):
                try:
                    with open(mp) as f:
                        d = json.load(f)
                    rec, rows = d['phase'], d['rows']
                    phases.append({
                        'id': rec['id'], 'name': rec['name'],
                        'params': rec['params'],
                        'n_img': rec['n_img'],
                        'forwards': sum(r['tiles'] for r in rows),
                        'ms_total': sum(r['ms'] for r in rows),
                        'ms_per_img': rec['avg_ms'],
                        'boxes': rec['total_box'],
                        'ms_per_box': round(sum(r['ms'] for r in rows) / max(1, rec['total_box']), 1),
                        'gt': rec.get('gt_total') or 0,
                        'tp': rec.get('tp') or 0, 'fp': rec.get('fp') or 0,
                        'fn': rec.get('fn') or 0,
                        'p': rec.get('p'), 'r': rec.get('r'), 'f': rec.get('f'),
                    })
                except Exception:
                    pass
    usage = []
    lp = os.path.join(out, 'usage_log.csv')
    if os.path.isfile(lp):
        with open(lp) as f:
            for r in csv.DictReader(f):
                usage.append(r)
    return phases, usage[-30:]


@app.route('/api/tracking')
def tracking():
    phases, usage = _load_tracking()
    best = None
    for p in phases:
        if p['f'] is not None and (best is None or p['f'] > best['f']):
            best = p
    return jsonify({'ok': True, 'phases': phases, 'usage': usage,
                    'best_id': best['id'] if best else None,
                    'note': 'TDD-Net khong phai LLM: 1 forward = 1 don vi chi phi.'})


@app.route('/api/detect', methods=['POST'])
def detect():
    det = get_detector()

    f = request.files.get('image')
    if f is None:
        return jsonify({'error': 'no image uploaded'}), 400

    try:
        threshold = float(request.form.get('threshold', cfgs.SHOW_SCORE_THRSHOLD))
    except ValueError:
        return jsonify({'error': 'bad threshold'}), 400
    threshold = min(max(threshold, 0.0), 1.0)

    try:
        img = Image.open(f.stream).convert('RGB')
    except Exception as exc:
        return jsonify({'error': 'cannot read image: %s' % exc}), 400

    result = det.run(np.array(img), threshold)
    result['filename'] = f.filename
    result['original_size'] = [img.width, img.height]
    return jsonify(result)


if __name__ == '__main__':
    print('=' * 70)
    print('TDD-Net PCB defect detection UI')
    print('  root       :', ROOT)
    print('  checkpoint :', cfgs.TRAINED_CKPT, '/', cfgs.VERSION)
    print('=' * 70)
    get_detector()
    print('Model ready. Open http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
