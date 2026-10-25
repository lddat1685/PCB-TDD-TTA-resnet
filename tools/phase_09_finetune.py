# -*- coding: utf-8 -*-
"""Phase 09 — fine-tune TDD-Net tren tfrecord PCB that (file MOI, khong sua logic goc).

Ke thua nguyen pipeline train.py goc (DetectionNetwork + next_batch + losses),
chi override cfgs TRONG process nay (monkey-patch, khong sua libs/configs/cfgs.py):
  --ohem      : FAST_RCNN_MINIBATCH_SIZE=-1 -> nhanh OHEM sum_ohem_loss(128)
  --lr        : learning rate fine-tune (mac dinh 0.0003 < 0.001 goc)
  --max_steps : so step (CPU cham -> mac dinh 300 de smoke + lay ckpt moi)
  --save_inte : cu moi N step luu pcb_<step>model.ckpt canh pcb_30000 (khong xoa cu)

Restore: latest_checkpoint(output/trained_weights/FPN_Res101_0117_OHEM)
         = pcb_30000model (neu train lan dau). File `checkpoint` duoc backup
         thanh `checkpoint.before_phase09` de rollback 1 lenh.

Chay (tu thu muc repo, sau khi convert xong pcb_train.tfrecord that):
  python\\python.exe tools\\phase_09_finetune.py --max_steps 300 --save_inte 150
Xuat: output/trained_weights/FPN_Res101_0117_OHEM/pcb_<step>model.ckpt.*
      output/phases/phase_09_finetune/train_summary.json + train_log.jsonl
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(HERE, '.venv', 'Lib', 'site-packages')
import site as _site
_site.addsitedir(SITE)
sys.path.insert(0, HERE)

os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.chdir(os.path.join(HERE, 'tools'))

import tf1_compat  # noqa: E402,F401  phai truoc tensorflow
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
import tensorflow.contrib.slim as slim  # noqa: E402

# tf1_compat giu tf.summary (TF2) -> train can API v1: scalar/merge_all/FileWriter
_v1sum = tf.compat.v1.summary

from libs.configs import cfgs  # noqa: E402
from libs.networks import build_whole_network  # noqa: E402
from data.io import image_preprocess  # noqa: E402
from help_utils import tools  # noqa: E402

PID = 'phase_09_finetune'


def _parse_pcb_example(serialized):
    features = tf.io.parse_single_example(
        serialized=serialized,
        features={
            'img_name': tf.io.FixedLenFeature([], tf.string),
            'img_height': tf.io.FixedLenFeature([], tf.int64),
            'img_width': tf.io.FixedLenFeature([], tf.int64),
            'img': tf.io.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.io.FixedLenFeature([], tf.string),
            'num_objects': tf.io.FixedLenFeature([], tf.int64),
        })
    img_name = features['img_name']
    img_h = tf.cast(features['img_height'], tf.int32)
    img_w = tf.cast(features['img_width'], tf.int32)
    img = tf.io.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, shape=tf.stack([img_h, img_w, 3]))
    gt = tf.io.decode_raw(features['gtboxes_and_label'], tf.int32)
    gt = tf.reshape(gt, [-1, 5])
    num_objects = tf.cast(features['num_objects'], tf.int32)
    return img_name, img, gt, num_objects


def next_batch_tfdata(dataset_name, batch_size, shortside_len, is_training):
    """Doc tfrecord that bang tf.data (thay next_batch queue cu hong tren TF2 graph).

    Tien xu ly GIONG HET read_tfrecord.read_and_prepocess_single_img:
    short_side_resize + (train: random flip) + tru PIXEL_MEAN.
    """
    assert batch_size == 1, 'phase09 chi ho tro batch_size=1 (giong train.py goc)'
    if dataset_name not in ('pcb',):
        raise ValueError('phase09 chi ho tro dataset pcb, nhan: %s' % dataset_name)
    sub = 'train' if is_training else 'test'
    import glob as _glob
    pattern = os.path.abspath(os.path.join('../data/tfrecord',
                                           'pcb_%s*' % sub))
    files = sorted(_glob.glob(pattern))
    if not files:
        raise IOError('khong thay tfrecord: %s' % pattern)
    print('tfrecord files -->', files)

    ds = tf.data.TFRecordDataset(files)

    def _map(name, img, gt, num):
        img = tf.cast(img, tf.float32)
        if is_training:
            img, gt = image_preprocess.short_side_resize(
                img_tensor=img, gtboxes_and_label=gt,
                target_shortside_len=shortside_len,
                length_limitation=cfgs.IMG_MAX_LENGTH)
            img, gt = image_preprocess.random_flip_left_right(
                img_tensor=img, gtboxes_and_label=gt)
        else:
            img, gt = image_preprocess.short_side_resize(
                img_tensor=img, gtboxes_and_label=gt,
                target_shortside_len=shortside_len,
                length_limitation=cfgs.IMG_MAX_LENGTH)
        img = img - tf.constant([[cfgs.PIXEL_MEAN]])
        gt = tf.cast(gt, tf.int32)
        return name, img, gt, num

    ds = ds.map(lambda s: _map(*_parse_pcb_example(s)))
    ds = ds.repeat()
    ds = ds.padded_batch(batch_size,
                         padded_shapes=([], [None, None, 3], [None, 5], []))
    it = tf.compat.v1.data.make_one_shot_iterator(ds)
    n, im, gt_b, num_b = it.get_next()
    return n, im, gt_b, num_b


def train(args):
    if args.ohem:
        cfgs.FAST_RCNN_MINIBATCH_SIZE = -1
    cfgs.LR = args.lr
    cfgs.DECAY_STEP = [args.decay1, args.decay2]
    cfgs.MAX_ITERATION = args.max_steps
    cfgs.SAVE_WEIGHTS_INTE = args.save_inte
    cfgs.SHOW_TRAIN_INFO_INTE = args.show_inte

    print('=' * 72)
    print('PHASE 09 FINETUNE — override (khong sua cfgs.py):')
    print('  FAST_RCNN_MINIBATCH_SIZE =', cfgs.FAST_RCNN_MINIBATCH_SIZE,
          '(OHEM)' if cfgs.FAST_RCNN_MINIBATCH_SIZE == -1 else '(sample 256)')
    print('  LR =', cfgs.LR, '| MAX_ITERATION =', cfgs.MAX_ITERATION,
          '| SAVE_INTE =', cfgs.SAVE_WEIGHTS_INTE)
    print('  VERSION =', cfgs.VERSION, '| DATASET =', cfgs.DATASET_NAME)
    print('=' * 72)

    faster_rcnn = build_whole_network.DetectionNetwork(
        base_network_name=cfgs.NET_NAME, is_training=True)

    with tf.name_scope('get_batch'):
        img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch = \
            next_batch_tfdata(dataset_name=cfgs.DATASET_NAME,
                              batch_size=cfgs.BATCH_SIZE,
                              shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                              is_training=True)
        gtboxes_and_label = tf.reshape(gtboxes_and_label_batch, [-1, 5])

    biases_regularizer = getattr(slim, 'no_regularizer', lambda _: None)
    weights_regularizer = tf.contrib.layers.l2_regularizer(cfgs.WEIGHT_DECAY)
    with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                         slim.conv2d_transpose, slim.separable_conv2d,
                         slim.fully_connected],
                        weights_regularizer=weights_regularizer,
                        biases_regularizer=biases_regularizer,
                        biases_initializer=tf.constant_initializer(0.0)):
        final_bbox, final_scores, final_category, loss_dict = \
            faster_rcnn.build_whole_detection_network(
                input_img_batch=img_batch, gtboxes_batch=gtboxes_and_label)

    weight_decay_loss = tf.add_n(slim.losses.get_regularization_losses())
    rpn_location_loss = loss_dict['rpn_loc_loss']
    rpn_cls_loss = loss_dict['rpn_cls_loss']
    rpn_total_loss = rpn_location_loss + rpn_cls_loss
    fastrcnn_cls_loss = loss_dict['fastrcnn_cls_loss']
    fastrcnn_loc_loss = loss_dict['fastrcnn_loc_loss']
    fastrcnn_total_loss = fastrcnn_cls_loss + fastrcnn_loc_loss
    total_loss = rpn_total_loss + fastrcnn_total_loss + weight_decay_loss

    _v1sum.scalar('RPN_LOSS/cls_loss', rpn_cls_loss)
    _v1sum.scalar('RPN_LOSS/location_loss', rpn_location_loss)
    _v1sum.scalar('FAST_LOSS/fastrcnn_cls_loss', fastrcnn_cls_loss)
    _v1sum.scalar('FAST_LOSS/fastrcnn_location_loss', fastrcnn_loc_loss)
    _v1sum.scalar('LOSS/total_loss', total_loss)
    _v1sum.scalar('LOSS/regular_weights', weight_decay_loss)

    global_step = slim.get_or_create_global_step()
    lr = tf.train.piecewise_constant(
        global_step,
        boundaries=[np.int64(cfgs.DECAY_STEP[0]), np.int64(cfgs.DECAY_STEP[1])],
        values=[cfgs.LR, cfgs.LR / 10., cfgs.LR / 100.])
    _v1sum.scalar('lr', lr)
    optimizer = tf.train.MomentumOptimizer(lr, momentum=cfgs.MOMENTUM)
    gradients = faster_rcnn.get_gradients(optimizer, total_loss)
    if cfgs.MUTILPY_BIAS_GRADIENT:
        gradients = faster_rcnn.enlarge_gradients_for_bias(gradients)
    if cfgs.GRADIENT_CLIPPING_BY_NORM:
        with tf.name_scope('clip_gradients_YJR'):
            gradients = slim.learning.clip_gradient_norms(
                gradients, cfgs.GRADIENT_CLIPPING_BY_NORM)

    train_op = optimizer.apply_gradients(grads_and_vars=gradients,
                                         global_step=global_step)
    summary_op = _v1sum.merge_all()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    restorer, restore_ckpt = faster_rcnn.get_restorer()
    saver = tf.train.Saver(max_to_keep=30)

    ckpt_dir = os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION)
    ckpt_state = os.path.join(ckpt_dir, 'checkpoint')
    if os.path.isfile(ckpt_state):
        bak = os.path.join(ckpt_dir, 'checkpoint.before_phase09')
        if not os.path.isfile(bak):
            shutil.copy2(ckpt_state, bak)
            print('backup checkpoint state ->', bak)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    out_dir = os.path.join(HERE, 'output', 'phases', PID)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'train_log.jsonl')
    lf = open(log_path, 'a', encoding='utf-8')

    history = []
    t_start = time.time()
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if restorer is not None:
            restorer.restore(sess, restore_ckpt)
            print('restore model:', restore_ckpt)
        # tf.data one-shot iterator: khong can Coordinator/queue runners

        summary_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION)
        tools.mkdir(summary_path)
        summary_writer = _v1sum.FileWriter(summary_path, graph=sess.graph)

        try:
            for step in range(cfgs.MAX_ITERATION):
                if step % cfgs.SHOW_TRAIN_INFO_INTE != 0 and step % cfgs.SMRY_ITER != 0:
                    _, _ = sess.run([train_op, global_step])
                else:
                    if step % cfgs.SMRY_ITER == 0:
                        _, gnp, sstr = sess.run([train_op, global_step, summary_op])
                        summary_writer.add_summary(sstr, gnp)
                        summary_writer.flush()
                    else:
                        t0 = time.time()
                        (_, gnp, img_name, rpnLoc, rpnCls, rpnTot,
                         frLoc, frCls, frTot, tot) = sess.run(
                            [train_op, global_step, img_name_batch,
                             rpn_location_loss, rpn_cls_loss, rpn_total_loss,
                             fastrcnn_loc_loss, fastrcnn_cls_loss,
                             fastrcnn_total_loss, total_loss])
                        dt = time.time() - t0
                        rec = {'step': int(gnp),
                               'img': (img_name[0].decode('utf-8', 'ignore')
                                       if isinstance(img_name[0], bytes) else str(img_name[0])),
                               'rpn_loc': float(rpnLoc), 'rpn_cls': float(rpnCls),
                               'rcnn_loc': float(frLoc), 'rcnn_cls': float(frCls),
                               'total': float(tot), 'sec': round(dt, 2)}
                        history.append(rec)
                        lf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                        lf.flush()
                        print('step %d img=%s total=%.4f (rpn %.3f+%.3f rcnn %.3f+%.3f) %.1fs' %
                              (gnp, rec['img'], tot, rpnLoc, rpnCls, frLoc, frCls, dt))

                if (step > 0 and step % cfgs.SAVE_WEIGHTS_INTE == 0) or \
                        (step == cfgs.MAX_ITERATION - 1):
                    gnp_v = sess.run(global_step)
                    save_ckpt = os.path.join(ckpt_dir, 'pcb_%dmodel.ckpt' % int(gnp_v))
                    saver.save(sess, save_ckpt)
                    print('saved:', save_ckpt)
        finally:
            pass  # tf.data: khong co queue thread can dung
    lf.close()

    summ = {'phase': PID, 'ohem': bool(args.ohem), 'lr': args.lr,
            'max_steps': args.max_steps, 'restore_from': str(restore_ckpt),
            'steps_logged': len(history),
            'first_total': history[0]['total'] if history else None,
            'last_total': history[-1]['total'] if history else None,
            'minutes': round((time.time() - t_start) / 60.0, 1),
            'ckpt_dir': ckpt_dir}
    with open(os.path.join(out_dir, 'train_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summ, f, indent=2, ensure_ascii=False)
    print('=' * 72)
    print('XONG phase 09 train:', json.dumps(summ, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--save_inte', type=int, default=150)
    ap.add_argument('--show_inte', type=int, default=10)
    ap.add_argument('--ohem', type=int, default=1)
    ap.add_argument('--lr', type=float, default=0.0003)
    ap.add_argument('--decay1', type=int, default=100000,
                    help='moc decay 1 (de xa de LR giu nguyen khi fine-tune ngan)')
    ap.add_argument('--decay2', type=int, default=200000,
                    help='moc decay 2 (de xa de LR giu nguyen khi fine-tune ngan)')
    sys.exit(train(ap.parse_args()))
