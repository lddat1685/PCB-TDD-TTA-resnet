# -*- coding: utf-8 -*-
"""SAHI inference cho TDD-Net: cat slice -> chay Detector goc -> gop NMS.

Khong sua logic baseline: tai su dung Detector tu ui/app.py (cung graph,
cung checkpoint, cung tien xu ly). Chi them buoc cat anh + map toa do + NMS.
"""

from __future__ import absolute_import, division, print_function

import time

from libs.sahi.slicing import (
    get_slices,
    merge_sahi_detections,
)


def sahi_detect(detector, rgb_array,
                threshold=0.6,
                slice_h=512, slice_w=512,
                overlap=0.2,
                nms_iou=0.5,
                include_full_image=True):
    """Chay SAHI tren 1 anh RGB (numpy HxWx3).

    Tra ve dict:
      detections: list {box (he toa do ANH GOC), score, label, name, color}
      num_slices, slice_ms, total_ms, full_ms
    Box tra ve o he toa do anh GOC (khac voi Detector.run tra ve he resized).
    De so sanh cong bang, benchmark se map ca baseline ve he goc.
    """
    H, W = rgb_array.shape[0], rgb_array.shape[1]
    slices = get_slices(H, W, slice_h, slice_w,
                        overlap_h=overlap, overlap_w=overlap)

    # Neu anh nho hon slice -> chi 1 slice = ca anh, khoi chay full rieng
    single_cover = (len(slices) == 1 and slices[0] == (0, 0, H, W))
    use_full = include_full_image and not single_cover

    all_raw = []   # detections da map ve he goc
    slice_ms = 0
    full_ms = 0
    t0 = time.time()

    def _run_crop(crop, x_off, y_off):
        res = detector.run(crop, threshold)
        rw, rh = res['size']          # kich thuoc sau resize trong Detector
        sh, sw = crop.shape[0], crop.shape[1]
        sx = float(sw) / float(rw) if rw else 1.0
        sy = float(sh) / float(rh) if rh else 1.0
        out = []
        for d in res['detections']:
            x1, y1, x2, y2 = d['box']
            out.append({
                'box': [float(x1 * sx + x_off), float(y1 * sy + y_off),
                        float(x2 * sx + x_off), float(y2 * sy + y_off)],
                'score': float(d['score']),
                'label': int(d['label']),
                'name': d.get('name', 'class_%d' % int(d['label'])),
                'color': d.get('color', '#888888'),
            })
        return out, res['elapsed_ms']

    if use_full:
        full_dets, ms = _run_crop(rgb_array, 0, 0)
        all_raw.extend(full_dets)
        full_ms = ms
        slice_ms += ms

    for (y_min, x_min, y_max, x_max) in slices:
        crop = rgb_array[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue
        dets, ms = _run_crop(crop, x_min, y_min)
        all_raw.extend(dets)
        slice_ms += ms

    merged = merge_sahi_detections(
        all_raw, H, W,
        iou_threshold=nms_iou,
        score_threshold=float(threshold))

    total_ms = int((time.time() - t0) * 1000)
    return {
        'detections': merged,
        'raw_count': len(all_raw),
        'num_slices': len(slices) + (1 if use_full else 0),
        'num_tiles': len(slices),
        'use_full': bool(use_full),
        'slice_ms': int(slice_ms),
        'full_ms': int(full_ms),
        'total_ms': total_ms,
        'slices': slices,
    }


def map_baseline_to_original(detector_result, orig_w, orig_h):
    """Map box baseline (he resized) ve he anh goc de so sanh voi SAHI."""
    rw, rh = detector_result['size']
    sx = float(orig_w) / float(rw) if rw else 1.0
    sy = float(orig_h) / float(rh) if rh else 1.0
    out = []
    for d in detector_result['detections']:
        x1, y1, x2, y2 = d['box']
        out.append({
            'box': [float(x1 * sx), float(y1 * sy),
                    float(x2 * sx), float(y2 * sy)],
            'score': float(d['score']),
            'label': int(d['label']),
            'name': d.get('name', ''),
            'color': d.get('color', '#888888'),
        })
    out.sort(key=lambda d: -d['score'])
    return out
