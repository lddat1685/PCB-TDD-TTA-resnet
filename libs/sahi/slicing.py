# -*- coding: utf-8 -*-
"""Helpers chia slice, map toa do, gop ket qua cho SAHI (khong phu thuoc TF)."""

from __future__ import absolute_import, division, print_function

import numpy as np


def get_slices(image_h, image_w, slice_h, slice_w, overlap_h=0.2, overlap_w=0.2):
    """Sinh danh sach slice (y_min, x_min, y_max, x_max) phu toan bo anh."""
    slices = []
    step_y = max(1, int(slice_h * (1.0 - overlap_h)))
    step_x = max(1, int(slice_w * (1.0 - overlap_w)))

    y = 0
    while y < image_h:
        x = 0
        while x < image_w:
            y_min, x_min = y, x
            y_max = min(y + slice_h, image_h)
            x_max = min(x + slice_w, image_w)
            if y_max - y_min < slice_h and y_min > 0:
                y_min = max(0, y_max - slice_h)
            if x_max - x_min < slice_w and x_min > 0:
                x_min = max(0, x_max - slice_w)
            slices.append((y_min, x_min, y_max, x_max))
            if x_max >= image_w:
                break
            x += step_x
        if y_max >= image_h:
            break
        y += step_y
    return slices


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_per_class(dets, iou_threshold=0.5):
    """NMS theo tung label. dets: list dict co keys box/score/label."""
    if not dets:
        return []
    by_label = {}
    for d in dets:
        by_label.setdefault(int(d['label']), []).append(d)
    keep = []
    for label, group in by_label.items():
        group = sorted(group, key=lambda d: -d['score'])
        suppressed = [False] * len(group)
        for i in range(len(group)):
            if suppressed[i]:
                continue
            keep.append(group[i])
            for j in range(i + 1, len(group)):
                if suppressed[j]:
                    continue
                if _iou(group[i]['box'], group[j]['box']) > iou_threshold:
                    suppressed[j] = True
    return keep


def merge_sahi_detections(all_slice_dets, image_h, image_w,
                          iou_threshold=0.5, score_threshold=0.0):
    """Gop detections tu cac slice: clip ve anh goc, loc score, NMS theo lop."""
    merged = []
    for d in all_slice_dets:
        x1, y1, x2, y2 = [float(v) for v in d['box']]
        x1 = min(max(x1, 0.0), float(image_w))
        x2 = min(max(x2, 0.0), float(image_w))
        y1 = min(max(y1, 0.0), float(image_h))
        y2 = min(max(y2, 0.0), float(image_h))
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue
        merged.append({
            'box': [x1, y1, x2, y2],
            'score': float(d['score']),
            'label': int(d['label']),
            'name': d.get('name', 'class_%d' % int(d['label'])),
            'color': d.get('color', '#888888'),
        })
    merged = [d for d in merged if d['score'] >= score_threshold]
    merged = nms_per_class(merged, iou_threshold=iou_threshold)
    merged.sort(key=lambda d: -d['score'])
    return merged
