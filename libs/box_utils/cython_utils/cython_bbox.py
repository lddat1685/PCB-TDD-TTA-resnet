# -*- coding: utf-8 -*-
"""Pure-numpy drop-in for cython_utils/cython_bbox.pyx.

The original module is Cython and needs a C compiler; this machine has none.
`bbox_overlaps` is imported at module scope by
  libs/detection_oprations/anchor_target_layer_without_boxweight.py
  libs/detection_oprations/proposal_target_layer.py
so the import must succeed even for inference, where the function is never called.

Semantics match bbox.pyx exactly: coordinates are inclusive (the +1 width/height
convention), inputs are float32 (N, 4) / (K, 4), output is float32 (N, K).
"""

from __future__ import absolute_import, division, print_function

import numpy as np


def bbox_overlaps(boxes, query_boxes):
    """(N, K) IoU matrix using the inclusive +1 box convention."""
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return overlaps

    # (K,) query areas
    box_area = ((query_boxes[:, 2] - query_boxes[:, 0] + 1) *
                (query_boxes[:, 3] - query_boxes[:, 1] + 1))

    for n in range(N):
        iw = (np.minimum(boxes[n, 2], query_boxes[:, 2]) -
              np.maximum(boxes[n, 0], query_boxes[:, 0]) + 1)
        ih = (np.minimum(boxes[n, 3], query_boxes[:, 3]) -
              np.maximum(boxes[n, 1], query_boxes[:, 1]) + 1)
        valid = (iw > 0) & (ih > 0)
        if not valid.any():
            continue
        area_n = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
        inter = iw * ih
        ua = np.asarray(area_n, dtype=np.float64) + box_area.astype(np.float64) - inter
        overlaps[n, valid] = (inter[valid] / ua[valid]).astype(np.float32)

    return overlaps


def bbox_overlaps_float(boxes, query_boxes):
    """Exclusive-convention variant (no +1), as in bbox.pyx."""
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)

    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return overlaps

    box_area = ((query_boxes[:, 2] - query_boxes[:, 0]) *
                (query_boxes[:, 3] - query_boxes[:, 1]))

    for n in range(N):
        iw = (np.minimum(boxes[n, 2], query_boxes[:, 2]) -
              np.maximum(boxes[n, 0], query_boxes[:, 0]))
        ih = (np.minimum(boxes[n, 3], query_boxes[:, 3]) -
              np.maximum(boxes[n, 1], query_boxes[:, 1]))
        valid = (iw > 0) & (ih > 0)
        if not valid.any():
            continue
        area_n = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
        inter = iw * ih
        ua = np.asarray(area_n, dtype=np.float64) + box_area.astype(np.float64) - inter
        overlaps[n, valid] = (inter[valid] / ua[valid]).astype(np.float32)

    return overlaps


def bbox_overlaps_self(boxes, query_boxes):
    return bbox_overlaps(boxes, query_boxes)
