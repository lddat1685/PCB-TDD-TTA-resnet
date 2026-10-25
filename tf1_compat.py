# -*- coding: utf-8 -*-
"""Compatibility shim: make TF1-era code (this repo) import and run on TensorFlow 2.5.1.

Import this module BEFORE any repo module touches `tensorflow`, either via
`import tf1_compat` at the top of tools/inference.py or through
PYTHONPATH + sitecustomize.

What it does:
  * tensorflow.compat.v1  -> tensorflow           (placeholder, Session, ConfigProto, ...)
  * tf.contrib.slim       -> tf_slim              (conv2d, arg_scope, nets, ...)
  * tf.contrib.layers     -> tf_slim.layers / tf.contrib v1 leftovers
  * tf.pack / tf.to_float / tf.Print / ...        (removed aliases)
  * tf_slim.python.slim.nets.*                    (path tf_slim dropped)
"""

from __future__ import absolute_import, division, print_function

import sys
import types

import tensorflow as tf

# ---------------------------------------------------------------- conv2d
# TF2's keras-backed conv2d hands a `reuse` kwarg through to Layer; the repo
# passes reuse=True on RPN/fuse heads. Force the classic v1 implementation.
try:
    from tensorflow.python.keras.layers.convolutional import Conv2D as _KConv2D
    from tf_slim.layers import layers as _tf_slim_layers
    from tf_slim.ops import variables as _tf_slim_variables
    from tf_slim import ops as _tf_slim_ops

    _tf_slim_layers.Conv2D = _KConv2D  # noqa: F811 - restore slim's expected symbol
except Exception:  # pragma: no cover - best effort
    pass

import tf_slim as _slim
import tf_slim.layers as _slim_layers
import tf_slim.nets as _slim_nets

# ------------------------------------------------------ tensorflow.compat.v1
_v1 = tf.compat.v1

# Modules that moved wholesale into compat.v1.
_MOVED_TO_V1 = {
    'Session', 'placeholder', 'ConfigProto', 'GPUOptions', 'GraphDef', 'Graph',
    'get_default_graph', 'reset_default_graph', 'set_random_seed', 'control_dependencies',
    'name_scope', 'variable_scope', 'get_variable', 'get_variable_scope',
    'global_variables_initializer', 'local_variables_initializer', 'trainable_variables',
    'global_variables', 'all_variables',
    'get_collection', 'add_graph', 'import_graph_def', 'device', 'assign',
    'GraphKeys', 'log', 'random_uniform', 'truncated_normal_initializer',
    'train', 'summary', 'nn', 'image',
}
for _name in _MOVED_TO_V1:
    if not hasattr(tf, _name) and hasattr(_v1, _name):
        setattr(tf, _name, getattr(_v1, _name))

# Removed aliases / renamed symbols.
if not hasattr(tf, 'pack'):
    tf.pack = tf.stack
if not hasattr(tf, 'to_float'):
    tf.to_float = _v1.to_float
if not hasattr(tf, 'to_int32'):
    tf.to_int32 = _v1.to_int32
if not hasattr(tf, 'to_int64'):
    tf.to_int64 = _v1.to_int64
if not hasattr(tf, 'Print'):
    tf.Print = _v1.Print
if not hasattr(tf, 'sparse_to_dense'):
    tf.sparse_to_dense = _v1.sparse_to_dense

# Sub-modules that only exist under compat.v1.
tf.train = _v1.train
tf.logging = _v1.logging
tf.app = _v1.app
tf.image.resize_bilinear = _v1.image.resize_bilinear
tf.image.flip_left_right = _v1.image.flip_left_right
tf.image.crop_and_resize = _v1.image.crop_and_resize
tf.image.non_max_suppression = _v1.image.non_max_suppression
tf.py_func = _v1.py_func
tf.gfile = _v1.gfile

# tf.compat.v1 disables eager by default but be explicit for graph-mode code.
_v1.disable_eager_execution()

# ------------------------------------------------------------ tf.contrib.slim
_contrib = types.ModuleType('tensorflow.contrib')
_contrib_slim = types.ModuleType('tensorflow.contrib.slim')

for _attr in dir(_slim):
    if not _attr.startswith('__'):
        setattr(_contrib_slim, _attr, getattr(_slim, _attr))
_contrib_slim.__path__ = list(getattr(_slim, '__path__', []))

# Paths the repo imports directly.
_contrib_slim.nets = _slim_nets
_contrib_slim.layers = _slim_layers
_contrib_slim.slim = _slim

_python_mod = types.ModuleType('tensorflow.contrib.slim.python')
_slim_py = types.ModuleType('tensorflow.contrib.slim.python.slim')
_slim_py_nets = types.ModuleType('tensorflow.contrib.slim.python.slim.nets')
for _attr in ('resnet_v1', 'resnet_utils', 'resnet_v2', 'vgg', 'inception'):
    if hasattr(_slim_nets, _attr):
        setattr(_slim_py_nets, _attr, getattr(_slim_nets, _attr))
_slim_py.nets = _slim_py_nets
_python_mod.slim = _slim_py
_contrib_slim.python = _python_mod

_contrib.slim = _contrib_slim

# contrib.layers: only l2_regularizer / softmax are used by this repo.
# tf_slim.layers is a module (not a package) and lacks l2_regularizer, so take
# both from the tf_slim top level where they are exported.
_contrib_layers = types.ModuleType('tensorflow.contrib.layers')
for _attr in ('l2_regularizer', 'softmax'):
    if hasattr(_slim, _attr):
        setattr(_contrib_layers, _attr, getattr(_slim, _attr))
_contrib.layers = _contrib_layers

sys.modules['tensorflow.contrib'] = _contrib
sys.modules['tensorflow.contrib.slim'] = _contrib_slim
sys.modules['tensorflow.contrib.slim.nets'] = _slim_nets
sys.modules['tensorflow.contrib.slim.layers'] = _slim_layers
sys.modules['tensorflow.contrib.slim.python'] = _python_mod
sys.modules['tensorflow.contrib.slim.python.slim'] = _slim_py
sys.modules['tensorflow.contrib.slim.python.slim.nets'] = _slim_py_nets
_python_mod.__path__ = []
_slim_py.__path__ = []
_slim_py_nets.__path__ = []
# Leaf modules so `from ...nets.resnet_v1 import resnet_v1_block` resolves.
for _leaf in ('resnet_v1', 'resnet_utils', 'resnet_v2', 'vgg', 'inception', 'inception_v1',
              'inception_v2', 'inception_v3', 'alexnet', 'overfeat'):
    _mod = getattr(_slim_nets, _leaf, None)
    if isinstance(_mod, types.ModuleType):
        sys.modules['tensorflow.contrib.slim.python.slim.nets.' + _leaf] = _mod
        sys.modules['tensorflow.contrib.slim.nets.' + _leaf] = _mod
if hasattr(_contrib, 'layers'):
    sys.modules['tensorflow.contrib.layers'] = _contrib.layers

tf.contrib = _contrib
