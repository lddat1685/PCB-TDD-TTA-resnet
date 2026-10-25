# RUN.md — how this repo was made to run on this machine

Original repo: `Tiny-Defect-Detection-for-PCB` (TDD-Net), TF 1.x code.
Target machine: Windows 11, Python 3.9.25 venv, TensorFlow 2.5.1, CPU only
(`cudnn64_8.dll` is missing, so TF silently falls back to CPU).

## What was changed

| File | Change |
|---|---|
| `tf1_compat.py` | **new.** Maps `tensorflow.compat.v1` symbols onto `tf`, and builds a fake `tensorflow.contrib.slim` package backed by `tf_slim`. Import it before anything imports TF. |
| `libs/box_utils/cython_utils/cython_bbox.py` | **new.** Pure-numpy `bbox_overlaps` (inclusive `+1` convention, float32). Replaces the unbuildable `.pyx` — no C compiler on this machine. |
| `tools/inference.py` | Adds `sys.path` setup + `import tf1_compat`; uses `os.path.basename` / `os.path.join` instead of `split('/')` for Windows paths; creates the save dir before writing. |
| `output/trained_weights/FPN_Res101_0117_OHEM/checkpoint` | Rewritten to a relative name, because the shipped file had a dead absolute Linux path. |
| `ui/app.py`, `ui/templates/index.html` | **new.** Local Flask UI: upload, live threshold, class filter, export. |
| venv | `protobuf 6.33.6 -> 3.20.3` (6.x makes TF 2.5.1 fail to import), `tf_slim 1.1.0`, `flask 3.1.3` installed. |

## Run — web UI

```bash
cd D:/FPTU/KLTN/Test_github/Tiny-Defect-Detection-for-PCB
.venv/Scripts/python.exe ui/app.py
# then open http://127.0.0.1:5000
```

The graph is built once at startup (~40 s); afterwards each image takes ~0.6–1.2 s
on CPU. The score threshold is a graph tensor, so moving the slider re-runs only the
forward pass — no graph rebuild. The UI can be launched from any working directory
(it `chdir`s into `tools/` itself, because `cfgs.py` resolves paths relative to CWD).

API: `GET /api/health`, `POST /api/detect` (multipart `image`, optional `threshold`).

## Run — CLI

```bash
cd D:/FPTU/KLTN/Test_github/Tiny-Defect-Detection-for-PCB/tools
../.venv/Scripts/python.exe inference.py --data_dir='../input' --save_dir='../output' --GPU='0'
```

Put images in `../input`, results land in `../output`.

## Verified

- Checkpoint: 680 tensors, `resnet_v1_101` backbone + `Fast-RCNN` head with 7 classes (6 defects + background).
- Loaded via the *trained* branch (`model restore from : .../output/trained_weights/.../pcb_30000model.ckpt`), not the ImageNet fallback.
- 3/3 demo images ran in ~0.7–1.3 s/image on CPU, all with sensible detections.

## Notes / limits

- CPU only. To use the RTX 3070 you need cuDNN 8.1 (`cudnn64_8.dll`) matching the installed CUDA 11.2.
- `train.py` / `eval.py` need `nms.pyx` and a C compiler; only the inference path was made to work.
- `libs/configs/cfgs.py` still contains hardcoded Linux paths (`test_annotate_path`); harmless for inference, breaks `eval.py`.
