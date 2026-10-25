# -*- coding: utf-8 -*-
"""Thanh tien do Phase 09 (file MOI, chi doc log — khong cham logic train).

Chay bat cu luc nao de xem tien do:
  python\\python.exe tools\\watch_phase09.py
  python\\python.exe tools\\watch_phase09.py --loop 20 --every 60
"""
from __future__ import print_function
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(HERE, 'output', 'phases', 'phase_09_finetune', 'train_log.jsonl')
CKPT_DIR = os.path.join(HERE, 'output', 'trained_weights', 'FPN_Res101_0117_OHEM')
START_STEP = 30000
TARGET = START_STEP + 300  # --max_steps 300
SMOKE_CUT = 30016          # step <= nay la cua 2 lan smoke-test, bo qua


def load_cur():
    rows = []
    if not os.path.isfile(LOG):
        return rows
    with open(LOG, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get('step', 0) > SMOKE_CUT:
                rows.append(r)
    return rows


def bar(done, total, width=30):
    frac = max(0.0, min(1.0, done / float(total))) if total else 0.0
    fill = int(round(frac * width))
    return '[%s%s] %5.1f%% (%d/%d)' % ('#' * fill, '-' * (width - fill),
                                       frac * 100.0, done, total)


def snapshot():
    rows = load_cur()
    ckpts = sorted(f for f in os.listdir(CKPT_DIR)
                   if f.startswith('pcb_') and f.endswith('.index')) \
        if os.path.isdir(CKPT_DIR) else []
    if not rows:
        return {'line': 'Phase 09: chua co log train (run that) — %s' % bar(0, 300),
                'done': False}
    last = rows[-1]
    done = last['step'] - START_STEP
    tots = [r['total'] for r in rows]
    secs = [r.get('sec', 11.0) for r in rows]
    avg = sum(secs) / len(secs)
    remain = max(0, TARGET - last['step'])
    eta = remain * avg
    ckpt_new = [c.replace('.index', '') for c in ckpts if '30000' not in c]
    line = ('Phase 09 OHEM %s | step %d/%d | loss=%.4f (min %.4f) | '
            '~%.0fs/step | con ~%d step ~%.0f phut | ckpt moi: %s'
            % (bar(done, 300), last['step'], TARGET, last['total'],
               min(tots), avg, remain, eta / 60.0,
               ', '.join(ckpt_new) if ckpt_new else 'chua co (ngoai pcb_30000)'))
    # tien do that ket thuc khi train ghi summary hoac du step
    done_flag = last['step'] >= TARGET - 1
    return {'line': line, 'done': done_flag, 'last': last, 'n': len(rows)}


def main():
    loop = 1
    every = 0
    if '--loop' in sys.argv:
        loop = int(sys.argv[sys.argv.index('--loop') + 1])
    if '--every' in sys.argv:
        every = int(sys.argv[sys.argv.index('--every') + 1])
    for i in range(loop):
        s = snapshot()
        print(s['line'], flush=True)
        # ghi file de UI/dashboard doc ve sau
        try:
            prog = os.path.join(HERE, 'output', 'phases', 'phase_09_finetune',
                                'PROGRESS.md')
            with open(prog, 'w', encoding='utf-8') as f:
                f.write('# Phase 09 — tien do\n\n%s\n' % s['line'])
        except Exception:
            pass
        if s['done'] or i == loop - 1:
            break
        time.sleep(every or 60)
    return 0


if __name__ == '__main__':
    sys.exit(main())
