# -*- coding: utf-8 -*-
"""Sinh bang tracking: chi phi inference + chi so chinh (file MOI).

Doc:  output/phases/*/metrics.json + output/phases/usage_log.csv
Xuat: output/phases/TRACKING.md + output/phases/dashboard.html (mo truc tiep)

Don vi chi phi (vi TDD-Net khong phai LLM -> khong co token):
  1 forward pass qua Detector = 1 don vi tinh (tuong duong 1 token).
  forwards = tong tiles (baseline 1/anh, SAHI = so slice + full, TTA = 2).

Chay: python\\python.exe tools\\gen_tracking.py
"""
from __future__ import absolute_import, division, print_function

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, 'output', 'phases')


def load_phases():
    phases = []
    if not os.path.isdir(OUT):
        return phases
    for pid in sorted(os.listdir(OUT)):
        mp = os.path.join(OUT, pid, 'metrics.json')
        if os.path.isfile(mp):
            with open(mp) as f:
                phases.append(json.load(f))
    return phases


def load_usage():
    p = os.path.join(OUT, 'usage_log.csv')
    rows = []
    if os.path.isfile(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def fmt(x, nd=3):
    if x is None or x == '' or x == 'None':
        return '-'
    try:
        return ('%.' + str(nd) + 'f') % float(x)
    except (TypeError, ValueError):
        return str(x)


def main():
    phases = load_phases()
    if not phases:
        print('Chua co metrics.json. Chay tools/phase_loop.py truoc.')
        return 1
    usage = load_usage()

    # ---------- TRACKING.md ----------
    with open(os.path.join(OUT, 'TRACKING.md'), 'w') as md:
        md.write('# Tracking — chi phi + chi so chinh (TDD-Net PCB)\n\n')
        md.write('> TDD-Net la mang Faster-RCNN, **khong phai LLM nen khong co token**. '
                 'Bang nay dung **1 forward pass = 1 don vi** (tuong duong token) de so chi phi giua cac phase.\n\n')
        md.write('## A. Bang chi phi (tuong duong token usage)\n\n')
        md.write('| phase | ten | forwards (don vi) | tong ms | ms/anh | box | ms/box | forwards/box |\n')
        md.write('|---|---|---|---|---|---|---|---|\n')
        for ph in phases:
            rec = ph['phase']; rows = ph['rows']
            fw = sum(r['tiles'] for r in rows)
            ms = sum(r['ms'] for r in rows)
            n = rec['total_box']
            md.write('| %s | %s | %d | %d | %.1f | %d | %.1f | %.2f |\n'
                     % (rec['id'], rec['name'], fw, ms, rec['avg_ms'], n,
                        ms / max(1, n), fw / max(1, n)))
        md.write('\n## B. Bang chi so chinh xac (IoU>=0.5, anh co GT)\n\n')
        md.write('| phase | GT | TP | FP | FN | Precision | Recall | F1 | nhan xet |\n')
        md.write('|---|---|---|---|---|---|---|---|---|\n')
        best = max([p['phase'] for p in phases if p['phase'].get('f') is not None],
                   key=lambda x: x['f'], default=None)
        for ph in phases:
            rec = ph['phase']
            tag = ' <-- TOT NHAT' if (best and rec['id'] == best['id']) else ''
            if rec.get('gt_total'):
                md.write('| %s | %d | %d | %d | %d | %.3f | %.3f | %.3f |%s |\n'
                         % (rec['id'], rec['gt_total'], rec['tp'], rec['fp'], rec['fn'],
                            rec['p'], rec['r'], rec['f'], tag))
            else:
                md.write('| %s | - | - | - | - | - | - | - | (khong GT) |\n' % rec['id'])
        md.write('\n## C. Lich su chay (usage_log.csv — cong don theo thoi gian)\n\n')
        if usage:
            md.write('| thoi diem | phase | anh | forwards | ms | ms/anh | box | P | R | F1 |\n')
            md.write('|---|---|---|---|---|---|---|---|---|---|\n')
            for u in usage[-30:]:
                md.write('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n'
                         % (u.get('timestamp'), u.get('phase_id'), u.get('n_img'),
                            u.get('forwards_tiles'), u.get('ms_total'), u.get('ms_per_img'),
                            u.get('boxes'), fmt(u.get('P')), fmt(u.get('R')), fmt(u.get('F1'))))
        else:
            md.write('(chua co log — chay lai phase_loop.py de ghi.)\n')
        md.write('\n## D. Chi so he thong (moi lan chay)\n\n')
        md.write('| chi so | gia tri |\n|---|---|\n')
        md.write('| checkpoint | FPN_Res101_0117_OHEM/pcb_30000model.ckpt (467 MB, 680 tensor) |\n')
        md.write('| backbone | ResNet-101 + FPN P2-P6 + RPN share-head + FastRCNN FC1024x2 |\n')
        md.write('| device | CPU (GTX 1650 thieu cuDNN) |\n')
        md.write('| preprocess | resize canh ngan 600, max 3000, tru PIXEL_MEAN |\n')

    # ---------- dashboard.html ----------
    data = json.dumps([{'id': p['phase']['id'], 'name': p['phase']['name'],
                        'params': p['phase']['params'],
                        'forwards': sum(r['tiles'] for r in p['rows']),
                        'ms': sum(r['ms'] for r in p['rows']),
                        'avg_ms': p['phase']['avg_ms'], 'box': p['phase']['total_box'],
                        'gt': p['phase'].get('gt_total') or 0,
                        'tp': p['phase'].get('tp') or 0, 'fp': p['phase'].get('fp') or 0,
                        'fn': p['phase'].get('fn') or 0,
                        'p': p['phase'].get('p'), 'r': p['phase'].get('r'),
                        'f': p['phase'].get('f')} for p in phases], ensure_ascii=False)
    html = DASH_HTML.replace('__DATA__', data)
    with open(os.path.join(OUT, 'dashboard.html'), 'w') as f:
        f.write(html)
    print('Da viet: TRACKING.md, dashboard.html (%d phase, %d dong log)'
          % (len(phases), len(usage)))
    return 0


DASH_HTML = """<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<title>TDD-Net · Tracking Dashboard</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--border:#2a2f3a;--text:#e6e9ef;--muted:#8b93a7;--acc:#4f8cff;--ok:#3fb950;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 "Segoe UI",Roboto,system-ui,sans-serif;padding:20px 28px}
h1{font-size:18px;margin:0 0 4px}.sub{color:var(--muted);font-size:12px;margin-bottom:16px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px 16px;min-width:180px;flex:1}
.card .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:22px;font-weight:700;margin-top:4px}.card .v small{font-size:12px;color:var(--muted);font-weight:400}
table{width:100%;border-collapse:collapse;font-size:12.5px;background:var(--panel);border-radius:10px;overflow:hidden;margin-bottom:22px}
th,td{padding:7px 10px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}
th{color:var(--muted);font-weight:500;background:#141821}td.num{text-align:right;font-variant-numeric:tabular-nums}
tr.best td{background:#12261c}.tag{font-size:10px;background:#1d3a2a;color:var(--ok);border-radius:4px;padding:1px 6px;margin-left:6px}
.bar{height:8px;background:#242a36;border-radius:4px;min-width:120px}.bar>span{display:block;height:100%;border-radius:4px;background:var(--acc)}
.bar.p>span{background:var(--ok)}.bar.r>span{background:#58a6ff}.bar.f>span{background:var(--warn)}
.note{color:var(--muted);font-size:12px;margin-top:6px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:18px 0 8px}
</style></head><body>
<h1>TDD-Net · Tracking Dashboard</h1>
<div class="sub">FPN_Res101_0117_OHEM · ResNet-101 + FPN P2-P6 + FastRCNN · CPU · 1 forward = 1 don vi chi phi (tuong duong token)</div>
<div class="cards" id="cards"></div>
<h2>A · Chi phi (cang thap cang re)</h2>
<table id="cost"><thead><tr><th>Phase</th><th class="num">Forwards</th><th class="num">Tong ms</th><th class="num">ms/anh</th><th>ms/anh</th><th class="num">Box</th><th class="num">ms/box</th></tr></thead><tbody></tbody></table>
<h2>B · Do chinh xac (IoU&gt;=0.5, anh co GT)</h2>
<table id="qual"><thead><tr><th>Phase</th><th class="num">GT</th><th class="num">TP/FP/FN</th><th>P</th><th>R</th><th>F1</th></tr></thead><tbody></tbody></table>
<div class="note">Mo file <b>TRACKING.md</b> de xem lich su chay chi tiet (usage_log.csv). Chay lai <b>tools/phase_loop.py</b> roi <b>tools/gen_tracking.py</b> de cap nhat so.</div>
<script>
const D=__DATA__;
const maxMs=Math.max(...D.map(d=>d.ms)),maxBox=Math.max(...D.map(d=>d.box));
let bestF=null,bestId=null;D.forEach(d=>{if(d.f!=null&&(bestF==null||d.f>bestF)){bestF=d.f;bestId=d.id}});
let totFw=D.reduce((a,d)=>a+d.forwards,0),totMs=D.reduce((a,d)=>a+d.ms,0);
document.getElementById('cards').innerHTML=
 `<div class="card"><div class="k">Tong forwards (7 phase)</div><div class="v">${totFw} <small>don vi</small></div></div>`+
 `<div class="card"><div class="k">Tong thoi gian</div><div class="v">${(totMs/1000).toFixed(1)}s</div></div>`+
 `<div class="card"><div class="k">F1 cao nhat</div><div class="v">${bestF!=null?bestF.toFixed(3):'-'} <small>${bestId||''}</small></div></div>`+
 `<div class="card"><div class="k">Nhanh nhat/anh</div><div class="v">${Math.min(...D.map(d=>d.avg_ms))} <small>ms</small></div></div>`;
document.querySelector('#cost tbody').innerHTML=D.map(d=>
 `<tr><td><b>${d.id}</b><br><span style="color:#8b93a7">${d.name}</span></td><td class="num">${d.forwards}</td><td class="num">${d.ms}</td><td class="num">${d.avg_ms}</td><td><div class="bar"><span style="width:${(100*d.ms/maxMs).toFixed(1)}%"></span></div></td><td class="num">${d.box}</td><td class="num">${(d.ms/Math.max(1,d.box)).toFixed(1)}</td></tr>`).join('');
const bar=(v,c)=>v==null?'-':`<div style="display:flex;align-items:center;gap:6px"><div class="bar ${c}" style="flex:1"><span style="width:${(100*v).toFixed(1)}%"></span></div><b>${v.toFixed(3)}</b></div>`;
document.querySelector('#qual tbody').innerHTML=D.map(d=>
 `<tr class="${d.id===bestId?'best':''}"><td><b>${d.id}</b>${d.id===bestId?'<span class="tag">TOT NHAT</span>':''}</td><td class="num">${d.gt||'-'}</td><td class="num">${d.gt?d.tp+'/'+d.fp+'/'+d.fn:'-'}</td><td>${bar(d.p,'p')}</td><td>${bar(d.r,'r')}</td><td>${bar(d.f,'f')}</td></tr>`).join('');
</script></body></html>"""


if __name__ == '__main__':
    sys.exit(main())
