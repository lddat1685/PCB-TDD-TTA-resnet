# -*- coding: utf-8 -*-
"""Tach moi phase thanh ho so doc lap de so sanh (file MOI, khong sua logic goc).

Doc:  output/phases/phase_*/metrics.json
Xuat: output/phases/cards/PHASE_01_...md (7 file doc lap)
      output/phases/cards/SO_SANH_TONG_HOP.md (bang so sanh ngang)
Moi card tu chua: kien truc | tap data test | ket qua | thay doi vs phase truoc.
Chay: python\\python.exe tools\\gen_phase_cards.py
"""
from __future__ import absolute_import, division, print_function
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASES_DIR = os.path.join(HERE, 'output', 'phases')
CARDS_DIR = os.path.join(PHASES_DIR, 'cards')

ORDER = [
    'phase_01_baseline',
    'phase_02_sahi_default',
    'phase_03_sahi_precision',
    'phase_04_sahi_fast',
    'phase_05_sahi_nofull',
    'phase_06_classwise',
    'phase_07_tta_flip',
    'phase_08_fusion',
    'phase_09_finetuned',
]

ARCH_CORE_ROWS = [
    ("Backbone", "ResNet-101 (`resnet_v1_101`): conv1 7x7 s2 + maxpool; block1 3u/d64, block2 4u/d128, block3 23u/d256, block4 3u/d512 s1. BN dong bang (is_training=False)"),
    ("Feature tap", "C2/C3/C4/C5 tai bottleneck cuoi moi block (`libs/networks/resnet.py:resnet_base`)"),
    ("FPN", "P5=conv1x1/256(C5); P6=maxpool(P5,s2); P4..P2=fusion_two_layer + conv3x3/256. Levels P2-P6, stride 4/8/16/32/64"),
    ("Anchor", "BASE [15,25,40,60,80] x SCALES [2,3,4] x RATIOS [2,3,4,5] = 12 anchor/diem (`cfgs.py`)"),
    ("RPN head", "Share-head conv3x3/512 + cls 1x1(24) + reg 1x1(48). NMS 0.7, test topK 6000->1000"),
    ("ROI", "assign_levels 4+log2(sqrt(wh)/224); ROI-warp; concat 4 level"),
    ("FastRCNN head", "`restnet_head`: flatten + FC1024 + FC1024, cls_fc(7) + reg_fc(28). KEEP_PROB=1.0 (tat dropout)"),
    ("Post trong graph", "decode + clip + NMS theo lop IoU 0.3, toi da 100 box/lop; loc score>=threshold (graph tensor)"),
    ("Preprocess", "short_side_resize canh ngan 600, max 3000; tru PIXEL_MEAN [21.25,85.936,28.729]"),
    ("Checkpoint (dung chung)", "FPN_Res101_0117_OHEM/pcb_30000model.ckpt (~466 MB, 548 variables). Tai su dung, KHONG sua/train lai"),
]

# Mo ta kien truc rieng tung phase (phan doc lap, de so sanh "dung kien truc gi")
ARCH_EXTRA = {
    'phase_01_baseline': (
        "Baseline thuan — KHONG cat slice, KHONG TTA.",
        ["1 forward Detector/anh (`ui/app.py:Detector.run`, threshold=0.6 trong graph)",
         "Box tu he resize -> map ve anh goc (`map_baseline_to_original`: nhan scale sx=W/rw, sy=H/rh)",
         "Khong NMS ngoai graph (chi NMS IoU 0.3 san trong FastRCNN post)"]),
    'phase_02_sahi_default': (
        "Baseline + SAHI mac dinh (slice nho, phu day).",
        ["`get_slices`: cat 512x512, overlap 0.2 -> ~10 tile/anh 1000x1000 + 1 anh full = 11 forwards (thuc te TB 5.86 vi anh nho it tile hon)",
         "Chay lai Detector goc tren TUNG slice + anh full (`sahi_detect`), map box slice ve anh goc (offset + scale)",
         "`merge_sahi_detections`: gop + NMS theo lop IoU 0.5, loc score>=0.6",
         "Muc dich: tang recall loi tiny (missing_hole/spur nho) nhung chap nhan them FP + cham hon"]),
    'phase_03_sahi_precision': (
        "SAHI giong phase_02 nhung siet manh de tri FP.",
        ["Slice 512 overlap 0.2 + full (giong phase_02) -> TB 5.86 tile/anh",
         "Nang threshold 0.6 -> 0.7 (loc ngay trong tung slice)",
         "NMS gop chat hon 0.5 -> 0.4 (tri box trung lap bien slice)",
         "Muc dich: giu recall, giam FP open_circuit/spurious_copper"]),
    'phase_04_sahi_fast': (
        "SAHI toc do (slice to, it tile).",
        ["Slice 640x640 overlap 0.25 -> ~5 tile + full = TB 3.14 tile/anh (giam ~46% forwards vs phase_02)",
         "Threshold 0.6, NMS gop 0.5, co anh full",
         "Muc dich: giu do phu SAHI nhung chay nhanh gap ~1.8x"]),
    'phase_05_sahi_nofull': (
        "SAHI ablation — BO nhanh anh full (chi slice).",
        ["Slice 512 overlap 0.2, KHONG chay anh full -> TB 5.00 tile/anh",
         "Threshold 0.6, NMS gop 0.5",
         "Muc dich (ablation): do rieng dong gop cua nhanh full — xem box mat di hay FP giam khi bo full"]),
    'phase_06_classwise': (
        "SAHI nhanh + threshold RIENG THEO LOP (tri FP co chon loc).",
        ["Nen slice640 overlap 0.25 + full (giong phase_04) -> TB 3.14 tile/anh",
         "Loc 2 tang: chay slice voi base_thr=0.6, sau do loc lai theo tung lop: missing_hole/short/mouse_bite>=0.6 (giu recall), spur>=0.65, open_circuit/spurious_copper>=0.8 (siet FP)",
         "NMS gop chat 0.3 (gom box trung bien slice manh tay)",
         "Muc dich: tri FP nhom yeu ma khong mat recall nhom manh"]),
    'phase_07_tta_flip': (
        "Baseline + TTA lat ngang (KHONG slice, KHONG train).",
        ["Chay Detector 2 lan: anh goc + anh lat ngang, threshold 0.6",
         "Flip box ve: x1'=W-x2, x2'=W-x1 (`phase_loop.py` nhanh tta_flip)",
         "Gop 2 tap box + `nms_per_class` IoU 0.5, loc score>=0.6 -> TB 2.00 forwards/anh",
         "Muc dich: tang on dinh (box nao hien ca 2 ban lat thi giu) voi chi phi re hon SAHI"]),
    'phase_08_fusion': (
        "Fusion phase06 + phase07 (KHONG slice, KHONG train).",
        ["Chay Detector 2 lan (goc + lat ngang, threshold 0.6) nhu phase07 -> TB 2.00 forwards/anh",
         "Flip box ve x1'=W-x2; loc class-wise nhu phase06: missing_hole/short/mouse_bite>=0.6, spur>=0.65, open_circuit/spurious_copper>=0.8",
         "Gop + `nms_per_class` IoU 0.5. Tren tap GT hien co, loc class-wise khong cat them box nao so voi phase07 (vi box yeu da bi TTA loai)",
         "Muc dich: lay on dinh cua TTA + bao hiem class-wise cho tap GT lon hon sau nay"]),
    'phase_09_finetuned': (
        "Fusion phase08 + CHECKPOINT FINE-TUNE that (OHEM 300 step, lr 0.0003, tu pcb_30000 -> pcb_30300).",
        ["Train that tren tfrecord PCB that: train 8534 anh/17342 box, tf.data pipeline (thay queue cu hong tren TF2 graph)",
         "OHEM (FAST_RCNN_MINIBATCH_SIZE=-1, sum_ohem_loss 128) + MomentumOptimizer; loss TB dau 0.9295 -> cuoi 0.9261, min 0.6856, 55.8 phut CPU",
         "Eval dung pipeline fusion Y HET phase_08 (TTA flip x2 + class-wise + NMS 0.5) nhung checkpoint = pcb_30300model.ckpt (461MB)",
         "Ket qua: demo 7 anh F1=1.0 voi 29 box (it hon phase_08: 37); test that 100 anh P=0.951 R=0.961 F1=0.956"]),
}

# Mo ta tap data test (phase 01-08: 7 anh demo, GT han che; phase 09 them test that 100 anh)
DATASET_NOTE = (
    "Phase 01-08 chay tren CUNG 1 tap 7 anh demo (fair compare, cung checkpoint pcb_30000, cung IoU>=0.5, cung he toa do anh goc). "
    "Chi 1/7 anh co GT (01_missing_hole_01.jpg, 3 box missing_hole) nen P/R/F1 micro chi phan anh tren anh do; "
    "6 anh con lai dung so box/mean score/thoi gian lam chi so proxy. "
    "Phase 09 chay tren CUNG 7 anh demo (de so cong bang pipeline, nhung checkpoint = pcb_30300 fine-tune) "
    "CONG THEM test that 100 anh rut tu pcb_test.tfrecord (180 GT, du 6 lop) -> P/R/F1 that."
)
DATASET_ROWS = [
    ("input/m13.jpg", "1000x1000", "khong", "-", "Board xanh 1 mat, nhieu missing_hole + short"),
    ("input/s14.jpg", "1000x1000", "khong", "-", "Board doi (2 mat), chu yeu loi short"),
    ("input/spurs (12).jpg", "762x579", "khong", "-", "Cum spur day, co lan open_circuit/mouse_bite"),
    ("01_missing_hole_01.jpg", "1147x600 (file) / XML goc 3034x1586 scale ve", "CO", "3x missing_hole (XML test_annotation/01_missing_hole_01.xml)", "Anh DUY NHAT co GT — quyet dinh P/R/F1"),
    ("04_mouse_bite_10.jpg", "744x600", "khong", "-", "Ten mouse_bite nhung khong co XML di kem"),
    ("TDD_results.jpg", "563x443", "khong", "-", "Anh demo nho, model tra ve 0 box moi phase"),
    ("a(missinghole).png", "179x82", "khong", "-", "Anh tiny 82px cao — test gioi han duoi"),
]


def load_all():
    data = {}
    for pid in ORDER:
        mp = os.path.join(PHASES_DIR, pid, 'metrics.json')
        with open(mp, encoding='utf-8') as f:
            data[pid] = json.load(f)
    return data


def fmt(x, nd=3):
    if x is None:
        return '-'
    return ('%.' + str(nd) + 'f') % float(x)


def card_md(pid, rec, rows, prev):
    p = rec['params']
    title, bullets = ARCH_EXTRA[pid]
    L = []
    L.append('# %s — %s (ho so doc lap)\n' % (pid, rec['name']))
    L.append('> File nay doc doc lap duoc: kien truc | tap data | ket qua | thay doi vs phase truoc. '
             'Checkpoint + Dieu kien danh gia giong het moi phase.\n')
    L.append('## 0. Thong tin nhanh\n')
    L.append('| muc | noi dung |')
    L.append('|---|---|')
    L.append('| phase | `%s` |' % pid)
    L.append('| ten | %s |' % rec['name'])
    L.append('| params | `%s` |' % json.dumps(p, ensure_ascii=False))
    L.append('| y tuong (delta) | %s |' % rec['delta'])
    L.append('| checkpoint | FPN_Res101_0117_OHEM/pcb_30000model.ckpt (tai su dung, khong sua) |')
    L.append('| dieu kien | IoU TP>=0.5, he toa do anh goc, greedy match theo score |')
    L.append('')
    L.append('## 1. Kien truc su dung\n')
    L.append('### 1a. Loi chung (giong het cac phase — chu xung quanh Detector goc)\n')
    L.append('| nhom | chi tiet |')
    L.append('|---|---|')
    for k, v in ARCH_CORE_ROWS:
        L.append('| %s | %s |' % (k, v))
    L.append('')
    L.append('### 1b. Phan rieng cua %s: %s\n' % (pid, title))
    for b in bullets:
        L.append('- %s' % b)
    L.append('')
    L.append('## 2. Tap data test (giong het cac phase)\n')
    L.append(DATASET_NOTE + '\n')
    L.append('| # | anh (nguon) | kich thuoc | GT | so box GT | mo ta |')
    L.append('|---|---|---|---|---|---|')
    for i, (f, sz, gt, n, d) in enumerate(DATASET_ROWS, 1):
        L.append('| %d | %s | %s | %s | %s | %s |' % (i, f, sz, gt, n, d))
    L.append('')
    L.append('## 3. Ket qua cua RIENG phase nay\n')
    L.append('| anh | kich thuoc | GT | n box | ms | forwards(raw) | mean score | phan bo lop | P | R | F1 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        if r['has_gt']:
            L.append('| %s | %s | %s | %d | %d | %d(%d) | %.3f | %s | %.3f | %.3f | %.3f |'
                     % (r['image'], r['size'], r['gt'], r['n'], r['ms'], r['tiles'], r['raw'],
                        r['mean'], json.dumps(r['counts'], ensure_ascii=False), r['p'], r['r'], r['f']))
        else:
            L.append('| %s | %s | - | %d | %d | %d(%d) | %.3f | %s | - | - | - |'
                     % (r['image'], r['size'], r['n'], r['ms'], r['tiles'], r['raw'],
                        r['mean'], json.dumps(r['counts'], ensure_ascii=False)))
    L.append('')
    fw = sum(r['tiles'] for r in rows)
    ms = sum(r['ms'] for r in rows)
    L.append('**Tong hop:** %d anh | %d box | TB %.1f ms/anh | TB %.2f forwards/anh | tong %d forwards, %d ms | '
             'ms/box %.1f | GT=%s TP=%s FP=%s FN=%s | **P=%s R=%s F1=%s**\n'
             % (rec['n_img'], rec['total_box'], rec['avg_ms'], rec['avg_tiles'], fw, ms,
                ms / max(1, rec['total_box']),
                rec['gt_total'], rec['tp'], rec['fp'], rec['fn'],
                fmt(rec['p']), fmt(rec['r']), fmt(rec['f'])))
    L.append('')
    L.append('## 4. Thay doi so voi phase truoc\n')
    if prev is None:
        L.append('- Day la phase moc (baseline). Moi phase sau so voi moc nay.\n')
    else:
        pr = prev['phase']
        dbox = rec['total_box'] - pr['total_box']
        dms = rec['avg_ms'] - pr['avg_ms']
        dtiles = rec['avg_tiles'] - pr['avg_tiles']
        L.append('| chi so | %s | %s | delta |' % (pr['id'], pid))
        L.append('|---|---|---|---|')
        L.append('| tong box | %d | %d | %+d |' % (pr['total_box'], rec['total_box'], dbox))
        L.append('| TB ms/anh | %.1f | %.1f | %+.1f |' % (pr['avg_ms'], rec['avg_ms'], dms))
        L.append('| TB forwards/anh | %.2f | %.2f | %+.2f |' % (pr['avg_tiles'], rec['avg_tiles'], dtiles))
        if pr['f'] is not None and rec['f'] is not None:
            L.append('| P / R / F1 | %s / %s / %s | %s / %s / %s | %+.3f / %+.3f / %+.3f |'
                     % (fmt(pr['p']), fmt(pr['r']), fmt(pr['f']),
                        fmt(rec['p']), fmt(rec['r']), fmt(rec['f']),
                        rec['p'] - pr['p'], rec['r'] - pr['r'], rec['f'] - pr['f']))
        else:
            L.append('| P / R / F1 | - | - | (thieu GT) |')
        # nhan xet tu dong theo so
        if dbox > 0 and (rec['f'] or 0) < (pr['f'] or 0):
            L.append('')
            L.append('- Nhan xet: box TANG (+%d) nhung F1 GIAM -> them chu yeu la FP (dung nhu phase SAHI 02-06 tren anh GT: TP giu 3/3, FP 0->2).' % dbox)
        elif (rec['f'] or 0) > (pr['f'] or 0):
            L.append('')
            L.append('- Nhan xet: F1 TANG -> phase nay tri duoc FP ma van giu TP (truong hop phase_07: FP 2->0, TP van 3/3).')
        elif dms < 0:
            L.append('')
            L.append('- Nhan xet: NHANH HON (%.1f ms/anh) voi F1 giu nguyen — dang trade-off tot ve toc do.' % (-dms))
    L.append('')
    L.append('## 5. Khi nao dung phase nay\n')
    use = {
        'phase_01_baseline': 'Dung lam MOC doi chieu + deploy khi can nhanh nhat (1848 ms/anh) va F1=1.0 tren GT hien co.',
        'phase_02_sahi_default': 'Dung khi anh hon hop, nghi ngo loi tiny bi miss; chap nhan cham (~9.4s/anh) va can GT them de loc FP.',
        'phase_03_sahi_precision': 'Ban siet FP cua phase_02 (thr 0.7, NMS 0.4) nhung tren GT hien co FP van 2 — chi dung khi tap GT lon hon chung minh duoc.',
        'phase_04_sahi_fast': 'Ban SAHI can bang toc do/chat luong (slice 640, ~5.3s/anh). Nen dung thay phase_02 khi can SAHI.',
        'phase_05_sahi_nofull': 'Chi de ablation (chung minh nhanh full co dong gop). Khong deploy rieng vi cham hon ma F1 khong hon phase_04.',
        'phase_06_classwise': 'Ban SAHI tot nhat ve y tuong (threshold theo lop + NMS 0.3) nhung tren GT hien co van FP=2. Giu lai lam nen de fine-tune tiep.',
        'phase_07_tta_flip': 'Kien truc ON DINH truoc phase08: F1=1.0, box hop ly (37), chi 2 forwards/anh (~3.9s).',
        'phase_08_fusion': 'Kien truc ON DINH truoc fine-tune: F1=1.0, 37 box, 2 forwards/anh, nhanh hon phase07 (~3.5s).',
        'phase_09_finetuned': 'Kien truc TOT NHAT hien tai: F1 demo=1.0 (29 box — gon hon phase_08 nho fine-tune) + F1 test that 100 anh=0.956 (P=0.951 R=0.961). Uu tien deploy.',
    }
    L.append('- %s' % use[pid])
    L.append('')
    L.append('_Nguon so: `output/phases/%s/metrics.json` + `metrics.csv`. Dieu kien: IoU>=0.5, he anh goc._\n' % pid)
    return '\n'.join(L)


def compare_md(data):
    L = []
    L.append('# SO SANH NGANG %d PHASE — TDD-Net PCB (doc 1 file la thay het)\n' % len(ORDER))
    L.append('> Cung checkpoint FPN_Res101_0117_OHEM, cung 7 anh, IoU>=0.5, he anh goc. '
             'P/R/F1 micro tren 1 anh co GT (01_missing_hole_01.jpg, 3 box).\n')
    L.append('## A. Moi phase dung kien truc gi\n')
    L.append('| phase | ten | kien truc them (ngoai loi chung) | params |')
    L.append('|---|---|---|---|')
    for pid in ORDER:
        rec = data[pid]['phase']
        title, _ = ARCH_EXTRA[pid]
        L.append('| %s | %s | %s | `%s` |' % (pid, rec['name'], title, json.dumps(rec['params'], ensure_ascii=False)))
    L.append('')
    L.append('Loi chung cac phase: ResNet-101 + FPN P2-P6 + RPN share-head + FastRCNN FC1024x2, '
             'preprocess short_side 600, checkpoint 466 MB (khong sua).\n')
    L.append('## B. Test tren tap data nao (giong het moi phase)\n')
    L.append('| # | anh | kich thuoc | GT | ghi chu |')
    L.append('|---|---|---|---|---|')
    for i, (f, sz, gt, n, d) in enumerate(DATASET_ROWS, 1):
        L.append('| %d | %s | %s | %s %s | %s |' % (i, f, sz, gt, n, d))
    L.append('')
    L.append('## C. Ket qua thay doi the nao (tong hop)\n')
    L.append('| phase | box | ms/anh | fwd/anh | ms/box | GT | TP | FP | FN | P | R | F1 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for pid in ORDER:
        rec = data[pid]['phase']
        rows = data[pid]['rows']
        ms = sum(r['ms'] for r in rows)
        L.append('| %s | %d | %.1f | %.2f | %.1f | %s | %s | %s | %s | %s | %s | %s |'
                 % (pid, rec['total_box'], rec['avg_ms'], rec['avg_tiles'], ms / max(1, rec['total_box']),
                    rec['gt_total'], rec['tp'], rec['fp'], rec['fn'], fmt(rec['p']), fmt(rec['r']), fmt(rec['f'])))
    L.append('')
    L.append('## D. Delta chuoi (phase N vs N-1) — ket qua thay doi the nao\n')
    L.append('| buoc | box | ms/anh | F1 | doc nhanh |')
    L.append('|---|---|---|---|---|')
    prev = None
    for pid in ORDER:
        rec = data[pid]['phase']
        if prev is None:
            L.append('| %s (moc) | %d | %.1f | %s | moc doi chieu |' % (pid, rec['total_box'], rec['avg_ms'], fmt(rec['f'])))
        else:
            pr = prev['phase']
            df = (rec['f'] or 0) - (pr['f'] or 0)
            if df < 0:
                note = 'SAHI sinh FP (F1 giam)'
            elif df > 0:
                note = 'tri FP thanh cong (F1 tang)'
            else:
                note = 'doi toc do/cau hinh (F1 giu nguyen)'
            L.append('| %s -> %s | %+d | %+.1f | %+.3f | %s |'
                     % (pr['id'], pid, rec['total_box'] - pr['total_box'],
                        rec['avg_ms'] - pr['avg_ms'], df, note))
        prev = data[pid]
    L.append('')
    L.append('Chi tiet tung anh (n box / ms):\n')
    L.append('| anh | size | GT | ' + ' | '.join(ORDER) + ' |')
    L.append('|---|---|---|' + '|'.join(['---'] * len(ORDER)) + '|')
    for i in range(len(data[ORDER[0]]['rows'])):
        r0 = data[ORDER[0]]['rows'][i]
        cells = ['%d/%d' % (data[pid]['rows'][i]['n'], data[pid]['rows'][i]['ms']) for pid in ORDER]
        L.append('| %s | %s | %s | %s |' % (r0['image'], r0['size'], r0['gt'], ' | '.join(cells)))
    L.append('')
    L.append('## E. Ket luan chon kien truc on dinh\n')
    L.append('- On dinh nhat: **phase_09_finetuned** (F1 demo=1.0, 29 box gon hon phase_08; F1 test that 100 anh=0.956, P=0.951 R=0.961). Uu tien deploy.')
    L.append('- Truoc fine-tune: phase_08_fusion (F1=1.0, 37 box, 2 forwards/anh, 3491 ms/anh — nhanh hon phase07).')
    L.append('- Nhanh nhat: phase_01_baseline (1848 ms/anh) nhung la moc, chua co TTA on dinh.')
    L.append('- SAHI (02-06) deu F1=0.75 tren GT demo (FP=2, TP=3/3) — chi co y nghia tham khao khi chua co GT day du.')
    L.append('- Fine-tune that: OHEM 300 step lr=0.0003 tu pcb_30000 -> pcb_30300, train 8534 anh/17342 box, 55.8 phut CPU.')
    L.append('')
    return '\n'.join(L)


def main():
    data = load_all()
    os.makedirs(CARDS_DIR, exist_ok=True)
    prev = None
    for pid in ORDER:
        rec = data[pid]['phase']
        rows = data[pid]['rows']
        num = pid.split('_')[1]
        outp = os.path.join(CARDS_DIR, 'PHASE_%s_%s.md' % (num, pid[len('phase_XX_'):]))
        with open(outp, 'w', encoding='utf-8') as f:
            f.write(card_md(pid, rec, rows, prev))
        print('wrote', outp)
        prev = data[pid]
    comp = os.path.join(CARDS_DIR, 'SO_SANH_TONG_HOP.md')
    with open(comp, 'w', encoding='utf-8') as f:
        f.write(compare_md(data))
    print('wrote', comp)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
