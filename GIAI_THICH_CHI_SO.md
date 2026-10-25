# Vì sao chỉ số toàn số tròn (1.000 / 0.600 / 0.750) — có phải lỗi không?

**Không lỗi.** Nguyên nhân: tập demo 7 ảnh chỉ có đúng **1 ảnh có đáp án (GT)** —
`01_missing_hole_01.jpg` với **3 box `missing_hole`**. Mọi con số P/R/F1 bạn thấy
chỉ tính trên đúng 3 box đó (IoU >= 0.5), nên phân số chỉ có vài giá trị:

| Phase | TP | FP | FN | P = TP/(TP+FP) | R = TP/(TP+FN) | F1 |
|---|---|---|---|---|---|---|
| 01, 07, 08, 09 | 3 | 0 | 0 | 3/3 = **1.000** | 3/3 = **1.000** | **1.000** |
| 02–06 (SAHI) | 3 | 2 | 0 | 3/5 = **0.600** | 3/3 = **1.000** | **0.750** |

Muốn thấy số lẻ lẻ như báo khoa học thì phải chấm trên nhiều GT — xem mục 4.

---

## 1. Định nghĩa cơ bản (đọc 1 lần là hiểu)

- **GT (Ground Truth)**: hộp đáp án do người gán nhãn. Là cái đúng.
- **box dự đoán**: hộp model vẽ ra.
- **IoU** = diện tích giao nhau / diện tích hợp nhau của 2 hộp.
  IoU >= 0.5 → coi là trúng (quy ước chung cả 9 phase).
- **TP**: đoán đúng (đúng chỗ + đúng lớp + IoU >= 0.5). Càng nhiều càng tốt.
- **FP**: đoán bừa — vẽ chỗ không có lỗi hoặc sai lớp. Làm bẩn kết quả.
- **FN**: bỏ sót — có lỗi mà không vẽ. Nguy hiểm nhất ngoài thực tế.
- **P (Precision) = TP/(TP+FP)**: trong những gì vẽ ra, bao nhiêu % đúng.
  P thấp = nói dối nhiều.
- **R (Recall) = TP/(TP+FN)**: trong những lỗi thật, bắt được bao nhiêu %.
  R thấp = bỏ sót.
- **F1 = 2·P·R/(P+R)**: muốn cao phải vừa ít nói dối vừa ít bỏ sót.
- **ms/ảnh**: thời gian xử lý 1 ảnh (CPU).
- **forwards/ảnh**: chạy qua mạng Detector mấy lần cho 1 ảnh — đơn vị chi phí
  của TDD-Net (model CNN, không phải LLM nên không có token).
  Baseline = 1, SAHI ≈ 3–6 (mỗi slice 1 lần + ảnh full), TTA/Fusion = 2.

## 2. Kiến trúc lõi (chung cả 9 phase, không đổi)

```
Ảnh → Preprocess → ResNet-101 → FPN P2-P6 → RPN → ROI → FastRCNN → NMS/thr → box
```

| Block | Nó làm gì (1 câu) |
|---|---|
| Preprocess | Cạnh ngắn về 600, trừ màu nền PCB trung bình |
| ResNet-101 | "Mắt nhìn": 101 layer conv, 4 block (3+4+23+3 bottleneck), đường tắt residual giữ tín hiệu |
| FPN P2–P6 | "Kính lúp đa cỡ": stride 4→64, P2 soi lỗi 10px, P6 soi lỗi 200px |
| Anchor | Quăng sẵn 12 hộp mẫu/điểm (5 cỡ × 3 tỉ lệ × 4 hình), mạng chỉ việc nắn lại |
| RPN head | "Khoanh vùng nghi vấn": conv3×3/512 + nhánh có/không vật + nhánh nắn tọa độ, NMS 0.7 |
| ROI | "Cắt ra soi": box nhỏ soi ở P2 nét, box to soi ở P5/P6 |
| FastRCNN head | "Chốt án": FC1024×2 → lớp (7: 6 lỗi + nền) + nắn box (28 số) |
| Post + threshold | NMS theo lớp IoU 0.3, giữ box có score >= ngưỡng |

## 3. So sánh 9 phase (đọc 3 bước)

1. Nhìn **F1** trước (cao hơn = chính xác hơn).
2. F1 bằng nhau → nhìn **số box** (ít box hơn mà cùng F1 = gọn, ít rác).
3. Bằng nhau nữa → nhìn **ms/ảnh + forwards** (nhanh/rẻ hơn thắng).

| Phase | Đổi gì | Box | ms/ảnh | F1 demo |
|---|---|---|---|---|
| 01 Baseline | mốc, 1 forward | 32 | 1848 | 1.000 |
| 02 SAHI 512 | cắt slice + ảnh full | 59 (+27) | 9378 | 0.750 (−0.250, sinh FP) |
| 03–06 | siết thr/NMS, slice to, threshold theo lớp | 51–55 | 5293–9535 | 0.750 (không gỡ được FP) |
| 07 TTA flip | gốc + lật ngang, gộp NMS | 37 (−14) | 3884 | **1.000 (+0.250, FP 2→0)** |
| 08 Fusion | TTA + threshold theo lớp | 37 | 3491 | 1.000 (nhanh hơn 07) |
| 09 Fine-tune | train thật OHEM 300 step → `pcb_30300` | **29 (−8)** | 4001 | 1.000 (gọn nhất) |

## 4. Con số thật để báo cáo (không phải bảng demo)

Bảng demo chỉ để so công bằng. Năng lực thật của model là test trên
**100 ảnh thật đủ 6 lớp, 180 GT** (`output/phases/phase_09_finetuned/testset_eval.json`):

```
TP 173 / FP 9 / FN 7 → P = 0.951, R = 0.961, F1 = 0.956, ~3136 ms/ảnh CPU
```

Câu chốt cho thầy: *"F1 = 1.0 trên demo chỉ tính trên 3 GT của 1 lớp;
F1 = 0.956 (P = 0.951, R = 0.961) trên 100 ảnh thật đủ 6 lớp mới là
năng lực thật. SAHI tăng box nhưng giảm F1 do sinh FP; TTA lật ngang là
bước ngoặt (+0.25 F1, triệt tiêu FP); fine-tune OHEM 300 step cho model
gọn nhất (29 box) và chính xác nhất."*

## 5. Chạy bản v2 trên máy khác

1. Copy cả folder `TDD-Net-PCB-portable-v2` (cần ~2 GB, **không cần mạng**,
   không cần cài Python — đã kèm `python/` + `.venv/`).
   Nên giải nén ra đường dẫn không dấu, ví dụ `D:\TDD-Net-PCB-portable-v2`.
2. Nháy đúp **`setup_venv.bat`** (chạy 1 lần — tự viết lại checkpoint theo máy).
3. Nháy đúp **`start_ui.bat`**, chờ `Model ready. Open http://127.0.0.1:5000`.
4. Mở Chrome vào `http://127.0.0.1:5000`, kéo-thả ảnh PCB vào.
5. Kiểm tra model: mở `http://127.0.0.1:5000/api/health` → trường
   `checkpoint` phải là `pcb_30300model.ckpt` (model fine-tune Phase 09).
6. Bảng tracking + báo cáo đầy đủ nằm ở `output/phases/`
   (`BAO_CAO_DO_AN.md`, `cards/SO_SANH_TONG_HOP.md`, `TRACKING.md`).
