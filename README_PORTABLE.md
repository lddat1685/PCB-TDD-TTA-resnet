# TDD-Net PCB Defect Detection — bản đóng gói

Bản này chạy được trên Windows x64 mà **không cần cài Python, TensorFlow hay pip**.
Mọi thứ (Python 3.9 runtime + toàn bộ package) đã nằm trong folder.

## Yêu cầu

- Windows 10/11 **x64**
- ~3 GB dung lượng trống
- Microsoft Visual C++ Redistributable 2015-2022 (x64). Hầu hết máy đã có sẵn;
  nếu thiếu sẽ báo lỗi `VCRUNTIME140.dll` / `MSVCP140.dll` →
  https://aka.ms/vs/17/release/vc_redist.x64.exe
- Không cần GPU. Nếu có GPU NVIDIA thì vẫn chạy CPU (bản này không kèm cuDNN).

## Chạy

1. Giải nén **toàn bộ** folder (đừng chạy trực tiếp trong file .zip — Windows
   sẽ giải nén từng file ra temp và không tìm thấy `python/`).
2. Double-click **`start_ui.bat`**.
3. Chờ ~40 giây (lần đầu build graph + nạp trọng số).
4. Mở trình duyệt: **http://127.0.0.1:5000**

Nếu Windows SmartScreen chặn, chọn *More info → Run anyway*.

## Dùng UI

| Thao tác | Cách dùng |
|---|---|
| Thêm ảnh | Kéo-thả vào ô bên trái, hoặc bấm để chọn (chọn được nhiều ảnh) |
| Đổi ảnh | Bấm thumbnail |
| Ngưỡng tin cậy | Kéo slider "Score ≥". Kéo xong tự chạy lại, ~0.5s |
| Ẩn/hiện loại lỗi | Tick/bỏ tick ở mục "Loại lỗi" |
| Xem chi tiết | Rê chuột lên box, hoặc lên dòng trong bảng kết quả |
| Lưu ảnh | Bấm "Tải ảnh kết quả" (chỉ vẽ các box đang hiện) |

6 loại lỗi: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`,
`spurious_copper`.

## Chạy bằng dòng lệnh

```bat
python\python.exe launcher.py
```

Hoặc dùng API trực tiếp:

```bat
curl -X POST -F "image=@duong\dan\anh.jpg" -F "threshold=0.6" http://127.0.0.1:5000/api/detect
```

## Cấu trúc

```
start_ui.bat / start_ui.sh   điểm khởi chạy
launcher.py                  nạp sys.path rồi chạy ui/app.py
python/                      Python 3.9 runtime (đã lược bớt, 113 MB)
.venv/Lib/site-packages/     TensorFlow 2.5.1 + Flask + OpenCV + ... (2.1 GB)
output/trained_weights/      trọng số TDD-Net (467 MB)
ui/                          Flask app + giao diện
tf1_compat.py                lớp tương thích TF1 -> TF2
input/                       ảnh mẫu
libs/, data/, tools/         code gốc của repo
```

## Vì sao có `launcher.py` và `python/`

`python.exe` bên trong `.venv/Scripts/` là một *redirector* trỏ tới đường dẫn
tuyệt đối của máy đã build venv. Copy sang máy khác nó sẽ không chạy.
`launcher.py` thay vào đó dùng Python runtime đi kèm và thêm thẳng
`.venv/Lib/site-packages` vào `sys.path` — không cần cài lại gì.

## Đã kiểm chứng

- `m13.jpg` @ threshold 0.6 → 5 × `missing_hole` + 1 × `short`
- `s14.jpg` @ 0.8 → 9 × `short`
- Kết quả trùng khớp với đường CLI gốc (`tools/inference.py`), cùng score tới 4 chữ số

## Xử lý sự cố

| Triệu chứng | Nguyên nhân / cách sửa |
|---|---|
| `Khong tim thay python\python.exe` | Chưa giải nén hết, hoặc chạy trong file zip |
| `VCRUNTIME140.dll` not found | Cài VC++ Redistributable (link ở trên) |
| `Port 5000 already in use` | Đổi port trong `ui/app.py` (dòng `port=5000`) hoặc tắt app đang chiếm |
| Trình duyệt báo không kết nối được | Server chưa build xong; xem cửa sổ console, chờ tới dòng `Model ready` |
| Ảnh rất lớn bị chậm | Model resize cạnh ngắn về 600px; ảnh gốc > 3000px bị giới hạn bởi `IMG_MAX_LENGTH` |

## Giới hạn

- Chỉ **inference**. `train.py` / `eval.py` không chạy được: cần Cython
  (`nms.pyx`) và compiler C, cả hai đã bỏ khỏi bản đóng gói.
- Chạy **CPU only**. Muốn dùng GPU: cài cuDNN 8.1 (`cudnn64_8.dll`) khớp CUDA 11.2.
- Trọng số huấn luyện trên bộ PCB-Defect (Peking University), 6 loại lỗi.
