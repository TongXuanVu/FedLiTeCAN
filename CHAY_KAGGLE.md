# Chạy FedLiTeCAN task-incremental trên Kaggle

> 5 stage × 30 round = 150 round, cùng quy ước với AFSIC-IoV nên ngưỡng sụp trùng
> khít từng task. Gọn trong một session 12 giờ.

---

## Cell 1 — chuẩn bị

```python
!pip install -q flwr

!rm -rf /kaggle/working/FedLiTeCAN
!git clone -q https://github.com/TongXuanVu/FedLiTeCAN.git /kaggle/working/FedLiTeCAN
%cd /kaggle/working/FedLiTeCAN

# Tu do dataset IoV trong /kaggle/input
import glob, os
tf = glob.glob('/kaggle/input/**/global_test_data.pt', recursive=True)
df = glob.glob('/kaggle/input/**/federated_data/client_*_task_*.pt', recursive=True)
assert tf, 'Khong tim thay global_test_data.pt trong /kaggle/input'
assert df, 'Khong tim thay federated_data/client_*_task_*.pt'
TEST_FILE = tf[0]
DATA_DIR  = os.path.dirname(df[0])
print('TEST_FILE =', TEST_FILE)
print('DATA_DIR  =', DATA_DIR)

# CANH BAO: phai la du lieu IoV (31 dac trung, 13 lop), KHONG phai IoT (33 dac trung)
import torch
_b = torch.load(sorted(glob.glob(DATA_DIR + '/client_0_task_*.pt'))[0],
                map_location='cpu', weights_only=False)
print('So dac trung =', _b['x'].shape[1], '(phai la 31)')
assert _b['x'].shape[1] == 31, 'SAI DATASET — day khong phai IoV'
del _b
```

## Cell 2 — chạy thử 2–3 phút

```python
!python tools/run_task_incremental.py --smoke \
    --data-dir "{DATA_DIR}" --test-file "{TEST_FILE}"
```

Phải thấy đủ 5 dòng `STAGE 0` … `STAGE 4` rồi `Xong sau ... phut`. Nếu dừng giữa
chừng, đọc `logs/10client_taskIL_smoke/task_<t>/logs/server.log`.

## Cell 3 — chạy thật

```python
!python tools/run_task_incremental.py \
    --data-dir "{DATA_DIR}" --test-file "{TEST_FILE}" \
    --out /kaggle/working/logs_taskIL \
    --max-samples 500000 --test-max-samples 1000000 \
    --rounds-per-task 30 --batch-size 128
```

## Cell 4 — xem kết quả ngay

```python
import pandas as pd
d = pd.read_csv('/kaggle/working/logs_taskIL/metrics_iov.csv')
NG = {30: 33.27, 60: 16.61, 90: 11.07, 120: 9.06, 150: 7.66}
for r, ng in NG.items():
    row = d[d['round'] == r]
    if row.empty:
        print(f'  round {r:>3}: chua chay toi'); continue
    f1 = row['macro_f1'].iloc[0] * 100
    print(f'  round {r:>3} (het task {r//30-1}): acc {row["accuracy"].iloc[0]*100:6.2f} | '
          f'macro-F1 {f1:6.2f} | nguong {ng:5.2f} | '
          f'{"TREN nguong" if f1 > ng else "duoi nguong"}')
```

## Cell 5 — tải kết quả về

```python
import shutil
shutil.make_archive('/kaggle/working/fedlitecan_taskIL', 'zip',
                    '/kaggle/working/logs_taskIL')
print('Tai file fedlitecan_taskIL.zip o tab Output')
```

---

## Chi phí ước tính

Chỉ **stage 0** nặng, bốn stage sau dữ liệu rất nhỏ:

| Stage | Số mẫu sau khi cắt 500k | #client |
|---|---|---|
| 0 | 2.500.000 | 5 |
| 1 | 265.083 | 5 |
| 2 | 36.540 | 6 |
| 3 | 18.658 | 8 |
| 4 | 124.334 | 7 |

Tổng 414.540 bước gradient cho trọn 150 round. Trên GPU Kaggle (T4/P100):
**khoảng 1,5–3 giờ**, kể cả 150 lần đánh giá trên 1 triệu mẫu test.

Thoải mái trong một session 12 giờ, nên **không cần resume giữa chừng**.

---

## Cần thấy gì trong log

1. `So dac trung = 31` ở Cell 1 — đúng dataset IoV, không phải IoT
2. `Client 0: loaded ...client_0_task_1.pt` — **file theo task**, không phải
   `client_0.pt` gộp. Nếu thấy file gộp thì cờ `--task` chưa vào và cả lần chạy
   vô nghĩa.
3. `Task 0: loc test ve lop 0-2 -> n=...`
4. `Task 0: 3 lop | lop da so chiem 99.62% | NGUONG SUP macro-F1 = 33.27%` — trùng
   ngưỡng của AFSIC-IoV, tức so sánh được
5. Từ stage 1: `Loaded checkpoint ... (round 30)` rồi `Resume: chay tiep 30 rounds`

---

## Dự đoán đặt trước

FedLiTeCAN không có replay lẫn KD nên sau mỗi stage trọng số cũ bị ghi đè:

| Round | Hết task | macro-F1 dự đoán | Ngưỡng |
|---|---|---|---|
| 30 | 0 | **trên ngưỡng** — chỉ 3 lớp | 33,27 |
| 60 | 1 | quanh hoặc dưới ngưỡng | 16,61 |
| 90 | 2 | dưới ngưỡng | 11,07 |
| 120 | 3 | dưới ngưỡng | 9,06 |
| 150 | 4 | dưới ngưỡng | 7,66 |

Nếu nó **không** quên thì phải kiểm tra lại — mô hình không có cơ chế nào giữ lớp cũ
thì không có lý do gì giữ được.

---

## Lưu ý

`metrics_iov.csv` được ghi tiếp qua cả 5 stage nên ra **một mạch 150 round**. Giá trị
để **thang 0–1**, nhân 100 khi đưa vào `Tổng hợp kết quả/iov10`.

`.gitignore` của repo chặn `*.pth`, `checkpoints_iov/`, `metrics_iov.csv`. Nếu muốn
đẩy kết quả lên GitHub phải dùng `git add -f`, và **luôn kiểm tra bằng `git ls-files`
sau khi push**.
