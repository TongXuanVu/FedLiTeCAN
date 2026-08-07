# Chạy FedLiTeCAN ở máy — hướng dẫn từng bước

> Không cần Kaggle. Toàn bộ dữ liệu đã có sẵn ở `C:\FederatedLearning\AFSIC-IOV\data\10client`.

---

## 0. Kiểm tra môi trường trước

```bat
cd C:\FederatedLearning\FedLiTeCAN
python -c "import torch, flwr, sklearn; print('torch', torch.__version__, '| GPU:', torch.cuda.is_available()); print('flwr', flwr.__version__)"
```

Nếu thiếu: `pip install torch flwr scikit-learn`.
Không có GPU vẫn chạy được, chỉ chậm hơn — xem mục 3 để chọn mức cắt trần.

---

## 1. Chạy thử 2–3 phút trước khi chạy thật

Bắt lỗi đường dẫn / thiếu thư viện / cổng bị chiếm, mà không tốn hàng giờ:

```bat
cd C:\FederatedLearning\FedLiTeCAN
python tools\run_task_incremental.py --smoke ^
    --data-dir "C:\FederatedLearning\AFSIC-IOV\data\10client\federated_data" ^
    --test-file "C:\FederatedLearning\AFSIC-IOV\data\10client\global_test_data.pt"
```

Chế độ này chạy **1 round mỗi stage, 20.000 mẫu/client, test 50.000 mẫu**. Kết quả
**không dùng để báo cáo** — chỉ để xác nhận đường ống chạy được.

Thấy đủ 5 dòng `STAGE 0..4` và cuối cùng là `Xong sau ... phút` thì sang bước 2.

---

## 2. Chạy thật

```bat
python tools\run_task_incremental.py ^
    --data-dir "C:\FederatedLearning\AFSIC-IOV\data\10client\federated_data" ^
    --test-file "C:\FederatedLearning\AFSIC-IOV\data\10client\global_test_data.pt" ^
    --out logs\10client_taskIL ^
    --max-samples 500000 --test-max-samples 1000000
```

Script tự làm hết: dò client nào có dữ liệu ở stage nào, khởi động server + đúng
những client đó, chờ xong 30 round, rồi nối checkpoint sang stage sau.

---

## 3. Chọn mức cắt trần theo phần cứng

Chỉ **stage 0** là nặng — các stage sau dữ liệu rất nhỏ:

| Stage | Không cắt | Cắt 500k | Cắt 100k | #client |
|---|---|---|---|---|
| 0 | 97.668.978 | **2.500.000** | 500.000 | 5 |
| 1 | 265.083 | 265.083 | 265.083 | 5 |
| 2 | 36.540 | 36.540 | 36.540 | 6 |
| 3 | 18.658 | 18.658 | 18.658 | 8 |
| 4 | 124.334 | 124.334 | 124.334 | 7 |

Số bước gradient cho trọn 150 round (batch 128, 60% dữ liệu dùng để train):

| `--max-samples` | Số bước | Máy có GPU | Máy chỉ có CPU |
|---|---|---|---|
| 500.000 | 414.540 | ~1,5–3 h | ~8–15 h |
| **100.000** | **133.290** | **~30–60 phút** | **~3–5 h** |

Nếu máy không có GPU, dùng `--max-samples 100000 --test-max-samples 200000`. Ghi rõ
con số này trong bài — cắt trần là lựa chọn thiết kế phải khai báo, không phải giấu.

Cách khác để nhanh hơn mà không giảm dữ liệu: `--batch-size 1024`. Nhưng batch đổi
thì tốc độ hội tụ đổi, nên nếu muốn so với con số 76,73 của lần chạy trước thì phải
giữ `--batch-size 128`.

---

## 4. Nếu bị lỗi

| Triệu chứng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Server chet ngay khi khoi dong` | cổng 8081 đang bị chiếm | `--address 127.0.0.1:8082` |
| Client treo không kết nối | server chưa nạp xong tập test | tăng `time.sleep(20)` trong script |
| `MemoryError` ở client | `--max-samples` quá lớn | hạ xuống 100000 |
| `MemoryError` ở server | tập test quá lớn | hạ `--test-max-samples` xuống 200000 |
| `KHONG TIM THAY` | sai đường dẫn | kiểm tra lại hai đường dẫn ở mục 2 |

Log của từng stage nằm ở `logs\10client_taskIL\task_<t>\logs\server.log` và
`client_<i>.log`.

---

## 5. Cần thấy gì trong log

1. `Task 0: loc test ve lop 0-2 -> n=...` — server đã lọc đúng lớp đã học
2. `Task 0: 3 lop | lop da so chiem 99.62% | NGUONG SUP macro-F1 = 33.27%` — trùng
   với ngưỡng của AFSIC-IoV, tức so sánh được
3. `Client 0: loaded ...client_0_task_1.pt` — đúng file của task, **không phải**
   `client_0.pt` gộp
4. Round 0 accuracy ≈ tỉ trọng lớp đa số của tập test đã lọc
5. Ở stage 1 trở đi: `Loaded checkpoint ... (round 30)` rồi `Resume: chay tiep 30 rounds`

---

## 6. Kết quả nằm ở đâu

```
logs\10client_taskIL\
    metrics_iov.csv          <- 150 round liền mạch, đây là file chính
    checkpoints_iov\         <- round_001.pth .. round_150.pth
    task_0\logs\             <- log server + từng client của stage 0
    task_1\logs\
    ...
```

`metrics_iov.csv` được **ghi tiếp** qua cả 5 stage nên là một mạch 150 round, đọc
thẳng được. Round cuối mỗi task và ngưỡng tương ứng:

| Round | Hết task | Số lớp | Ngưỡng sụp macro-F1 |
|---|---|---|---|
| 30 | 0 | 3 | 33,27 |
| 60 | 1 | 6 | 16,61 |
| 90 | 2 | 9 | 11,07 |
| 120 | 3 | 11 | 9,06 |
| 150 | 4 | 13 | 7,66 |

Giá trị trong file để **thang 0–1**; nhân 100 khi đưa vào bảng tổng hợp (xem
`Tổng hợp kết quả/iov10/fedlitecan_iov_31_rounds.csv` để biết định dạng chuẩn).

---

## 7. Dự đoán đặt trước

FedLiTeCAN không có replay lẫn KD, nên sau mỗi stage trọng số cũ bị ghi đè và nó sẽ
quên các lớp đã học. Dự đoán:

| Round | macro-F1 dự đoán | Ngưỡng |
|---|---|---|
| 30 (task 0) | **trên ngưỡng** — chỉ 3 lớp, học được | 33,27 |
| 60 (task 1) | quanh hoặc dưới ngưỡng | 16,61 |
| 90–150 | dưới ngưỡng | 11,07 → 7,66 |

Nếu nó **không** quên thì phải kiểm tra lại — mô hình không có cơ chế nào giữ lớp cũ
thì không có lý do gì giữ được.
