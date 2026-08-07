# FedLiTeCAN — kế hoạch chạy lại trên dữ liệu AFSIC-IOV

> 2026-08-06. Mục đích: tách riêng ảnh hưởng của **việc cắt trần lớp đa số** khỏi
> ảnh hưởng của **bộ phân loại**, dùng đúng dữ liệu mà AFSIC-IoV đang chạy.

---

## 1. Dữ liệu — thực ra là cùng một bộ

Lần chạy FedLiTeCAN trước dùng dataset `ids-iov` trên Kaggle. Đối chiếu số mẫu với
`AFSIC-IOV/data/10client/allocation_plan.csv`:

| Client | FedLiTeCAN (log cũ) | AFSIC-IOV (gộp 5 task) |
|---|---|---|
| 0 | 29.304.512 | **29.304.512** |
| 5 | 87.690 | **87.690** |
| 6 | 7.208 | **7.208** |
| 7 | 4.414 | **4.414** |
| 8 | 16.675 | **16.675** |
| 9 | 31.400 | **31.400** |

Khớp từng con số. Khác biệt duy nhất là **cách tổ chức file**:

```
AFSIC-IOV : federated_data/client_<i>_task_<t>.pt   (t = 1..5)
FedLiTeCAN: <data-dir>/client_<i>.pt                 (gộp mọi task)
```

Nên chỉ cần gộp lại, không cần chia lại dữ liệu:

```bash
python tools/merge_for_fedlitecan.py \
    --src "C:/FederatedLearning/AFSIC-IOV/data/10client/federated_data" \
    --out "C:/FederatedLearning/FedLiTeCAN/data/10client"
```

Script tự đối chiếu với bảng trên và báo `OK` hoặc `!! LECH` cho từng client.

---

## 2. Điều quan trọng phát hiện khi đọc lại code

**`subsample_capped` không hề đưa về cân bằng.** Nó chia quota đều, lớp nào ít hơn
quota thì giữ hết, phần dư dồn cho lớp lớn. Với client 0 và `--max-samples 500000`:

| | Benign trong tập train của client 0 |
|---|---|
| không cắt | 99,61% |
| cắt 500.000 | **77,18%** |

Tức bản "đã cắt" vẫn còn rất lệch. **Chỗ tạo khác biệt lớn là tập TEST**, nơi Benign
bị kéo từ 99,17% xuống 65,19%.

**Hàm mất mát cũng khác AFSIC-IoV.** FedLiTeCAN dùng Focal loss với
`alpha = sqrt(total/count)`, AFSIC-IoV dùng CE với `w = total/(count·C)` — tức nghịch
đảo tuyến tính. Với lớp hiếm nhất:

| | Trọng số so với Benign |
|---|---|
| FedLiTeCAN (căn bậc hai) | **136 : 1** |
| AFSIC-IoV (tuyến tính) | **18.474 : 1** |

Đây là biến thứ ba, cần nhớ khi diễn giải kết quả.

---

## 3. ⭐ Chế độ task-incremental — theo góp ý của thầy

> "Đúng rồi nhưng vẫn bắt tụi nó chạy trong điều kiện data task luôn, dị mới nói nó
> yếu vì không hỗ trợ IL chớ."

Góp ý này đúng và quan trọng. Nếu gộp hết 5 task rồi huấn luyện chung thì FedLiTeCAN
được lợi thế **joint training** — nó thấy mọi lớp cùng lúc. Lúc đó con số 76,73 không
chứng minh được gì về khả năng IL, và cũng không thể dùng để nói nó yếu.

Cách đúng: chạy **tuần tự theo task**, mỗi stage client chỉ thấy dữ liệu của task đó.
FedLiTeCAN không có replay, không có KD, nên **quên các lớp cũ là kết quả cần đo** —
đó chính là bằng chứng cho việc không hỗ trợ IL.

```bash
python tools/run_task_incremental.py \
    --data-dir "C:/FederatedLearning/AFSIC-IOV/data/10client/federated_data" \
    --test-file "C:/FederatedLearning/AFSIC-IOV/data/10client/global_test_data.pt" \
    --out logs/10client_taskIL --max-samples 500000
```

Script tự dò client nào có dữ liệu ở stage nào, tự nối checkpoint giữa các stage:

| Stage | Lớp | Client tham gia | Ngưỡng sụp |
|---|---|---|---|
| 0 | 0–2 | 0, 1, 2, 3, 4 | 33,27 |
| 1 | 0–5 | 0, 1, 2, 3, 5 | 16,61 |
| 2 | 0–8 | 0, 1, 2, 3, 5, 6 | 11,07 |
| 3 | 0–10 | 0, 1, 2, 3, 4, 5, 6, 7 | 9,06 |
| 4 | 0–12 | 0, 1, 2, 3, 5, 8, 9 | 7,66 |

5 stage × 30 round = **150 round, đúng bằng AFSIC-IoV**. Server đánh giá trên tập
test lọc theo lớp đã học, cùng quy ước với AFSIC-IoV nên **ngưỡng sụp giống hệt** —
so sánh được trực tiếp từng task.

**Dự đoán:** FedLiTeCAN sẽ quên sạch lớp cũ sau mỗi stage, macro-F1 rơi xuống quanh
hoặc dưới ngưỡng từ task 1. Đó là kết quả mong đợi và là luận điểm cần cho bài.

---

## 4. Ba lần chạy còn lại — nhưng chỉ tốn công cho hai

| | Train | Test | Mục đích | Chi phí |
|---|---|---|---|---|
| **A** | gộp hết task, cắt 500k/client | cắt 1 triệu | tái lập baseline 76,73 (cận trên joint training) | ~1–2 h |
| **B** | gộp hết task, không cắt (98,1 M) | không cắt (42 M) | điều kiện giống hệt AFSIC-IoV | ~4–8 h |
| **C** | *dùng lại checkpoint của A* | không cắt (42 M) | tách riêng ảnh hưởng của tập test | **~10 phút** |

C **không cần huấn luyện** — chỉ đánh giá lại `round_030.pth` của A trên tập test đầy
đủ. Đây là phép đo rẻ nhất mà lại trả lời trúng câu hỏi.

### Cách đọc kết quả

| A | C | B | Kết luận |
|---|---|---|---|
| ~77 | **sụp** | sụp | Tập test lệch là nguyên nhân chính |
| ~77 | ~77 | **sụp** | Cắt trần khi *huấn luyện* mới là nguyên nhân |
| ~77 | ~77 | ~77 | Cả hai đều không phải — bộ phân loại prototype của AFSIC-IoV mới là vấn đề |

Trường hợp thứ ba là **kết quả mạnh nhất** cho bài viết: nó chứng minh cùng dữ liệu,
cùng độ lệch, cùng 10 client, chỉ đổi bộ phân loại là được 77 thay vì 0,03.

---

## 5. Lệnh chạy

### A — tái lập baseline

```bat
python server_iov.py --mode train --rounds 30 --local-epochs 1 ^
    --test-file "C:\FederatedLearning\AFSIC-IOV\data\10client\global_test_data.pt" ^
    --test-max-samples 1000000

REM 10 cua so client, moi cai:
python client_iov.py --client-id %i% ^
    --data-dir "C:\FederatedLearning\FedLiTeCAN\data\10client" ^
    --max-samples 500000 --batch-size 128
```

Hoặc dùng `run_all.bat 30 1` sau khi sửa `DEFAULT_DATA_DIR` và `DEFAULT_TEST`.

### C — đánh giá lại checkpoint của A trên tập test đầy đủ (chạy ngay sau A)

```bat
python server_iov.py --mode test --checkpoint logs\10client_A\checkpoints_iov\round_030.pth ^
    --test-file "C:\FederatedLearning\AFSIC-IOV\data\10client\global_test_data.pt" ^
    --test-max-samples 0
```

### B — không cắt trần

```bat
python server_iov.py --mode train --rounds 30 --local-epochs 1 ^
    --test-file "C:\FederatedLearning\AFSIC-IOV\data\10client\global_test_data.pt" ^
    --test-max-samples 0

python client_iov.py --client-id %i% ^
    --data-dir "C:\FederatedLearning\FedLiTeCAN\data\10client" ^
    --max-samples 0 --batch-size 128
```

---

## 6. Đã sửa gì trong code

Bản gốc sẽ **hết RAM** khi `--max-samples 0`. Ba chỗ, đều **không đổi kết quả tính
toán**, chỉ đổi cách dùng bộ nhớ:

| File | Vấn đề | Sửa |
|---|---|---|
| `client_iov.py` | ép `float32` **trước** khi subsample → client 29 M mẫu tốn 3,6 GB, rồi `train_test_split` nhân thêm 3 bản | ép `float32` **sau** khi subsample; nếu vẫn trên 5 triệu mẫu thì giữ `float16` và ép theo từng batch |
| `client_iov.py` / `server_iov.py` | `list.extend()` gom dự đoán → list Python 42 triệu phần tử (~3 GB mỗi list) | gom vào list mảng numpy `int16` rồi `np.concatenate` một lần |
| `server_iov.py` | tập test ép `float32` → 5,2 GB thay vì 2,6 GB | giữ `float16` khi `max_samples = 0`, ép theo batch |

**Nếu vẫn thiếu RAM** (máy dưới 24 GB): dùng `--max-samples 5000000` thay cho `0`.
Client 0 khi đó vẫn có Benign chiếm **97,7%** — gần như nguyên vẹn độ lệch — nhưng bộ
nhớ giảm 6 lần. Ghi rõ con số này trong bài nếu dùng.

---

## 7. Cần thấy gì trong log

1. `Client 0: loaded ... x=(29304512, 31)` — đúng dữ liệu, đúng client
2. `Client 0: after subsample n=...` — 500000 (A) hoặc 29304512 (B)
3. `Client 0: dtype dac trung = float16` ở B, `float32` ở A
4. `Evaluating each round on n=...` — 1000000 (A) hoặc 42048683 (B/C)
5. Round 0 accuracy phải **đúng bằng tỉ trọng Benign của tập test**: 0,6519 ở A,
   0,9917 ở B/C. Nếu lệch thì mô hình không đoán thuần Benign ở round 0 — kiểm tra lại.

---

## 8. Ngưỡng sụp tương ứng

macro-F1 mà bộ chỉ đoán Benign đạt được, 13 lớp:

| Tập test | Benign | Ngưỡng sụp |
|---|---|---|
| A — cắt 1 triệu | 65,19% | **6,07%** |
| B, C — đầy đủ 42 triệu | 99,17% | **7,66%** |

Con số 6,07 kiểm chứng được: round 0 của lần chạy trước có macro-F1 đúng **6,07** và
accuracy đúng **0,651909**.

---

## 9. Ghi vào đâu

Sau khi chạy xong, thêm vào `Tổng hợp kết quả/iov10/`:

- `fedlitecan_iov_A_31_rounds.csv` — bản cắt trần (thay cho file hiện tại)
- `fedlitecan_iov_B_31_rounds.csv` — bản không cắt
- một dòng cho C trong `so_sanh_phuong_phap.csv`

và cập nhật mục 5 của `README.md`.
