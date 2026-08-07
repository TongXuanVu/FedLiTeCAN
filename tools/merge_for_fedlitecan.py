"""Gộp dữ liệu AFSIC-IOV (chia theo task) thành định dạng FedLiTeCAN (một file/client).

AFSIC-IOV : data/10client/federated_data/client_<i>_task_<t>.pt   (t = 1..5)
FedLiTeCAN: <out>/client_<i>.pt                                    (gộp mọi task)

Cả hai đều là dict {'x': (N,31) float16, 'y': (N,) int64}.

    python tools/merge_for_fedlitecan.py \
        --src "C:/FederatedLearning/AFSIC-IOV/data/10client/federated_data" \
        --out "C:/FederatedLearning/FedLiTeCAN/data/10client"

Kiểm chứng: số mẫu sau khi gộp phải khớp với log của lần chạy trước —
client 0 = 29.304.512, client 5 = 87.690, client 6 = 7.208, client 7 = 4.414,
client 8 = 16.675, client 9 = 31.400.
"""
import argparse
import glob
import os
import re
from collections import Counter

import torch

# Số mẫu mong đợi, lấy từ log lần chạy FedLiTeCAN trước — dùng để tự kiểm tra.
EXPECTED = {0: 29_304_512, 1: 29_392_930, 2: 29_423_906, 3: 6_072_832,
            4: 3_772_026, 5: 87_690, 6: 7_208, 7: 4_414, 8: 16_675, 9: 31_400}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='thu muc federated_data cua AFSIC-IOV')
    ap.add_argument('--out', required=True, help='thu muc dich cho FedLiTeCAN')
    ap.add_argument('--clients', type=int, default=10)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    total = 0
    for cid in range(a.clients):
        files = sorted(glob.glob(os.path.join(a.src, f'client_{cid}_task_*.pt')),
                       key=lambda p: int(re.search(r'task_(\d+)', p).group(1)))
        if not files:
            print(f'  client {cid}: khong co file nao — bo qua')
            continue

        xs, ys = [], []
        for f in files:
            d = torch.load(f, map_location='cpu', weights_only=False)
            xs.append(d['x'])
            ys.append(d['y'])
            del d

        x = torch.cat(xs)
        y = torch.cat(ys)
        del xs, ys

        dst = os.path.join(a.out, f'client_{cid}.pt')
        torch.save({'x': x, 'y': y}, dst)

        n = len(y)
        total += n
        exp = EXPECTED.get(cid)
        ok = '' if exp is None else ('  OK' if n == exp else f'  !! LECH, mong doi {exp:,}')
        cls = dict(sorted(Counter(y.tolist()).items()))
        print(f'  client {cid}: {len(files)} file task -> {n:>12,} mau, {len(cls)} lop{ok}')
        print(f'             {cls}')
        del x, y

    print(f'\n  TONG: {total:,} mau (mong doi 98.113.593)')


if __name__ == '__main__':
    main()
