"""Chạy FedLiTeCAN theo đúng điều kiện task-incremental của AFSIC-IoV.

Vì sao cần: FedLiTeCAN **không có cơ chế IL** (không replay, không KD). Nếu gộp hết
các task rồi huấn luyện chung thì nó được lợi thế joint training, và không thể dùng
làm bằng chứng "phương pháp không hỗ trợ IL thì yếu". Chạy tuần tự theo task mới
đúng: mỗi stage client chỉ thấy dữ liệu của task đó, quên các lớp cũ là kết quả cần
đo, không phải lỗi.

5 stage × 30 round = 150 round, đúng bằng AFSIC-IoV. Server đánh giá trên tập test
lọc theo các lớp đã học (0-2, 0-5, 0-8, 0-10, 0-12) — cùng quy ước với AFSIC-IoV nên
ngưỡng sụp giống hệt: 33.27 / 16.61 / 11.07 / 9.06 / 7.66.

CHẠY THỬ TRƯỚC (khoảng 2-3 phút, để bắt lỗi cấu hình):

    python tools/run_task_incremental.py --smoke ^
        --data-dir "C:/FederatedLearning/AFSIC-IOV/data/10client/federated_data" ^
        --test-file "C:/FederatedLearning/AFSIC-IOV/data/10client/global_test_data.pt"

CHẠY THẬT:

    python tools/run_task_incremental.py ^
        --data-dir "C:/FederatedLearning/AFSIC-IOV/data/10client/federated_data" ^
        --test-file "C:/FederatedLearning/AFSIC-IOV/data/10client/global_test_data.pt" ^
        --out logs/10client_taskIL --max-samples 500000
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

TASK_INCREMENTS = [3, 3, 3, 2, 2]
NGUONG = [33.27, 16.61, 11.07, 9.06, 7.66]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, 'checkpoints_iov')
CSV_FILE = os.path.join(ROOT, 'metrics_iov.csv')


def clients_of(data_dir, task):
    """Client nao co file du lieu cho task nay."""
    return sorted(int(re.search(r'client_(\d+)_task', os.path.basename(f)).group(1))
                  for f in glob.glob(os.path.join(data_dir, f'client_*_task_{task + 1}.pt')))


def archive_previous(out):
    """Don artifact cua lan chay truoc de khong tron lan."""
    stamp = datetime.now().strftime('%d-%m-%y_%H-%M')
    for p in (CKPT_DIR, CSV_FILE):
        if os.path.exists(p):
            dst = os.path.join(out, f'_cu_{stamp}')
            os.makedirs(dst, exist_ok=True)
            shutil.move(p, os.path.join(dst, os.path.basename(p)))
            print(f'  da chuyen {os.path.basename(p)} cu -> {dst}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--test-file', required=True)
    ap.add_argument('--out', default='logs/10client_taskIL')
    ap.add_argument('--rounds-per-task', type=int, default=30)
    ap.add_argument('--local-epochs', type=int, default=1)
    ap.add_argument('--max-samples', type=int, default=500_000)
    ap.add_argument('--test-max-samples', type=int, default=1_000_000)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--address', default='127.0.0.1:8081')
    ap.add_argument('--smoke', action='store_true',
                    help='Chay thu: 1 round/stage, 20.000 mau/client, test 50.000. '
                         'Chi de kiem tra duong ong, KHONG dung lam ket qua.')
    a = ap.parse_args()

    if a.smoke:
        a.rounds_per_task, a.max_samples, a.test_max_samples = 1, 20_000, 50_000
        a.out = a.out.rstrip('/') + '_smoke'

    out = a.out if os.path.isabs(a.out) else os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    py = sys.executable

    print(f'\nThu muc ket qua : {out}')
    print(f'Du lieu         : {a.data_dir}')
    print(f'Tap test        : {a.test_file}')
    print(f'Cat tran        : {a.max_samples:,} mau/client | test {a.test_max_samples:,}')
    print(f'{"CHE DO CHAY THU — ket qua khong dung de bao cao" if a.smoke else ""}\n')

    for f in (a.data_dir, a.test_file):
        if not os.path.exists(f):
            sys.exit(f'KHONG TIM THAY: {f}')

    archive_previous(out)
    prev_ckpt = None
    t0_all = time.time()

    for t in range(len(TASK_INCREMENTS)):
        n_cls = sum(TASK_INCREMENTS[:t + 1])
        cl = clients_of(a.data_dir, t)
        if not cl:
            print(f'  Stage {t}: khong client nao co du lieu — bo qua')
            continue
        log_dir = os.path.join(out, f'task_{t}', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        print(f'\n{"=" * 72}')
        print(f'  STAGE {t}  |  lop 0-{n_cls - 1}  |  {len(cl)} client: {cl}')
        print(f'  nguong sup macro-F1 = {NGUONG[t]}  |  round '
              f'{t * a.rounds_per_task + 1}-{(t + 1) * a.rounds_per_task}')
        print(f'{"=" * 72}')
        t0 = time.time()

        cmd = [py, os.path.join(ROOT, 'server_iov.py'),
               '--rounds', str((t + 1) * a.rounds_per_task),
               '--local-epochs', str(a.local_epochs),
               '--num-clients', str(len(cl)),
               '--address', a.address,
               '--test-file', a.test_file,
               '--test-max-samples', str(a.test_max_samples),
               '--task', str(t)]
        cmd += (['--mode', 'resume', '--checkpoint', prev_ckpt] if prev_ckpt
                else ['--mode', 'train'])

        srv_log = open(os.path.join(log_dir, 'server.log'), 'w')
        srv = subprocess.Popen(cmd, cwd=ROOT, stdout=srv_log, stderr=subprocess.STDOUT)
        time.sleep(20)          # cho server nap tap test va mo cong
        if srv.poll() is not None:
            srv_log.close()
            sys.exit(f'Server chet ngay khi khoi dong. Xem {log_dir}/server.log')

        procs, fhs = [], []
        for cid in cl:
            fh = open(os.path.join(log_dir, f'client_{cid}.log'), 'w')
            fhs.append(fh)
            procs.append(subprocess.Popen(
                [py, os.path.join(ROOT, 'client_iov.py'),
                 '--client-id', str(cid), '--data-dir', a.data_dir,
                 '--server-address', a.address,
                 '--max-samples', str(a.max_samples),
                 '--batch-size', str(a.batch_size), '--task', str(t)],
                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT))
            time.sleep(2)

        srv.wait()
        for p in procs:
            p.wait()
        srv_log.close()
        for fh in fhs:
            fh.close()

        cks = sorted(glob.glob(os.path.join(CKPT_DIR, 'round_*.pth')))
        if not cks:
            sys.exit(f'Stage {t} khong sinh checkpoint nao. Xem {log_dir}/server.log')
        prev_ckpt = cks[-1]
        print(f'  Stage {t} xong sau {(time.time() - t0) / 60:.1f} phut. '
              f'Checkpoint cuoi: {os.path.basename(prev_ckpt)}')

    # metrics_iov.csv duoc GHI TIEP qua ca 5 stage nen la mot mach 150 round
    for p in (CSV_FILE, CKPT_DIR):
        if os.path.exists(p):
            dst = os.path.join(out, os.path.basename(p))
            shutil.move(p, dst)
            print(f'  -> {dst}')

    print(f'\nXong sau {(time.time() - t0_all) / 60:.1f} phut.')
    print('Doc metrics_iov.csv (150 round lien mach) va doi chieu macro_f1 voi nguong:')
    for t, ng in enumerate(NGUONG):
        r = (t + 1) * a.rounds_per_task
        print(f'   round {r:>3} (het task {t}): nguong {ng}')


if __name__ == '__main__':
    main()
