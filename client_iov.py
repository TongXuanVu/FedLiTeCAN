"""Flower client cho bo data iov (CICIoV) voi model CNN1D.

Chay:  python client_iov.py --client-id 0
Data:  client_<id>.pt dang dict {'x': (N,31) float16, 'y': (N,) int64}
       mac dinh tai C:/FederatedLearning/FL/core/data iov/federated_data

Luu y: cac client lon co hang chuc trieu mau -> mac dinh gioi han
--max-samples 500000 (giu het lop thieu so, cat bot lop da so).
Dat --max-samples 0 de dung toan bo du lieu.
"""
import argparse
import logging
import os
import time
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model_cnn1d import CNN1D_IDS, FocalLoss, NUM_GLOBAL_CLASSES, INPUT_LEN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = r"C:\FederatedLearning\FL\core\data iov\federated_data"


def subsample_capped(x: np.ndarray, y: np.ndarray, max_samples: int, seed=42):
    """Giu toan bo lop thieu so, cat bot lop da so cho den khi <= max_samples."""
    if max_samples <= 0 or len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    counts = Counter(y.tolist())
    # cap per-class: chia deu quota, lop nao it hon quota thi giu het,
    # phan quota du don cho cac lop lon
    classes = sorted(counts, key=lambda c: counts[c])
    remaining = max_samples
    keep_idx = []
    for i, c in enumerate(classes):
        quota = remaining // (len(classes) - i)
        idx = np.where(y == c)[0]
        if len(idx) > quota:
            idx = rng.choice(idx, quota, replace=False)
        keep_idx.append(idx)
        remaining -= len(idx)
    keep = np.concatenate(keep_idx)
    rng.shuffle(keep)
    return x[keep], y[keep]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, data_dir: str, device: torch.device,
                 max_samples: int, batch_size: int, task: int = None):
        self.client_id = client_id
        self.device = device
        self.task = task

        # ----- Load data -----
        path = f"{data_dir}/client_{client_id}.pt"
        if task is not None:
            # Che do task-incremental: chi nap du lieu cua DUNG task nay, khong
            # gop cac task truoc. FedLiTeCAN khong co replay/KD nen day chinh la
            # dieu kien de do muc do quen thang khi khong ho tro IL.
            path = f"{data_dir}/client_{client_id}_task_{task + 1}.pt"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Client {client_id} khong co du lieu cho task {task}: {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        # Doc float16, CHI ep sang float32 SAU KHI subsample. Ban cu ep truoc nen
        # voi client 29 trieu mau se ton 3,6 GB float32 + 3 ban copy cua
        # train_test_split -> OOM khi chay --max-samples 0. Ket qua khong doi.
        x = blob["x"].numpy()
        y = blob["y"].numpy().astype(np.int64)
        logger.info(f"Client {client_id}: loaded {path} x={x.shape} classes={dict(sorted(Counter(y.tolist()).items()))}")

        x, y = subsample_capped(x, y, max_samples)
        logger.info(f"Client {client_id}: after subsample n={len(y)}")
        # Giu float16 khi khong cat tran (>5 trieu mau), ep float32 theo tung
        # batch trong fit()/evaluate(). Duoi nguong do thi ep het cho nhanh.
        if len(y) <= 5_000_000:
            x = x.astype(np.float32)
        logger.info(f"Client {client_id}: dtype dac trung = {x.dtype}")

        # ----- Split 60/20/20 (stratify neu duoc) -----
        try:
            x_tmp, x_test, y_tmp, y_test = train_test_split(
                x, y, test_size=0.2, stratify=y, random_state=42)
            x_train, x_val, y_train, y_val = train_test_split(
                x_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)
        except ValueError:
            x_tmp, x_test, y_tmp, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42)
            x_train, x_val, y_train, y_val = train_test_split(
                x_tmp, y_tmp, test_size=0.25, random_state=42)

        def loader(xa, ya, shuffle):
            ds = TensorDataset(torch.from_numpy(xa), torch.from_numpy(ya))
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        self.train_loader = loader(x_train, y_train, True)
        self.val_loader = loader(x_val, y_val, False)
        self.test_loader = loader(x_test, y_test, False)
        self.num_train = len(y_train)

        # ----- Model + loss + optimizer -----
        self.model = CNN1D_IDS(input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES,
                               dropout=0.15).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)

        # alpha = sqrt(total / count_c)  (sqrt-inverse class freq, nhu Table II bai bao)
        cnt = Counter(y_train.tolist())
        total = len(y_train)
        weights = [np.sqrt(total / cnt[c]) if c in cnt else 1.0
                   for c in range(NUM_GLOBAL_CLASSES)]
        alpha = torch.tensor(weights, dtype=torch.float32).to(device)
        self.criterion = FocalLoss(alpha=alpha, gamma=2.0)

    # ----- Flower API -----
    def get_parameters(self, config) -> List[np.ndarray]:
        return [v.cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        keys = self.model.state_dict().keys()
        state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
        self.model.load_state_dict(state, strict=True)

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 5))
        self.model.train()
        epoch_loss, epoch_acc = 0.0, 0.0
        for epoch in range(epochs):
            running, correct, total = 0.0, 0, 0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device).float(), yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                running += loss.item()
                correct += (out.argmax(1) == yb).sum().item()
                total += yb.size(0)
            epoch_loss = running / len(self.train_loader)
            epoch_acc = correct / total
            logger.info(f"Client {self.client_id} epoch {epoch+1}/{epochs}: "
                        f"loss={epoch_loss:.4f} acc={epoch_acc:.4f}")
        return self.get_parameters({}), self.num_train, {
            "loss": epoch_loss, "accuracy": epoch_acc}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        pbuf, tbuf = [], []
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device).float(), yb.to(self.device)
                out = self.model(xb)
                loss_sum += self.criterion(out, yb).item()
                p = out.argmax(1)
                correct += (p == yb).sum().item()
                total += yb.size(0)
                pbuf.append(p.cpu().numpy().astype(np.int16))
                tbuf.append(yb.cpu().numpy().astype(np.int16))
        preds = np.concatenate(pbuf)
        targs = np.concatenate(tbuf)
        del pbuf, tbuf
        test_loss = loss_sum / len(self.test_loader)
        acc = correct / total
        bal_acc = balanced_accuracy_score(targs, preds)

        # Tinh du 9 chi so P/R/F1 theo micro / macro / weighted
        metrics: Dict = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "loss": float(test_loss),
        }
        for avg in ("micro", "macro", "weighted"):
            prec, rec, f1, _ = precision_recall_fscore_support(
                targs, preds, average=avg, zero_division=0)
            metrics[f"{avg}_precision"] = float(prec)
            metrics[f"{avg}_recall"] = float(rec)
            metrics[f"{avg}_f1"] = float(f1)

        logger.info(
            f"Client {self.client_id} eval: loss={test_loss:.4f} acc={acc:.4f} "
            f"micro_f1={metrics['micro_f1']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f}")
        return float(test_loss), total, metrics


def main():
    parser = argparse.ArgumentParser(description="CNN1D IoV Flower client")
    parser.add_argument("--client-id", type=int, required=True, choices=range(10))
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8081")
    parser.add_argument("--max-samples", type=int, default=500_000,
                        help="Gioi han so mau moi client (0 = dung het)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--task", type=int, default=None, choices=range(5),
                        help="Che do task-incremental: chi nap client_<i>_task_<task+1>.pt. "
                             "Bo qua = gop het cac task (client_<i>.pt)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    client = FlowerClient(args.client_id, args.data_dir, device,
                          args.max_samples, args.batch_size, args.task)

    for attempt in range(3):
        try:
            fl.client.start_numpy_client(server_address=args.server_address, client=client)
            break
        except Exception as e:
            logger.error(f"Connect attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(5)
            else:
                raise


if __name__ == "__main__":
    main()
