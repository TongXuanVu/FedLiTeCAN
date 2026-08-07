"""Flower server cho bo data iov voi model CNN1D (FedAvg).

Checkpoint: CHI luu model toan cuc sau khi server tong hop (aggregate),
moi round 1 file trong checkpoints_iov/.

Danh gia: tap trung tai server tren global_test_data.pt sau moi round
(khong danh gia tren test cuc bo cua client).

3 che do (--mode):
  train  : huan luyen tu dau
  resume : tiep tuc tu 1 checkpoint bat ky (--checkpoint)
  test   : danh gia 1 checkpoint tren global test (KHONG can chay client)

Vi du:
  python server_iov.py --mode train  --rounds 30 --local-epochs 1
  python server_iov.py --mode resume --checkpoint checkpoints_iov/round_015.pth --rounds 30
  python server_iov.py --mode test   --checkpoint checkpoints_iov/round_030.pth

Moi round ghi vao metrics_iov.csv: Loss, Accuracy,
Micro/Macro/Weighted-Precision/Recall/F1 (tren global test set).
"""
import argparse
import csv
import logging
import os
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import Parameters
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from model_cnn1d import CNN1D_IDS, FocalLoss, NUM_GLOBAL_CLASSES, INPUT_LEN
from client_iov import subsample_capped

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CKPT_DIR = "checkpoints_iov"
CSV_FILE = "metrics_iov.csv"
DEFAULT_TEST = r"C:\FederatedLearning\FL\core\data iov\global_test_data.pt"

METRIC_KEYS = [
    "loss", "accuracy",
    "micro_precision", "micro_recall", "micro_f1",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
]
CSV_HEADER = ["round"] + METRIC_KEYS


def get_model_parameters(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def ndarrays_to_state_dict(model, ndarrays):
    keys = model.state_dict().keys()
    return OrderedDict({k: torch.tensor(v) for k, v in zip(keys, ndarrays)})


def append_csv_row(path: str, row: List):
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
        w.writerow(row)


# Chia task giong AFSIC-IoV, de so sanh cong bang: [3, 3, 3, 2, 2] lop moi task.
TASK_INCREMENTS = [3, 3, 3, 2, 2]


def learned_classes(task: int) -> int:
    """So lop da hoc tinh den het task nay (0-indexed)."""
    return sum(TASK_INCREMENTS[:task + 1])


def load_global_test(test_file: str, max_samples: int, task: int = None):
    """Nap global_test_data.pt, subsample (giu tron lop thieu so).

    task khac None -> loc test set ve cac lop DA HOC (0..learned_classes-1),
    dung quy uoc voi trainer cua AFSIC-IoV nen hai ben so sanh duoc.
    """
    logger.info(f"Loading global test set: {test_file}")
    blob = torch.load(test_file, map_location="cpu", weights_only=False)
    x = blob["x"].numpy()                      # float16, khong copy
    y = blob["y"].numpy().astype(np.int64)
    del blob
    logger.info(f"Global test: n={len(y)}, classes={dict(sorted(Counter(y.tolist()).items()))}")
    if task is not None:
        n_cls = learned_classes(task)
        keep = y < n_cls
        x, y = x[keep], y[keep]
        logger.info(f"Task {task}: loc test ve lop 0-{n_cls - 1} -> n={len(y)}")
    x, y = subsample_capped(x, y, max_samples)
    # Giu float16 khi max_samples = 0 (toan bo 42 trieu mau): ep sang float32 se
    # ton 5,2 GB thay vi 2,6 GB. Ep sang float32 theo tung batch trong
    # evaluate_on_global_test, ket qua khong doi.
    if max_samples != 0:
        x = x.astype(np.float32)
    logger.info(f"Evaluating each round on n={len(y)} samples (dtype={x.dtype})")
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=4096, shuffle=False)
    return loader, y


def make_criterion(y: np.ndarray, device):
    """Focal loss voi alpha sqrt-inverse tren phan bo global test."""
    cnt = Counter(y.tolist())
    total = len(y)
    weights = [np.sqrt(total / cnt[c]) if c in cnt else 1.0
               for c in range(NUM_GLOBAL_CLASSES)]
    return FocalLoss(alpha=torch.tensor(weights, dtype=torch.float32).to(device), gamma=2.0)


def evaluate_on_global_test(model, loader, criterion, device) -> Dict[str, float]:
    """Tinh du 11 metric tren global test set."""
    model.eval()
    loss_sum, n_batches, correct, total = 0.0, 0, 0, 0
    # Gom vao list cac mang numpy roi concatenate mot lan. Ban cu dung
    # list.extend() nen voi 42 trieu mau se tao list Python 42 trieu phan tu
    # (~3 GB moi list) va OOM. Ket qua tinh ra hoan toan khong doi.
    preds_buf, targs_buf = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            out = model(xb)
            loss_sum += criterion(out, yb).item()
            n_batches += 1
            p = out.argmax(1)
            correct += (p == yb).sum().item()
            total += yb.size(0)
            preds_buf.append(p.cpu().numpy().astype(np.int16))
            targs_buf.append(yb.cpu().numpy().astype(np.int16))
    preds = np.concatenate(preds_buf)
    targs = np.concatenate(targs_buf)
    del preds_buf, targs_buf
    metrics = {"loss": loss_sum / n_batches, "accuracy": correct / total}
    for avg in ("micro", "macro", "weighted"):
        prec, rec, f1, _ = precision_recall_fscore_support(
            targs, preds, average=avg, zero_division=0)
        metrics[f"{avg}_precision"] = float(prec)
        metrics[f"{avg}_recall"] = float(rec)
        metrics[f"{avg}_f1"] = float(f1)
    return metrics


def log_and_save_metrics(global_round: int, m: Dict[str, float]):
    logger.info(
        f"[Round {global_round}] GLOBAL TEST: loss={m['loss']:.4f} acc={m['accuracy']:.4f} | "
        f"micro P/R/F1={m['micro_precision']:.4f}/{m['micro_recall']:.4f}/{m['micro_f1']:.4f} | "
        f"macro P/R/F1={m['macro_precision']:.4f}/{m['macro_recall']:.4f}/{m['macro_f1']:.4f} | "
        f"weighted P/R/F1={m['weighted_precision']:.4f}/{m['weighted_recall']:.4f}/{m['weighted_f1']:.4f}")
    append_csv_row(CSV_FILE, [global_round] + [round(m[k], 6) for k in METRIC_KEYS])


class CheckpointFedAvg(fl.server.strategy.FedAvg):
    """FedAvg: luu checkpoint model TOAN CUC sau aggregate moi round."""

    def __init__(self, template_model, local_epochs=5, start_round=0, **kwargs):
        super().__init__(**kwargs)
        self.template_model = template_model
        self.local_epochs = local_epochs
        self.start_round = start_round
        self.latest_parameters: Optional[Parameters] = None

    def configure_fit(self, server_round, parameters, client_manager):
        config = {"local_epochs": self.local_epochs,
                  "server_round": self.start_round + server_round}
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        return [(c, fl.common.FitIns(parameters, config)) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            self.latest_parameters = params
            # Luu DUY NHAT model toan cuc da tong hop
            global_round = self.start_round + server_round
            os.makedirs(CKPT_DIR, exist_ok=True)
            state = ndarrays_to_state_dict(
                self.template_model, fl.common.parameters_to_ndarrays(params))
            path = os.path.join(CKPT_DIR, f"round_{global_round:03d}.pth")
            torch.save({"round": global_round, "model_state_dict": state}, path)
            logger.info(f"[Round {global_round}] global checkpoint saved -> {path}")
        return params, metrics


def load_checkpoint(path: str, model) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return int(ckpt.get("round", 0))
    model.load_state_dict(ckpt)
    return 0


def main():
    parser = argparse.ArgumentParser(description="CNN1D IoV Flower server")
    parser.add_argument("--mode", choices=["train", "resume", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint (bat buoc voi resume/test)")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=30,
                        help="Tong so round muc tieu (resume chay phan con lai)")
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--address", type=str, default="0.0.0.0:8081")
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST)
    parser.add_argument("--test-max-samples", type=int, default=1_000_000,
                        help="So mau global test dung moi round (0 = het 42M, cham)")
    parser.add_argument("--task", type=int, default=None, choices=range(5),
                        help="Che do task-incremental: danh gia tren cac lop DA HOC "
                             "(0-2, 0-5, 0-8, 0-10, 0-12). Bo qua = danh gia ca 13 lop")
    args = parser.parse_args()

    if args.mode in ("resume", "test") and not args.checkpoint:
        parser.error(f"--mode {args.mode} can --checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D_IDS(input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES, dropout=0.15)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"CNN1D model: {n_params:,} trainable params | device: {device}")

    start_round = 0
    if args.mode in ("resume", "test"):
        start_round = load_checkpoint(args.checkpoint, model)
        logger.info(f"Loaded checkpoint '{args.checkpoint}' (round {start_round})")

    # Nap global test 1 lan duy nhat
    test_loader, y_test = load_global_test(args.test_file, args.test_max_samples, args.task)
    if args.task is not None:
        n_cls = learned_classes(args.task)
        p_max = np.bincount(y_test, minlength=n_cls)[:n_cls].max() / len(y_test)
        logger.info(f"Task {args.task}: {n_cls} lop | lop da so chiem {p_max * 100:.2f}% | "
                    f"NGUONG SUP macro-F1 = {2 * p_max / (n_cls * (1 + p_max)) * 100:.2f}%")
    criterion = make_criterion(y_test, device)
    model.to(device)

    # ----- MODE TEST: danh gia truc tiep, khong can Flower/client -----
    if args.mode == "test":
        m = evaluate_on_global_test(model, test_loader, criterion, device)
        log_and_save_metrics(start_round, m)
        return

    # ----- MODE TRAIN / RESUME -----
    if args.mode == "resume":
        num_rounds = args.rounds - start_round
        if num_rounds <= 0:
            logger.error(f"Checkpoint da o round {start_round} >= --rounds {args.rounds}.")
            return
        logger.info(f"Resume: chay tiep {num_rounds} rounds ({start_round + 1} -> {args.rounds})")
    else:
        num_rounds = args.rounds

    def evaluate_fn(server_round: int, parameters, config):
        """Server tu danh gia model toan cuc tren global test sau moi round."""
        if server_round == 0 and args.mode == "resume":
            return None  # da co metric cua round nay tu lan chay truoc
        model.load_state_dict(ndarrays_to_state_dict(model, parameters))
        model.to(device)
        m = evaluate_on_global_test(model, test_loader, criterion, device)
        log_and_save_metrics(start_round + server_round, m)
        return m["loss"], m

    strategy = CheckpointFedAvg(
        template_model=model,
        local_epochs=args.local_epochs,
        start_round=start_round,
        fraction_fit=1.0,
        fraction_evaluate=0.0,      # bo danh gia phia client
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters(model)),
        evaluate_fn=evaluate_fn,    # danh gia tap trung tren global test
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    if strategy.latest_parameters is not None:
        ndarrays = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
        model.load_state_dict(ndarrays_to_state_dict(model, ndarrays))
        torch.save(model.state_dict(), "cnn1d_iov_global.pth")
        logger.info("Saved final global model -> cnn1d_iov_global.pth")


if __name__ == "__main__":
    main()
