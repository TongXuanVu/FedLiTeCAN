"""Flower server cho bo data iov voi model CNN1D (FedAvg).

3 che do (--mode):
  train  : huan luyen tu dau, luu checkpoint moi round vao checkpoints_iov/
  resume : tiep tuc tu 1 checkpoint bat ky (--checkpoint), round tiep noi
  test   : chi danh gia 1 checkpoint (--checkpoint), khong huan luyen

Vi du:
  python server_iov.py --mode train  --rounds 40
  python server_iov.py --mode resume --checkpoint checkpoints_iov/round_015.pth --rounds 40
  python server_iov.py --mode test   --checkpoint checkpoints_iov/round_040.pth

Moi round luu:
  - checkpoints_iov/round_XXX.pth  (state_dict + so round)
  - metrics_iov.csv: Loss, Accuracy, Micro/Macro/Weighted-Precision/Recall/F1
    (trung binh co trong so theo so mau test cua tung client)
  - metrics_iov_per_client.json: metric chi tiet tung client tung round
"""
import argparse
import csv
import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common import EvaluateRes, Parameters

from model_cnn1d import CNN1D_IDS, NUM_GLOBAL_CLASSES, INPUT_LEN

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CKPT_DIR = "checkpoints_iov"
CSV_FILE = "metrics_iov.csv"
PER_CLIENT_FILE = "metrics_iov_per_client.json"

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


class CheckpointFedAvg(fl.server.strategy.FedAvg):
    """FedAvg + luu checkpoint moi round + tong hop day du metric."""

    def __init__(self, template_model, local_epochs=5, start_round=0,
                 test_only=False, **kwargs):
        super().__init__(**kwargs)
        self.template_model = template_model
        self.local_epochs = local_epochs
        self.start_round = start_round        # round da hoc truoc do (resume)
        self.test_only = test_only
        self.latest_parameters: Optional[Parameters] = None
        self.per_client_log: Dict = {}

    # ---------- FIT ----------
    def configure_fit(self, server_round, parameters, client_manager):
        if self.test_only:
            return []  # test mode: bo qua huan luyen
        config = {"local_epochs": self.local_epochs,
                  "server_round": self.start_round + server_round}
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        return [(c, fl.common.FitIns(parameters, config)) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        if self.test_only:
            return None, {}
        params, metrics = super().aggregate_fit(server_round, results, failures)
        if params is not None:
            self.latest_parameters = params
            self._save_checkpoint(self.start_round + server_round, params)
        return params, metrics

    def _save_checkpoint(self, global_round: int, params: Parameters):
        os.makedirs(CKPT_DIR, exist_ok=True)
        ndarrays = fl.common.parameters_to_ndarrays(params)
        state = ndarrays_to_state_dict(self.template_model, ndarrays)
        path = os.path.join(CKPT_DIR, f"round_{global_round:03d}.pth")
        torch.save({"round": global_round, "model_state_dict": state}, path)
        logger.info(f"[Round {global_round}] checkpoint saved -> {path}")

    # ---------- EVALUATE ----------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures,
    ) -> Tuple[Optional[float], Dict]:
        if not results:
            return None, {}
        global_round = self.start_round + server_round
        total = sum(r.num_examples for _, r in results)

        # Trung binh co trong so theo so mau test cua tung client
        agg: Dict[str, float] = {}
        for key in METRIC_KEYS:
            agg[key] = sum(
                float(r.metrics.get(key, r.loss if key == "loss" else 0.0)) * r.num_examples
                for _, r in results) / total

        logger.info(
            f"[Round {global_round}] loss={agg['loss']:.4f} acc={agg['accuracy']:.4f} | "
            f"micro P/R/F1={agg['micro_precision']:.4f}/{agg['micro_recall']:.4f}/{agg['micro_f1']:.4f} | "
            f"macro P/R/F1={agg['macro_precision']:.4f}/{agg['macro_recall']:.4f}/{agg['macro_f1']:.4f} | "
            f"weighted P/R/F1={agg['weighted_precision']:.4f}/{agg['weighted_recall']:.4f}/{agg['weighted_f1']:.4f}")

        # Ghi CSV tong hop
        append_csv_row(CSV_FILE, [global_round] + [round(agg[k], 6) for k in METRIC_KEYS])

        # Ghi metric tung client
        self.per_client_log[global_round] = {
            str(cp.cid): {"num_examples": r.num_examples,
                          **{k: float(v) for k, v in r.metrics.items()}}
            for cp, r in results
        }
        with open(PER_CLIENT_FILE, "w") as f:
            json.dump(self.per_client_log, f, indent=2)

        return agg["loss"], {k: agg[k] for k in METRIC_KEYS}


def load_checkpoint(path: str, model) -> int:
    """Nap checkpoint vao model, tra ve so round da hoc."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return int(ckpt.get("round", 0))
    model.load_state_dict(ckpt)  # file chi chua state_dict
    return 0


def main():
    parser = argparse.ArgumentParser(description="CNN1D IoV Flower server")
    parser.add_argument("--mode", choices=["train", "resume", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Duong dan checkpoint (bat buoc voi resume/test)")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=40,
                        help="Tong so round muc tieu (resume se chay phan con lai)")
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--address", type=str, default="0.0.0.0:8081")
    args = parser.parse_args()

    if args.mode in ("resume", "test") and not args.checkpoint:
        parser.error(f"--mode {args.mode} can --checkpoint")

    model = CNN1D_IDS(input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES, dropout=0.15)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"CNN1D model: {n_params:,} trainable params")

    start_round = 0
    if args.mode in ("resume", "test"):
        start_round = load_checkpoint(args.checkpoint, model)
        logger.info(f"Loaded checkpoint '{args.checkpoint}' (round {start_round})")

    if args.mode == "test":
        num_rounds = 1  # chi 1 vong danh gia
    elif args.mode == "resume":
        num_rounds = args.rounds - start_round
        if num_rounds <= 0:
            logger.error(f"Checkpoint da o round {start_round} >= --rounds {args.rounds}, "
                         f"khong con round nao de chay.")
            return
        logger.info(f"Resume: chay tiep {num_rounds} rounds "
                    f"({start_round + 1} -> {args.rounds})")
    else:
        num_rounds = args.rounds

    strategy = CheckpointFedAvg(
        template_model=model,
        local_epochs=args.local_epochs,
        start_round=start_round,
        test_only=(args.mode == "test"),
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters(model)),
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Luu global model cuoi cung (mode train/resume)
    if strategy.latest_parameters is not None:
        ndarrays = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
        model.load_state_dict(ndarrays_to_state_dict(model, ndarrays))
        torch.save(model.state_dict(), "cnn1d_iov_global.pth")
        logger.info("Saved final global model -> cnn1d_iov_global.pth")


if __name__ == "__main__":
    main()
