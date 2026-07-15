"""Danh gia global model CNN1D tren global_test_data.pt (42M mau).

Chay:  python evaluate_global_iov.py [--max-samples 2000000]
"""
import argparse
import logging
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch.utils.data import DataLoader, TensorDataset

from model_cnn1d import CNN1D_IDS, NUM_GLOBAL_CLASSES, INPUT_LEN
from client_iov import subsample_capped

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_TEST = r"C:\FederatedLearning\FL\core\data iov\global_test_data.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cnn1d_iov_global.pth")
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST)
    parser.add_argument("--max-samples", type=int, default=2_000_000,
                        help="0 = danh gia tren toan bo 42M mau (cham)")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN1D_IDS(input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    # Ho tro ca checkpoint round_XXX.pth va file state_dict thuan
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        logger.info(f"Loading checkpoint from round {ckpt.get('round', '?')}")
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()

    blob = torch.load(args.test_file, map_location="cpu", weights_only=False)
    x = blob["x"].to(torch.float32).numpy()
    y = blob["y"].numpy().astype(np.int64)
    logger.info(f"Global test: {x.shape}, classes={dict(sorted(Counter(y.tolist()).items()))}")

    x, y = subsample_capped(x, y, args.max_samples)
    logger.info(f"Evaluating on n={len(y)}")

    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                        batch_size=args.batch_size)
    preds, targs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            targs.extend(yb.numpy())

    preds, targs = np.array(preds), np.array(targs)
    acc = (preds == targs).mean()
    bal = balanced_accuracy_score(targs, preds)
    print(f"\nAccuracy: {acc:.4f} | Balanced accuracy: {bal:.4f}\n")
    print(classification_report(targs, preds, zero_division=0))


if __name__ == "__main__":
    main()
