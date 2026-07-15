# FedLiTeCAN — Transformer & CNN1D Intrusion Detection for CAN / IoV Networks

This repository contains two parts:

## Part 1: Transformer-based IDS for CAN (paper reproduction)
Implementation of an encoder-only Transformer used for intrusion detection in CAN Bus network (paper: *FedLiTeCAN: A Federated Lightweight Transformer for Fast and Robust CAN Bus Intrusion Detection*, arXiv:2512.24088).

### Datasets
1. [Car Hacking](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
2. [Survival Analysis](https://ocslab.hksecurity.net/Datasets/survival-ids)
3. [Car Hacking: Attack & Defense Challenge 2020](https://ocslab.hksecurity.net/Datasets/carchallenge2020)

### Code
1. `car_hacking.py`: transformer model for intrusion detection on the car hacking dataset
2. `survival_analysis.py`: transformer model for intrusion detection on the survival analysis dataset
3. `unseen_attack.py`: cross-dataset evaluation, training on survival analysis and testing on Car Hacking
4. `server.py`, `client.py`: Flower FL setup (4 clients) with the transformer model

## Part 2: CNN1D federated IDS for IoV dataset
Lightweight 1D-CNN (~40k params) trained with Flower FedAvg on a partitioned IoV dataset
(10 clients, 31 features, 13 classes, non-IID).

### Code
1. `model_cnn1d.py`: CNN1D model (3 Conv1d blocks + GAP + FC) and Focal Loss
2. `client_iov.py`: Flower client, loads `client_<id>.pt`, full per-round metrics
   (Loss, Accuracy, Micro/Macro/Weighted Precision-Recall-F1)
3. `server_iov.py`: Flower FedAvg server with per-round checkpoints and 3 modes
4. `evaluate_global_iov.py`: evaluate any checkpoint on the global test set

### Usage
```bash
# Train from scratch (checkpoints saved to checkpoints_iov/ every round)
python server_iov.py --mode train --rounds 40
python client_iov.py --client-id 0   # ... repeat for ids 0..9

# Resume from any checkpoint
python server_iov.py --mode resume --checkpoint checkpoints_iov/round_015.pth --rounds 40

# Test (evaluation only) from any checkpoint
python server_iov.py --mode test --checkpoint checkpoints_iov/round_040.pth

# Evaluate a checkpoint on the global test set
python evaluate_global_iov.py --model checkpoints_iov/round_040.pth
```

Per-round metrics are written to `metrics_iov.csv` (aggregated, weighted by client test size)
and `metrics_iov_per_client.json` (per client).
