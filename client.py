

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections import OrderedDict, Counter
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import logging
import argparse
import gc
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GLOBAL_CLASSES = ["Normal", "DoS", "Fuzzy", "Spoofing", "Gear", "RPM"]
NUM_GLOBAL_CLASSES = len(GLOBAL_CLASSES)


CLIENT_CLASS_MAPPINGS = {
    1: {
        "local_classes": ["Normal", "DoS", "Fuzzy", "Spoofing", "Gear", "RPM"],
        "local_to_global": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        "dataset_file": "can_combined_attacks_preprocessed.csv",
        "max_samples": 16000000,
        "sample_strategy": "balanced",
        "data_split": "first_third",
        "partition_seed": 42
    },

    2: {
        "local_classes": ["Normal", "Flood", "Fuzzy", "Malfunc"],
        "local_to_global": {0: 0, 1: 1, 2: 2, 3: 3},
        "dataset_file": "sonata_combined_attacks_preprocessed.csv",
        "max_samples": None,
        "sample_strategy": "all",
        "data_split": "all",
        "partition_seed": 42
    },
    3: {
        "local_classes": ["Normal", "Flood", "Fuzzy", "Malfunc"],
        "local_to_global": {0: 0, 1: 1, 2: 2, 3: 3},
        "dataset_file": "kia_combined_attacks_preprocessed.csv",
        "max_samples": None,
        "sample_strategy": "all",
        "data_split": "all",
        "partition_seed": 42
    },
    4: {
        "local_classes": ["Normal", "Flood", "Fuzzy", "Malfunc"],
        "local_to_global": {0: 0, 1: 1, 2: 2, 3: 3},
        "dataset_file": "spark_combined_attacks_preprocessed.csv",
        "max_samples": None,
        "sample_strategy": "all",
        "data_split": "all",
        "partition_seed": 42
    }
}

class SequenceTransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=64, nhead=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.positional_encoding = nn.Parameter(torch.randn(seq_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.input_projection(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.positional_encoding.unsqueeze(0)[:, :x.size(1), :]
        x = self.dropout(x)

        x = self.transformer_encoder(x)

        cls_output = x[:, 0, :]

        return self.classifier(cls_output)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info(f"Early stopping triggered. Restored best weights.")


class ImbalancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, present_classes=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.present_classes = present_classes if present_classes is not None else set(range(NUM_GLOBAL_CLASSES))

    def forward(self, outputs, targets):
        ce_loss = nn.functional.cross_entropy(outputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        weights = torch.ones_like(targets, dtype=torch.float32)
        for class_idx in self.present_classes:
            weights[targets == class_idx] = 1.5

        weighted_loss = focal_loss * weights
        return weighted_loss.mean()


def create_distinct_partitions(data, labels, partition_seed=42):

    np.random.seed(partition_seed)

    total_samples = len(data)
    indices = np.arange(total_samples)

    np.random.shuffle(indices)

    partition_size = total_samples // 3

    partitions = {
        'first_third': indices[:partition_size],
        'second_third': indices[partition_size:2*partition_size],
        'third_third': indices[2*partition_size:]
    }


    return partitions


def sample_data_balanced(data, labels, max_samples, min_samples_per_class=1000):

    if len(data) <= max_samples:
        return data, labels

    class_counts = Counter(labels)
    unique_classes = list(class_counts.keys())

    samples_per_class = min(
        max_samples // len(unique_classes),
        min_samples_per_class
    )

    sampled_indices = []
    for class_label in unique_classes:
        class_indices = np.where(labels == class_label)[0]
        if len(class_indices) > samples_per_class:
            selected_indices = np.random.choice(
                class_indices, samples_per_class, replace=False
            )
        else:
            selected_indices = class_indices
        sampled_indices.extend(selected_indices)

    np.random.shuffle(sampled_indices)

    if len(sampled_indices) > max_samples:
        sampled_indices = sampled_indices[:max_samples]

    if hasattr(data, 'iloc'):
        sampled_data = data.iloc[sampled_indices]
        sampled_labels = labels.iloc[sampled_indices] if hasattr(labels, 'iloc') else labels[sampled_indices]
    else:
        sampled_data = data[sampled_indices]
        sampled_labels = labels[sampled_indices]

    return sampled_data, sampled_labels


def create_sequences_with_stride(features, labels, window_size, stride):
    sequences = []
    sequence_labels = []

    for i in range(0, len(features) - window_size + 1, stride):
        sequence = features.iloc[i:i + window_size].values

        window_labels = labels.iloc[i:i + window_size] if hasattr(labels, 'iloc') else labels[i:i + window_size]

        window_label = 0

        for label in window_labels:
            if label != 0:
                window_label = label
                break

        sequences.append(sequence)
        sequence_labels.append(window_label)

    return np.array(sequences), np.array(sequence_labels)


def get_model_parameters(model: nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, client_id: int, device: torch.device):
        self.client_id = client_id
        self.device = device

        self.config = CLIENT_CLASS_MAPPINGS[client_id]
        self.local_to_global = self.config["local_to_global"]
        self.global_classes_present = set(self.local_to_global.values())


        self._load_and_prepare_data()

        self.model = SequenceTransformerClassifier(
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            num_classes=NUM_GLOBAL_CLASSES,
            d_model=64,
            nhead=2,
            num_layers=2,
            dropout=0.15
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        class_weights = self._calculate_global_class_weights()
        self.criterion = ImbalancedFocalLoss(
            alpha=class_weights,
            gamma=2.0,
            present_classes=self.global_classes_present
        )

        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    def _load_and_prepare_data(self):
        try:
            dataset_file = self.config["dataset_file"]
            data = pd.read_csv(dataset_file)

            logger.info(f"Client {self.client_id} loaded data from {dataset_file}: {data.shape}")

            labels = data.iloc[:, -1].astype(np.int64)
            features = data.iloc[:, :-1]


            if self.client_id in [1, 2, 3] and self.config["data_split"] != "all":
                logger.info(f"Client {self.client_id} creating distinct data partitions...")


                partitions = create_distinct_partitions(
                    features, labels,
                    partition_seed=self.config["partition_seed"]
                )


                partition_indices = partitions[self.config["data_split"]]


                if hasattr(features, 'iloc'):
                    features = features.iloc[partition_indices].reset_index(drop=True)
                    labels = labels.iloc[partition_indices].reset_index(drop=True)
                else:
                    features = features[partition_indices]
                    labels = labels[partition_indices]



                if self.client_id == 1:

                    self.partition_indices = partition_indices
                    with open(f"client_{self.client_id}_partition_indices.txt", "w") as f:
                        f.write(",".join(map(str, partition_indices)))


                partition_class_counts = Counter(labels)
                logger.info(f"Client {self.client_id} partition class distribution: {dict(partition_class_counts)}")


            if self.config["max_samples"] is not None and len(features) > self.config["max_samples"]:

                features, labels = sample_data_balanced(
                    features, labels, self.config["max_samples"]
                )



            global_labels = []
            for label in labels:
                if label in self.local_to_global:
                    global_labels.append(self.local_to_global[label])
                else:
                    logger.warning(f"Unknown local label {label} for client {self.client_id}, mapping to Normal (0)")
                    global_labels.append(0)

            global_labels = np.array(global_labels)


            local_counts = Counter(labels)
            global_counts = Counter(global_labels)
            logger.info(f"Client {self.client_id} local class distribution: {dict(local_counts)}")
            logger.info(f"Client {self.client_id} global class distribution: {dict(global_counts)}")


            try:
                X_temp, X_test, y_temp, y_test = train_test_split(
                    features, global_labels, test_size=0.2, stratify=global_labels, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
                )
            except ValueError:
                logger.warning(f"Stratification failed for client {self.client_id}, using random split")
                X_temp, X_test, y_temp, y_test = train_test_split(
                    features, global_labels, test_size=0.2, random_state=42
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=0.25, random_state=42
                )

            # Normalize features
            self.features_min = X_train.min()
            self.features_max = X_train.max()
            X_train_norm = (X_train - self.features_min) / (self.features_max - self.features_min + 1e-8)
            X_val_norm = (X_val - self.features_min) / (self.features_max - self.features_min + 1e-8)
            X_test_norm = (X_test - self.features_min) / (self.features_max - self.features_min + 1e-8)


            WINDOW_SIZE = 10
            STRIDE = 2

            X_train_seq, y_train_seq = create_sequences_with_stride(X_train_norm, pd.Series(y_train), WINDOW_SIZE, STRIDE)
            X_val_seq, y_val_seq = create_sequences_with_stride(X_val_norm, pd.Series(y_val), WINDOW_SIZE, STRIDE)
            X_test_seq, y_test_seq = create_sequences_with_stride(X_test_norm, pd.Series(y_test), WINDOW_SIZE, STRIDE)

            # Convert to torch tensors
            self.X_train = torch.tensor(X_train_seq, dtype=torch.float32)
            self.y_train = torch.tensor(y_train_seq, dtype=torch.long)
            self.X_val = torch.tensor(X_val_seq, dtype=torch.float32)
            self.y_val = torch.tensor(y_val_seq, dtype=torch.long)
            self.X_test = torch.tensor(X_test_seq, dtype=torch.float32)
            self.y_test = torch.tensor(y_test_seq, dtype=torch.long)

            # Set dimensions
            self.input_dim = self.X_train.shape[2]
            self.seq_len = self.X_train.shape[1]

            batch_size = 128
            self.train_loader = DataLoader(TensorDataset(self.X_train, self.y_train), batch_size=batch_size, shuffle=True)
            self.val_loader = DataLoader(TensorDataset(self.X_val, self.y_val), batch_size=batch_size, shuffle=False)
            self.test_loader = DataLoader(TensorDataset(self.X_test, self.y_test), batch_size=batch_size, shuffle=False)



            del X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq
            del X_train_norm, X_val_norm, X_test_norm
            gc.collect()

        except Exception as e:
            logger.error(f"Failed to load data for client {self.client_id}: {e}")
            raise

    def _calculate_global_class_weights(self):

        global_class_counts = Counter(self.y_train.numpy())
        total_samples = len(self.y_train)

        class_weights = torch.ones(NUM_GLOBAL_CLASSES, dtype=torch.float32)

        for class_idx in range(NUM_GLOBAL_CLASSES):
            if class_idx in global_class_counts:
                weight = total_samples / (global_class_counts[class_idx] * NUM_GLOBAL_CLASSES)
                class_weights[class_idx] = weight
            else:
                if global_class_counts:
                    median_count = np.median(list(global_class_counts.values()))
                    class_weights[class_idx] = total_samples / (median_count * NUM_GLOBAL_CLASSES)
                else:
                    class_weights[class_idx] = 1.0

        return class_weights.to(self.device)

    def get_parameters(self, config) -> List[np.ndarray]:

        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_model_parameters(self.model, parameters)

    def fit(self, parameters: List[np.ndarray], config) -> Tuple[List[np.ndarray], int, Dict]:

        self.set_parameters(parameters)

        epochs = config.get("local_epochs", 5)

        logger.info(f"Client {self.client_id} starting local training for {epochs} epochs")

        self.model.train()
        self.early_stopping.early_stop = False

        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                del inputs, targets, outputs, loss
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

            epoch_loss = running_loss / len(self.train_loader)
            epoch_acc = correct / total if total > 0 else 0.0

            val_loss = self._validate()
            self.scheduler.step(val_loss)
            self.early_stopping(val_loss, self.model)

            if epoch % 2 == 0:
                logger.info(f"Client {self.client_id} Epoch {epoch+1}/{epochs}: "
                           f"Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}, Val_Loss={val_loss:.4f}")

            if self.early_stopping.early_stop:
                logger.info(f"Client {self.client_id} early stopping at epoch {epoch+1}")
                break

        parameters_updated = self.get_parameters(config={})
        num_examples = len(self.train_loader.dataset)

        metrics = {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "val_loss": val_loss,
            "early_stopped": self.early_stopping.early_stop,
            "epochs_completed": epoch + 1,
        }

        logger.info(f"Client {self.client_id} training completed")

        return parameters_updated, num_examples, metrics

    def _validate(self) -> float:
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def evaluate(self, parameters: List[np.ndarray], config) -> Tuple[float, int, Dict]:

        self.set_parameters(parameters)

        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss = self.criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        test_loss /= len(self.test_loader)
        accuracy = correct / total if total > 0 else 0.0

        balanced_acc = balanced_accuracy_score(all_targets, all_predictions) if len(all_targets) > 0 else 0.0

        if len(set(all_targets)) > 1:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='macro', zero_division=0
            )
            precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
                all_targets, all_predictions, average='weighted', zero_division=0
            )
        else:
            precision = recall = f1 = precision_w = recall_w = f1_w = 0.0

        num_examples = len(self.test_loader.dataset)

        metrics = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_acc),
            "loss": float(test_loss),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "weighted_precision": float(precision_w),
            "weighted_recall": float(recall_w),
            "weighted_f1": float(f1_w),
            "total_samples": int(total),
            "num_classes_present": int(len(self.global_classes_present)),
        }


        logger.info(f"Client {self.client_id} Test Evaluation: "
                   f"Loss={test_loss:.4f}, Acc={accuracy:.4f}, Balanced Acc={balanced_acc:.4f}")

        return float(test_loss), int(num_examples), metrics


def main():

    parser = argparse.ArgumentParser(description="Enhanced Flower Client")
    parser.add_argument("--client-id", type=int, required=True, choices=[1, 2, 3, 4],
                       help="Client ID (1, 2, 3, or 4)")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8081",
                       help="Server address (default: 127.0.0.1:8081)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    try:
        client = FlowerClient(client_id=args.client_id, device=device)

        logger.info(f"Starting Enhanced Flower client {args.client_id}")
        logger.info(f"Connecting to server at {args.server_address}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                fl.client.start_numpy_client(
                    server_address=args.server_address,
                    client=client
                )
                break
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    # logger.error("Max retries reached. Exiting.")
                    raise

    except Exception as e:
        logger.error(f"Client {args.client_id} failed to start: {e}")
        raise


if __name__ == "__main__":
    main()
