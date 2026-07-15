
import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics, Parameters, FitRes, EvaluateRes
import logging
from collections import OrderedDict, defaultdict
import json
import os
from datetime import datetime
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GLOBAL_CLASSES = ["Normal", "DoS", "Fuzzy", "Spoofing", "Gear", "RPM"]
NUM_GLOBAL_CLASSES = len(GLOBAL_CLASSES)


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


def get_model_parameters(model: nn.Module):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class MetricsTracker:

    def __init__(self):
        self.round_metrics = {}
        self.client_metrics = defaultdict(list)
        self.global_metrics = []

    def add_round_metrics(self, round_num: int, fit_metrics: Dict, eval_metrics: Dict):
        self.round_metrics[round_num] = {
            'fit_metrics': fit_metrics,
            'eval_metrics': eval_metrics,
            'timestamp': datetime.now().isoformat()
        }

        self.global_metrics.append({
            'round': round_num,
            'weighted_accuracy': eval_metrics.get('weighted_accuracy', 0.0),
            'weighted_balanced_accuracy': eval_metrics.get('weighted_balanced_accuracy', 0.0),
            'min_accuracy': eval_metrics.get('min_accuracy', 0.0),
            'max_accuracy': eval_metrics.get('max_accuracy', 0.0),
            'std_accuracy': eval_metrics.get('std_accuracy', 0.0),
        })

    def add_client_metrics(self, client_id: str, round_num: int, metrics: Dict):
        self.client_metrics[client_id].append({
            'round': round_num,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })

    def get_summary(self) -> Dict:
        if not self.global_metrics:
            return {}

        latest_round = max(self.round_metrics.keys())
        best_round = max(self.global_metrics, key=lambda x: x['weighted_balanced_accuracy'])

        return {
            'total_rounds': len(self.round_metrics),
            'latest_round': latest_round,
            'best_round': best_round['round'],
            'best_weighted_balanced_accuracy': best_round['weighted_balanced_accuracy'],
            'latest_weighted_balanced_accuracy': self.global_metrics[-1]['weighted_balanced_accuracy'],
            'convergence_trend': self._calculate_convergence_trend()
        }

    def _calculate_convergence_trend(self) -> str:
        if len(self.global_metrics) < 5:
            return "insufficient_data"

        recent_accuracies = [m['weighted_balanced_accuracy'] for m in self.global_metrics[-5:]]
        trend = np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0]

        if trend > 0.001:
            return "improving"
        elif trend < -0.001:
            return "degrading"
        else:
            return "stable"

    def save_to_file(self, filename: str):
        data = {
            'round_metrics': self.round_metrics,
            'client_metrics': dict(self.client_metrics),
            'global_metrics': self.global_metrics,
            'summary': self.get_summary()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class EnhancedFederatedStrategy(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.round_num = 0
        self.metrics_tracker = MetricsTracker()
        self.client_info = {}

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple]:
        config = {
            "local_epochs": 5,
            "server_round": server_round,
        }

        if server_round <= 10:
            config["local_epochs"] = 5
        elif server_round <= 25:
            config["local_epochs"] = 3
        else:
            config["local_epochs"] = 2


        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        return [(client, fl.common.FitIns(parameters, config)) for client in clients]



    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple]:
        config = {
            "server_round": server_round,
        }

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns] | BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, fl.common.Scalar]]:

        self.round_num = server_round


        client_info = {}
        total_examples = 0

        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics

            client_info[client_id] = {
                'num_examples': num_examples,
                'loss': metrics.get('loss', 0.0),
                'accuracy': metrics.get('accuracy', 0.0),
            }

            total_examples += num_examples

            self.metrics_tracker.add_client_metrics(client_id, server_round, metrics)

        for client_id, info in client_info.items():
            contribution = (info['num_examples'] / total_examples) * 100


        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        fit_metrics = {
            'total_examples': total_examples,
            'participating_clients': len(results),
            'failed_clients': len(failures),
            'avg_client_loss': np.mean([info['loss'] for info in client_info.values()]),
            'avg_client_accuracy': np.mean([info['accuracy'] for info in client_info.values()]),
            'std_client_accuracy': np.std([info['accuracy'] for info in client_info.values()]),
        }

        self.client_info[server_round] = client_info

        return aggregated_parameters, fit_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns] | BaseException],
    ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:


        client_results = {}
        total_examples = 0

        for client_proxy, eval_res in results:
            client_id = client_proxy.cid
            metrics = eval_res.metrics
            num_examples = eval_res.num_examples

            client_results[client_id] = {
                'num_examples': num_examples,
                'loss': eval_res.loss,
                'accuracy': metrics.get('accuracy', 0.0),
                'balanced_accuracy': metrics.get('balanced_accuracy', 0.0),
                'macro_precision': metrics.get('macro_precision', 0.0),
                'macro_recall': metrics.get('macro_recall', 0.0),
                'macro_f1': metrics.get('macro_f1', 0.0),
                'weighted_precision': metrics.get('weighted_precision', 0.0),
                'weighted_recall': metrics.get('weighted_recall', 0.0),
                'weighted_f1': metrics.get('weighted_f1', 0.0),
                'total_samples': metrics.get('total_samples', 0),
                'num_classes_present': metrics.get('num_classes_present', 0),
            }

            total_examples += num_examples

        weighted_accuracy = sum([
            client_results[cid]['accuracy'] * client_results[cid]['num_examples']
            for cid in client_results
        ]) / total_examples

        weighted_balanced_accuracy = sum([
            client_results[cid]['balanced_accuracy'] * client_results[cid]['num_examples']
            for cid in client_results
        ]) / total_examples

        weighted_loss = sum([
            client_results[cid]['loss'] * client_results[cid]['num_examples']
            for cid in client_results
        ]) / total_examples

        macro_accuracy = np.mean([client_results[cid]['accuracy'] for cid in client_results])
        macro_balanced_accuracy = np.mean([client_results[cid]['balanced_accuracy'] for cid in client_results])
        macro_f1 = np.mean([client_results[cid]['macro_f1'] for cid in client_results])

        client_accuracies = [client_results[cid]['accuracy'] for cid in client_results]
        client_balanced_accuracies = [client_results[cid]['balanced_accuracy'] for cid in client_results]

        eval_metrics = {
            'weighted_accuracy': weighted_accuracy,
            'weighted_balanced_accuracy': weighted_balanced_accuracy,
            'weighted_loss': weighted_loss,
            'macro_accuracy': macro_accuracy,
            'macro_balanced_accuracy': macro_balanced_accuracy,
            'macro_f1': macro_f1,
            'min_accuracy': min(client_accuracies),
            'max_accuracy': max(client_accuracies),
            'std_accuracy': np.std(client_accuracies),
            'min_balanced_accuracy': min(client_balanced_accuracies),
            'max_balanced_accuracy': max(client_balanced_accuracies),
            'std_balanced_accuracy': np.std(client_balanced_accuracies),
            'total_examples': total_examples,
            'participating_clients': len(results),
        }



        fit_metrics = self.client_info.get(server_round, {})
        self.metrics_tracker.add_round_metrics(server_round, fit_metrics, eval_metrics)


        summary = self.metrics_tracker.get_summary()

        if server_round % 5 == 0:
            self.metrics_tracker.save_to_file(f"federated_metrics_round_{server_round}.json")

        return weighted_balanced_accuracy, eval_metrics

def main():

    INPUT_DIM = 9
    SEQ_LEN = 10
    global_model = SequenceTransformerClassifier(
        input_dim=INPUT_DIM,
        seq_len=SEQ_LEN,
        num_classes=NUM_GLOBAL_CLASSES,
        d_model=64,
        nhead=2,
        num_layers=2,
        dropout=0.15
    )

    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)

    initial_parameters = get_model_parameters(global_model)


    strategy = EnhancedFederatedStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=4,
        min_evaluate_clients=4,
        min_available_clients=4,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        fit_metrics_aggregation_fn=None,
        evaluate_metrics_aggregation_fn=None,
    )


    num_rounds = 40
    config = fl.server.ServerConfig(num_rounds=num_rounds)


    os.makedirs("federated_results", exist_ok=True)

    try:
        fl.server.start_server(
            server_address="0.0.0.0:8081",
            config=config,
            strategy=strategy,
        )


        strategy.metrics_tracker.save_to_file("federated_results/final_metrics.json")
        logger.info("Federated learning completed successfully")

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
        strategy.metrics_tracker.save_to_file("federated_results/interrupted_metrics.json")
    except Exception as e:
        logger.error(f"Server error: {e}")
        strategy.metrics_tracker.save_to_file("federated_results/error_metrics.json")
        raise


if __name__ == "__main__":
    main()
