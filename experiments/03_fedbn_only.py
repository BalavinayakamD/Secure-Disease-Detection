"""
Experiment 03: FedBN Only (without trust weighting).

Standard Federated Batch Normalization where:
- BatchNorm layers (bn1, bn2) stay local on each client
- Only shared layers (fc1, fc2, fc3) are aggregated
- Uses standard FedAvg aggregation (volume-weighted, no trust mechanism)
- No differential privacy noise

This serves as an ablation study to isolate the contribution of:
1. FedBN (personalization via local BN) vs standard FedAvg
2. Trust mechanism in AT-FedBN vs plain FedBN

Usage:
    python experiments/03_fedbn_only.py
"""

import json
import os
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import flwr as fl
from flwr.server import ServerConfig
from flwr.simulation import start_simulation
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.preprocessing import load_processed_data
from src.fl.client import FlowerClient
from src.models.cardio_mlp import CardioMLP
from src.utils.metrics import binary_metrics


def load_hospital_data(hospital_id: int, data_dir: str = 'data/processed'):
    """Load training and validation data for a specific hospital."""
    train_path = os.path.join(data_dir, f'hospital_{hospital_id}_train.csv')
    test_path = os.path.join(data_dir, f'hospital_{hospital_id}_test.csv')
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop(columns=['cardio']).values.astype(np.float32)
    y_train = train_df['cardio'].values.astype(np.float32)
    X_test = test_df.drop(columns=['cardio']).values.astype(np.float32)
    y_test = test_df['cardio'].values.astype(np.float32)
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )
    
    return train_dataset, test_dataset


def make_client_fn(config: dict):
    """Factory function to create Flower clients for simulation."""
    def client_fn(context):
        # Extract client ID from context
        if hasattr(context, 'node_config') and 'partition-id' in context.node_config:
            client_id = context.node_config['partition-id']
        elif hasattr(context, 'node_id'):
            client_id = context.node_id
        else:
            client_id = 0
        
        # Load hospital data
        train_dataset, test_dataset = load_hospital_data(client_id)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False
        )
        
        # Create client with FedBN enabled but DP disabled
        client_config = config.copy()
        client_config['privacy'] = {'enabled': False}  # No DP for FedBN-only
        client_config['federated'] = {'strategy': 'fedbn'}  # Explicitly FedBN
        
        client = FlowerClient(
            cid=f'hospital_{client_id}',
            train_loader=train_loader,
            val_loader=test_loader,
            config=client_config,
            fedbn=True  # Enable FedBN
        )
        
        return client.to_client()
    
    return client_fn


def evaluate_global_model(server_round, parameters, config):
    """Evaluate the global model on the held-out test set."""
    # Load global test set
    X_test, _, y_test, _ = load_processed_data(config['data']['processed_dir'])
    
    # Convert to tensors
    X_test_tensor = torch.from_numpy(X_test.values.astype(np.float32))
    y_test_tensor = torch.from_numpy(y_test.values.astype(np.float32))
    
    # Create model and load parameters
    model = CardioMLP(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )
    
    # Parameters are already ndarrays from the strategy
    params = parameters if isinstance(parameters, list) else parameters
    
    # Get shared parameter names
    shared_names = [name for name, _ in model.named_parameters() 
                   if not any(bn in name for bn in config['model']['bn_layers'])]
    
    # Update model state dict with shared parameters
    state_dict = model.state_dict()
    for name, param in zip(shared_names, params):
        state_dict[name] = torch.from_numpy(param) if isinstance(param, np.ndarray) else param
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Evaluate
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.sigmoid(logits).squeeze().numpy()
    
    metrics = binary_metrics(y_test_tensor.numpy().astype(int), probs)
    
    return metrics['roc_auc'], metrics


def run_fedbn_experiment(
    num_rounds: int = 10,
    num_clients: int = 3,
    config_path: str = 'config.yaml'
):
    """
    Run FedBN-only experiment.
    
    Args:
        num_rounds: Number of federated learning rounds
        num_clients: Number of hospitals
        config_path: Path to configuration file
        
    Returns:
        Tuple of (history, results_dict)
    """
    print("=" * 80)
    print("EXPERIMENT 03: FedBN ONLY (NO TRUST WEIGHTING)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Clients: {num_clients} hospitals")
    print(f"  - Rounds: {num_rounds}")
    print(f"  - Strategy: FedBN (local BatchNorm, shared Linear layers)")
    print(f"  - Aggregation: Standard FedAvg (volume-weighted)")
    print(f"  - Privacy: Disabled")
    print("=" * 80)
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Set random seeds
    seed = config['data']['random_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create initial model
    initial_model = CardioMLP(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )
    
    # Extract shared parameters for initialization
    shared_params = [p.detach().cpu().numpy() for name, p in initial_model.named_parameters()
                    if not any(bn in name for bn in config['model']['bn_layers'])]
    
    from flwr.common import ndarrays_to_parameters
    initial_parameters = ndarrays_to_parameters(shared_params)
    
    print(f"\nInitialized global model with {len(shared_params)} shared parameter arrays")
    
    # Create evaluation function
    def eval_fn(server_round, parameters, config_dict):
        loss, metrics = evaluate_global_model(server_round, parameters, config)
        print(f"\n[Round {server_round}] Global Test Set:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        return loss, metrics
    
    # Create strategy (standard FedAvg for FedBN)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=eval_fn
    )
    
    # Create client factory
    client_fn = make_client_fn(config)
    
    # Run simulation
    print(f"\nStarting FedBN simulation...\n")
    history = start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={'num_cpus': 1, 'num_gpus': 0.0}
    )
    
    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    
    # Evaluate final model on each hospital's local test set
    print("\nEvaluating final model on each hospital's local test set...")
    local_results = {}
    
    final_params = strategy.parameters_aggregated[-1] if hasattr(strategy, 'parameters_aggregated') else None
    if final_params is None:
        print("Warning: Could not retrieve final parameters")
        final_params = initial_parameters
    
    for hospital_id in range(num_clients):
        _, test_dataset = load_hospital_data(hospital_id)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False
        )
        
        # Create model and load parameters
        model = CardioMLP(
            input_dim=config['model']['input_dim'],
            hidden_dims=config['model']['hidden_dims'],
            dropout=config['model']['dropout']
        )
        
        from flwr.common import parameters_to_ndarrays
        params = parameters_to_ndarrays(final_params)
        
        shared_names = [name for name, _ in model.named_parameters()
                       if not any(bn in name for bn in config['model']['bn_layers'])]
        
        state_dict = model.state_dict()
        for name, param in zip(shared_names, params):
            state_dict[name] = torch.from_numpy(param)
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # Evaluate
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                logits = model(X_batch)
                probs = torch.sigmoid(logits).squeeze()
                all_probs.extend(probs.numpy())
                all_labels.extend(y_batch.numpy())
        
        metrics = binary_metrics(np.array(all_labels, dtype=int), np.array(all_probs))
        local_results[f'hospital_{hospital_id}'] = metrics
        
        print(f"\n  Hospital {hospital_id}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"    F1: {metrics['f1']:.4f}")
        print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"    Majority Baseline: {metrics['majority_baseline']:.4f}")
    
    # Aggregate results
    results = {
        'experiment': 'fedbn_only',
        'num_rounds': num_rounds,
        'num_clients': num_clients,
        'config': {
            'model': config['model'],
            'training': config['training'],
            'federated': {'strategy': 'fedbn', 'trust': False},
            'privacy': {'enabled': False}
        },
        'rounds': list(range(1, num_rounds + 1)),
        'global': {},
        'local': {
            'per_hospital': local_results,
            'average': {}
        }
    }
    
    # Extract global metrics per round from history
    if hasattr(history, 'metrics_centralized') and history.metrics_centralized:
        for round_num in range(1, num_rounds + 1):
            round_metrics = {}
            for metric_name, values in history.metrics_centralized.items():
                for rnd, value in values:
                    if rnd == round_num:
                        round_metrics[metric_name] = value
            if round_metrics:
                results['global'][str(round_num)] = round_metrics
    
    # Compute average local metrics
    if local_results:
        avg_metrics = {}
        for metric in ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc']:
            values = [m[metric] for m in local_results.values() if metric in m]
            if values:
                avg_metrics[metric] = float(np.mean(values))
        results['local']['average'] = avg_metrics
    
    # Save results
    os.makedirs('outputs/results', exist_ok=True)
    results_path = 'outputs/results/fedbn_only.json'
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    
    # Save model
    os.makedirs('outputs/models', exist_ok=True)
    model_path = 'outputs/models/fedbn_only.pth'
    
    # Save the final global model state
    final_model = CardioMLP(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout']
    )
    
    params = parameters_to_ndarrays(final_params)
    shared_names = [name for name, _ in final_model.named_parameters()
                   if not any(bn in name for bn in config['model']['bn_layers'])]
    
    state_dict = final_model.state_dict()
    for name, param in zip(shared_names, params):
        state_dict[name] = torch.from_numpy(param)
    
    torch.save(state_dict, model_path)
    print(f"✓ Model saved to {model_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 03 COMPLETE")
    print("=" * 80)
    
    return history, results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FedBN Only experiment')
    parser.add_argument('--rounds', type=int, default=10, help='Number of FL rounds')
    parser.add_argument('--clients', type=int, default=3, help='Number of clients')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    history, results = run_fedbn_experiment(
        num_rounds=args.rounds,
        num_clients=args.clients,
        config_path=args.config
    )
