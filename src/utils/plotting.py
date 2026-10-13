"""
Visualization utilities for federated learning experiments.

Generates plots for:
- Convergence analysis (accuracy, ROC-AUC over rounds)
- Data distribution across hospitals
- Trust score heatmaps
- Hospital-wise performance comparisons
- Privacy-utility tradeoffs
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_convergence(
    results: Dict[str, dict],
    metrics: List[str] = ['roc_auc', 'accuracy', 'f1'],
    output_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    Plot convergence of metrics over federated learning rounds.
    
    Args:
        results: Dictionary mapping strategy names to their results
        metrics: List of metric names to plot
        output_path: Path to save the figure
        title: Plot title
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for strategy_name, result in results.items():
            if 'rounds' not in result or 'global' not in result:
                continue
            
            rounds = result['rounds']
            values = []
            
            # Extract metric values per round
            for r in rounds:
                round_key = str(r)
                if round_key in result['global']:
                    val = result['global'][round_key].get(metric, None)
                    if val is not None:
                        values.append(val)
                    else:
                        values.append(np.nan)
            
            if values:
                ax.plot(rounds[:len(values)], values, marker='o', linewidth=2, 
                       markersize=6, label=strategy_name, alpha=0.8)
        
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.replace("_", " ").title()} Convergence', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved convergence plot to {output_path}")
    
    plt.close()


def plot_data_distribution(
    data_dir: str = 'data/processed',
    num_clients: int = 3,
    output_path: Optional[str] = None
):
    """
    Visualize class distribution across hospitals (Non-IID analysis).
    
    Args:
        data_dir: Directory containing hospital data files
        num_clients: Number of hospitals
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Collect data
    hospital_data = []
    for i in range(num_clients):
        train_path = os.path.join(data_dir, f'hospital_{i}_train.csv')
        if os.path.exists(train_path):
            df = pd.read_csv(train_path)
            counts = df['cardio'].value_counts().sort_index()
            total = len(df)
            pos_rate = counts.get(1, 0) / total
            
            hospital_data.append({
                'hospital': f'Hospital {i}',
                'negative': counts.get(0, 0),
                'positive': counts.get(1, 0),
                'total': total,
                'pos_rate': pos_rate
            })
    
    df_dist = pd.DataFrame(hospital_data)
    
    # Plot 1: Stacked bar chart
    ax1 = axes[0]
    x = np.arange(len(df_dist))
    width = 0.6
    
    p1 = ax1.bar(x, df_dist['negative'], width, label='Negative (0)', 
                 color='#3498db', alpha=0.8)
    p2 = ax1.bar(x, df_dist['positive'], width, bottom=df_dist['negative'],
                 label='Positive (1)', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Hospital', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax1.set_title('Class Distribution Across Hospitals', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_dist['hospital'])
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add total counts on top
    for i, total in enumerate(df_dist['total']):
        ax1.text(i, total + 500, f'n={total}', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    # Plot 2: Positive rate comparison
    ax2 = axes[1]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    bars = ax2.bar(x, df_dist['pos_rate'], width, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, 
               label='Balanced (50%)')
    ax2.set_xlabel('Hospital', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Positive Class Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Label Skew (Non-IID)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_dist['hospital'])
    ax2.set_ylim(0, 1.0)
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add percentage labels
    for i, rate in enumerate(df_dist['pos_rate']):
        ax2.text(i, rate + 0.03, f'{rate*100:.1f}%', ha='center', va='bottom',
                fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved data distribution plot to {output_path}")
    
    plt.close()


def plot_trust_heatmap(
    trust_scores: Dict[int, Dict[str, float]],
    output_path: Optional[str] = None
):
    """
    Plot heatmap of trust scores across rounds and clients.
    
    Args:
        trust_scores: Nested dict {round: {client_id: trust_score}}
        output_path: Path to save the figure
    """
    # Convert to DataFrame
    data = []
    for round_num, scores in trust_scores.items():
        for client_id, score in scores.items():
            data.append({
                'Round': int(round_num),
                'Client': client_id,
                'Trust': score
            })
    
    df = pd.DataFrame(data)
    
    # Pivot for heatmap
    pivot = df.pivot(index='Client', columns='Round', values='Trust')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)), max(6, len(pivot))))
    
    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Trust Score'},
                linewidths=0.5, linecolor='gray', ax=ax, vmin=0, vmax=1)
    
    ax.set_title('Update Consistency Trust Scores Over Rounds', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Federated Learning Round', fontsize=12, fontweight='bold')
    ax.set_ylabel('Client ID', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved trust heatmap to {output_path}")
    
    plt.close()


def plot_hospital_performance(
    results: Dict[str, dict],
    metric: str = 'balanced_accuracy',
    output_path: Optional[str] = None
):
    """
    Compare performance across hospitals for different strategies.
    
    Args:
        results: Dictionary mapping strategy names to their results
        metric: Metric to plot
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect data
    data = []
    for strategy_name, result in results.items():
        if 'local' not in result or 'per_hospital' not in result['local']:
            continue
        
        for hospital, metrics in result['local']['per_hospital'].items():
            if metric in metrics:
                data.append({
                    'Strategy': strategy_name,
                    'Hospital': hospital,
                    'Value': metrics[metric],
                    'Baseline': metrics.get('majority_baseline', 0.5)
                })
    
    if not data:
        print("Warning: No hospital performance data found")
        return
    
    df = pd.DataFrame(data)
    
    # Group by strategy and hospital
    hospitals = sorted(df['Hospital'].unique())
    strategies = sorted(df['Strategy'].unique())
    
    x = np.arange(len(hospitals))
    width = 0.8 / len(strategies)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    
    for idx, strategy in enumerate(strategies):
        strategy_data = df[df['Strategy'] == strategy]
        values = [strategy_data[strategy_data['Hospital'] == h]['Value'].values[0] 
                 if h in strategy_data['Hospital'].values else 0 
                 for h in hospitals]
        
        offset = (idx - len(strategies)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=strategy, color=colors[idx], 
              alpha=0.8, edgecolor='black', linewidth=1)
    
    # Plot majority baselines
    for i, hospital in enumerate(hospitals):
        hospital_data = df[df['Hospital'] == hospital]
        if not hospital_data.empty:
            baseline = hospital_data['Baseline'].values[0]
            ax.plot([x[i] - 0.4, x[i] + 0.4], [baseline, baseline], 
                   color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Create custom legend entry for baseline
    from matplotlib.lines import Line2D
    baseline_line = Line2D([0], [0], color='red', linestyle='--', linewidth=2, 
                          label='Majority Baseline')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(baseline_line)
    labels.append('Majority Baseline')
    
    ax.set_xlabel('Hospital', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_title(f'{metric.replace("_", " ").title()} by Hospital and Strategy', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(hospitals)
    ax.legend(handles, labels, loc='best', frameon=True, shadow=True)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved hospital performance plot to {output_path}")
    
    plt.close()


def plot_comparison_bars(
    results: Dict[str, dict],
    metrics: List[str] = ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc'],
    output_path: Optional[str] = None
):
    """
    Create grouped bar chart comparing final metrics across strategies.
    
    Args:
        results: Dictionary mapping strategy names to their results
        metrics: List of metrics to compare
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect final metrics
    data = []
    for strategy_name, result in results.items():
        metric_values = {}
        
        # Try to get final global metrics
        if 'final' in result:
            metric_values = result['final']
        elif 'best' in result:
            metric_values = result['best']
        elif 'global' in result:
            # Get last round
            rounds = sorted([int(k) for k in result['global'].keys() if k.isdigit()])
            if rounds:
                metric_values = result['global'][str(rounds[-1])]
        
        for metric in metrics:
            if metric in metric_values:
                data.append({
                    'Strategy': strategy_name,
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': metric_values[metric]
                })
    
    if not data:
        print("Warning: No comparison data found")
        return
    
    df = pd.DataFrame(data)
    
    # Group by metric
    metrics_list = sorted(df['Metric'].unique())
    strategies = sorted(df['Strategy'].unique())
    
    x = np.arange(len(metrics_list))
    width = 0.8 / len(strategies)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
    
    for idx, strategy in enumerate(strategies):
        strategy_data = df[df['Strategy'] == strategy]
        values = [strategy_data[strategy_data['Metric'] == m]['Value'].values[0] 
                 if m in strategy_data['Metric'].values else 0 
                 for m in metrics_list]
        
        offset = (idx - len(strategies)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=strategy, color=colors[idx], 
              alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Strategy Comparison: Final Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list, rotation=15, ha='right')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison bar chart to {output_path}")
    
    plt.close()


def plot_privacy_utility_tradeoff(
    sweep_results: List[dict],
    output_path: Optional[str] = None
):
    """
    Plot privacy-utility tradeoff from sigma sweep.
    
    Args:
        sweep_results: List of results with 'sigma', 'epsilon', and metrics
        output_path: Path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    df = pd.DataFrame(sweep_results)
    
    # Plot 1: Utility vs Sigma
    ax1 = axes[0]
    metrics_to_plot = ['roc_auc', 'balanced_accuracy', 'accuracy']
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            ax1.plot(df['sigma'], df[metric], marker='o', linewidth=2, 
                    markersize=8, label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax1.set_xlabel('Noise Scale (σ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance', fontsize=12, fontweight='bold')
    ax1.set_title('Utility vs Privacy Noise', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.legend(loc='best', frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Utility vs Epsilon
    ax2 = axes[1]
    
    for metric in metrics_to_plot:
        if metric in df.columns and 'epsilon' in df.columns:
            ax2.plot(df['epsilon'], df[metric], marker='o', linewidth=2,
                    markersize=8, label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax2.set_xlabel('Privacy Budget (ε)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance', fontsize=12, fontweight='bold')
    ax2.set_title('Privacy-Utility Tradeoff', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(loc='best', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Add privacy strength regions
    ax2.axvspan(0, 1, alpha=0.1, color='green', label='Strong Privacy')
    ax2.axvspan(1, 10, alpha=0.1, color='yellow', label='Moderate Privacy')
    ax2.axvspan(10, ax2.get_xlim()[1], alpha=0.1, color='red', label='Weak Privacy')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved privacy-utility tradeoff plot to {output_path}")
    
    plt.close()


def plot_sweep_analysis(
    sweep_results: List[dict],
    sweep_param: str,
    output_path: Optional[str] = None
):
    """
    Generic sweep analysis plotter (for alpha, beta sweeps).
    
    Args:
        sweep_results: List of results with sweep parameter and metrics
        sweep_param: Name of parameter being swept ('alpha', 'beta', etc.)
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = pd.DataFrame(sweep_results)
    
    if sweep_param not in df.columns:
        print(f"Warning: {sweep_param} not found in sweep results")
        return
    
    # Plot multiple metrics
    metrics_to_plot = ['roc_auc', 'balanced_accuracy', 'accuracy', 'f1']
    
    for metric in metrics_to_plot:
        if metric in df.columns:
            ax.plot(df[sweep_param], df[metric], marker='o', linewidth=2,
                   markersize=8, label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xlabel(sweep_param.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('Performance', fontsize=12, fontweight='bold')
    ax.set_title(f'Performance vs {sweep_param.title()}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {sweep_param} sweep plot to {output_path}")
    
    plt.close()


if __name__ == '__main__':
    print("=" * 80)
    print("PLOTTING UTILITIES TEST")
    print("=" * 80)
    
    # Create output directory
    os.makedirs('outputs/graphs', exist_ok=True)
    
    # Test 1: Data distribution plot
    print("\n1. Testing data distribution plot...")
    plot_data_distribution(
        data_dir='data/processed',
        num_clients=3,
        output_path='outputs/graphs/data_distribution.png'
    )
    
    # Test 2: Create dummy results for testing other plots
    print("\n2. Testing convergence plot...")
    dummy_results = {
        'FedAvg': {
            'rounds': list(range(1, 11)),
            'global': {
                str(i): {
                    'accuracy': 0.65 + i * 0.02,
                    'roc_auc': 0.70 + i * 0.015,
                    'f1': 0.68 + i * 0.018
                } for i in range(1, 11)
            }
        },
        'AT-FedBN': {
            'rounds': list(range(1, 11)),
            'global': {
                str(i): {
                    'accuracy': 0.67 + i * 0.022,
                    'roc_auc': 0.72 + i * 0.017,
                    'f1': 0.69 + i * 0.019
                } for i in range(1, 11)
            }
        }
    }
    
    plot_convergence(
        results=dummy_results,
        metrics=['roc_auc', 'accuracy'],
        output_path='outputs/graphs/test_convergence.png'
    )
    
    # Test 3: Trust heatmap
    print("\n3. Testing trust heatmap...")
    dummy_trust = {
        i: {
            'hospital_0': 0.8 + np.random.rand() * 0.15,
            'hospital_1': 0.7 + np.random.rand() * 0.2,
            'hospital_2': 0.75 + np.random.rand() * 0.18
        } for i in range(1, 11)
    }
    
    plot_trust_heatmap(
        trust_scores=dummy_trust,
        output_path='outputs/graphs/test_trust_heatmap.png'
    )
    
    print("\n" + "=" * 80)
    print("✓ ALL PLOTTING TESTS PASSED")
    print("=" * 80)
