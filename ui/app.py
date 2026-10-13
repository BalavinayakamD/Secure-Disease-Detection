"""
AT-FedBN Research Dashboard

A professional web interface for the Adaptive Trust-Weighted Federated Batch
Normalization project, showcasing federated learning with differential privacy
on cardiovascular disease prediction across non-IID hospital data.

Usage:
    streamlit run ui/app.py
"""

import json
import os
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import streamlit as st
import yaml

RESULTS_DIR = 'outputs/results'
GRAPHS_DIR = 'outputs/graphs'
DATA_DIR = 'data/processed'

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title='AT-FedBN Research Dashboard',
    page_icon='🏥',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ============================================================================
# Styling
# ============================================================================

st.markdown("""
<style>
    /* Clean, professional typography */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    h1 {
        color: #1f2937;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #374151;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #e5e7eb;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #4b5563;
        font-weight: 500;
        margin-top: 1.5rem;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    /* Tables */
    .dataframe {
        font-size: 0.95rem;
    }
    
    /* Remove excessive spacing */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Info boxes */
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 1.5rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Utility Functions
# ============================================================================

def load_json(filename):
    """Load a JSON result file if it exists."""
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def load_config():
    """Load the project configuration."""
    with open('config.yaml') as f:
        return yaml.safe_load(f)

def format_metric(value, format_type='float'):
    """Format a metric value for display."""
    if value is None:
        return '—'
    if format_type == 'float':
        return f'{value:.4f}'
    elif format_type == 'percent':
        return f'{value*100:.2f}%'
    elif format_type == 'int':
        return f'{int(value):,}'
    return str(value)

def get_image_path(filename):
    """Get the full path to a graph image."""
    path = os.path.join(GRAPHS_DIR, filename)
    return path if os.path.exists(path) else None

# ============================================================================
# Header
# ============================================================================

st.title('AT-FedBN: Adaptive Trust-Weighted Federated Batch Normalization')

st.markdown("""
<div style="background: #f9fafb; padding: 1.5rem; border-radius: 0.5rem; border-left: 4px solid #3b82f6; margin-bottom: 2rem;">
    <strong>Privacy-preserving federated learning for cardiovascular disease prediction</strong><br/>
    Combining trust-weighted aggregation, local batch normalization, and differential privacy
    to enable collaborative learning across hospitals without sharing patient data.
</div>
""", unsafe_allow_html=True)

# Load configuration
cfg = load_config()

# ============================================================================
# Main Navigation Tabs
# ============================================================================

tab_overview, tab_data, tab_training, tab_trust, tab_comparison, tab_demo = st.tabs([
    '📋 Overview',
    '🏥 Hospital Data',
    '🔄 Training Results',
    '🔒 Trust Mechanism',
    '📊 Model Comparison',
    '⚡ Live Demo'
])

# ============================================================================
# TAB 1: Overview
# ============================================================================

with tab_overview:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('## Research Objective')
        st.markdown("""
        This project addresses a critical challenge in healthcare AI: how can multiple
        hospitals collaboratively train a disease prediction model without sharing
        sensitive patient data?
        
        **Key Innovation:** AT-FedBN combines three techniques:
        - **Federated Learning**: Models train locally, only sharing parameter updates
        - **Trust-Weighted Aggregation**: Low-quality client updates receive lower weight
        - **Local BatchNorm**: Each hospital retains domain-specific normalization statistics
        - **Differential Privacy**: Noise protects individual patient privacy (optional)
        """)
        
        st.markdown('## System Architecture')
        st.code("""
Hospital Data (Non-IID, Privacy-Sensitive)
    ↓
Local Model Training (CardioMLP)
    ↓ (2 epochs per round)
Extract Shared Parameters (fc1, fc2, fc3 weights)
    ↓
Apply Differential Privacy (Gaussian noise, σ=0.01)
    ↓
Send to Central Server
    ↓
Calculate Trust Scores (cosine similarity with global trend)
    ↓
Hybrid Aggregation (β·volume + (1-β)·trust, β=0.7)
    ↓
Broadcast Updated Global Parameters
    ↓
Merge with Local BatchNorm (stays on hospital side)
    ↓
Next Training Round
        """, language='text')
    
    with col2:
        st.markdown('## Project Status')
        
        # Load summary statistics
        summary = load_json('summary_statistics.json')
        
        if summary:
            st.metric('Experiments Complete', summary.get('experiments_completed', 0))
            st.metric('Training Rounds', summary.get('federated_config', {}).get('rounds', 0))
            st.metric('Hospitals', summary.get('dataset_stats', {}).get('hospitals', 0))
            st.metric('Total Samples', 
                     format_metric(summary.get('dataset_stats', {}).get('total_samples', 0), 'int'))
        
        st.markdown('### Model Architecture')
        st.markdown(f"""
        - **Input Dimension**: {cfg['model']['input_dim']} features
        - **Hidden Layers**: {cfg['model']['hidden_dims']}
        - **Dropout**: {cfg['model']['dropout']}
        - **Batch Norm Layers**: {', '.join(cfg['model']['bn_layers'])}
        """)
        
        st.markdown('### Training Configuration')
        st.markdown(f"""
        - **Local Epochs**: {cfg['training']['local_epochs']} per round
        - **Batch Size**: {cfg['training']['batch_size']}
        - **Learning Rate**: {cfg['training']['learning_rate']}
        - **DP Noise (σ)**: {cfg['privacy']['noise_multiplier']}
        - **Gradient Clip (C)**: {cfg['privacy']['max_grad_norm']}
        """)
    
    # Results Summary Table
    st.markdown('## Performance Summary')
    
    master_table_path = os.path.join(RESULTS_DIR, 'master_table.md')
    if os.path.exists(master_table_path):
        with open(master_table_path) as f:
            st.markdown(f.read())
    else:
        st.info('Run experiments to generate results table.')
    
    # Key Findings
    summary = load_json('summary_statistics.json')
    if summary and 'performance_summary' in summary:
        st.markdown('## Key Findings')
        
        perf = summary['performance_summary']
        central = perf.get('centralized', {})
        at_fedbn = perf.get('at_fedbn', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                'ROC-AUC',
                format_metric(at_fedbn.get('roc_auc')),
                delta=format_metric(at_fedbn.get('roc_auc', 0) - central.get('roc_auc', 0)),
                help='AT-FedBN vs. Centralized baseline'
            )
        
        with col2:
            st.metric(
                'Balanced Accuracy',
                format_metric(at_fedbn.get('balanced_accuracy')),
                delta=format_metric(at_fedbn.get('balanced_accuracy', 0) - central.get('balanced_accuracy', 0)),
                help='AT-FedBN vs. Centralized baseline'
            )
        
        with col3:
            st.metric(
                'F1 Score',
                format_metric(at_fedbn.get('f1')),
                delta=format_metric(at_fedbn.get('f1', 0) - central.get('f1', 0)),
                help='AT-FedBN vs. Centralized baseline'
            )
        
        with col4:
            st.metric(
                'Accuracy',
                format_metric(at_fedbn.get('accuracy')),
                delta=format_metric(at_fedbn.get('accuracy', 0) - central.get('accuracy', 0)),
                help='AT-FedBN vs. Centralized baseline'
            )
        
        st.markdown("""
        **Interpretation**: AT-FedBN achieves comparable (or in some metrics, superior) performance
        to centralized training while maintaining patient privacy through local training and
        differential privacy.
        """)

# ============================================================================
# TAB 2: Hospital Data Distribution
# ============================================================================

with tab_data:
    st.markdown('## Non-IID Data Distribution')
    
    st.markdown("""
    Real-world hospital data is **non-IID** (not independent and identically distributed).
    Different hospitals see different patient populations, leading to varying disease
    prevalence and feature distributions.
    """)
    
    # Distribution visualization
    dist_img = get_image_path('data_distribution.png')
    if dist_img:
        st.image(dist_img, use_container_width=True, 
                caption='Class distribution across hospitals (Dirichlet α=1.0)')
    
    # Hospital statistics table
    st.markdown('### Hospital-Level Statistics')
    
    rows = []
    for i in range(cfg['data']['num_clients']):
        train_file = os.path.join(DATA_DIR, f'hospital_{i}_train.csv')
        test_file = os.path.join(DATA_DIR, f'hospital_{i}_test.csv')
        
        if os.path.exists(train_file) and os.path.exists(test_file):
            df_train = pd.read_csv(train_file)
            df_test = pd.read_csv(test_file)
            
            rows.append({
                'Hospital': f'Hospital {i}',
                'Train Samples': len(df_train),
                'Test Samples': len(df_test),
                'Total': len(df_train) + len(df_test),
                'Positive Rate (Train)': f'{df_train.cardio.mean():.1%}',
                'Positive Rate (Test)': f'{df_test.cardio.mean():.1%}',
                'Skew': 'High' if abs(df_train.cardio.mean() - 0.5) > 0.2 else 'Moderate'
            })
    
    if rows:
        df_hospitals = pd.DataFrame(rows)
        st.dataframe(df_hospitals, use_container_width=True, hide_index=True)
        
        st.markdown("""
        **Non-IID Challenge**: Notice the varying positive rates across hospitals. A model
        trained only on Hospital 1's high-prevalence population may perform poorly on
        Hospital 2's different distribution. FedBN addresses this by keeping batch
        normalization statistics local to each hospital.
        """)
    
    # Local performance by hospital
    st.markdown('### Local Model Performance')
    
    local_perf_img = get_image_path('local_balanced_accuracy_by_hospital.png')
    if local_perf_img:
        st.image(local_perf_img, use_container_width=True,
                caption='Each bar shows balanced accuracy minus the majority baseline')
    else:
        st.info('Run experiments to generate hospital-wise performance comparison.')

# ============================================================================
# TAB 3: Training Results
# ============================================================================

with tab_training:
    st.markdown('## Federated Training Convergence')
    
    # Convergence plots
    col1, col2 = st.columns(2)
    
    with col1:
        conv_roc = get_image_path('convergence_roc_auc.png')
        if conv_roc:
            st.image(conv_roc, use_container_width=True, caption='ROC-AUC over training rounds')
        else:
            st.info('Run experiments to generate convergence plots.')
    
    with col2:
        conv_bal = get_image_path('convergence_balanced_accuracy.png')
        if conv_bal:
            st.image(conv_bal, use_container_width=True, 
                    caption='Balanced Accuracy over training rounds')
    
    # All metrics convergence
    st.markdown('### Complete Metrics Progression')
    conv_all = get_image_path('convergence_all_metrics.png')
    if conv_all:
        st.image(conv_all, use_container_width=True)
    
    # Detailed results
    st.markdown('### Detailed Training Results')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('#### FedBN Only')
        fedbn_data = load_json('fedbn_only.json')
        if fedbn_data and 'local' in fedbn_data:
            local_results = fedbn_data['local']
            st.metric('Global ROC-AUC', format_metric(local_results.get('global', {}).get('roc_auc')))
            st.metric('Avg Hospital ROC-AUC', format_metric(local_results.get('mean_roc_auc')))
            st.metric('Training Rounds', fedbn_data.get('num_rounds', 'N/A'))
        else:
            st.info('Run Experiment 03 (FedBN Only)')
    
    with col2:
        st.markdown('#### AT-FedBN (Full System)')
        at_fedbn_data = load_json('at_fedbn.json')
        if at_fedbn_data and 'local' in at_fedbn_data:
            local_results = at_fedbn_data['local']
            st.metric('Global ROC-AUC', format_metric(local_results.get('global', {}).get('roc_auc')))
            st.metric('Avg Hospital ROC-AUC', format_metric(local_results.get('mean_roc_auc')))
            st.metric('Training Rounds', at_fedbn_data.get('num_rounds', 'N/A'))
            st.metric('Trust Beta', format_metric(at_fedbn_data.get('beta', 0.7)))
        else:
            st.info('Run Experiment 04 (AT-FedBN)')

# ============================================================================
# TAB 4: Trust Mechanism
# ============================================================================

with tab_trust:
    st.markdown('## Adaptive Trust-Weighted Aggregation')
    
    st.markdown("""
    **Problem**: In federated learning, not all client updates are equally valuable.
    Some hospitals may have poor data quality, adversarial behavior, or convergence issues.
    
    **Solution**: AT-FedBN calculates a trust score for each client based on how well
    their parameter update aligns with the global training direction (measured by cosine
    similarity).
    
    **Formula**:
    ```
    Trust_i = ReLU(CosineSimilarity(ΔW_i, ΔW_global))
    Weight_i = β · (n_i / N) + (1 - β) · (Trust_i / Σ Trust_j)
    ```
    
    where β=0.7 balances data volume and trust.
    """)
    
    # Trust heatmap
    trust_heatmap = get_image_path('trust_heatmap.png')
    if trust_heatmap:
        st.image(trust_heatmap, use_container_width=True,
                caption='Trust scores across training rounds')
    else:
        st.info('Run Experiment 04 (AT-FedBN) to generate trust visualization.')
    
    # Final round weights
    st.markdown('### Final Round Aggregation Weights')
    
    at_fedbn_data = load_json('at_fedbn.json')
    if at_fedbn_data and 'weights' in at_fedbn_data:
        final_round = str(at_fedbn_data['num_rounds'])
        if final_round in at_fedbn_data['weights']:
            weights_data = at_fedbn_data['weights'][final_round]
            
            rows = []
            for client_id, metrics in weights_data.items():
                rows.append({
                    'Client': client_id[:16] + '...',
                    'Samples': metrics['num_examples'],
                    'Cosine Similarity': format_metric(metrics['cosine']),
                    'Volume Weight': format_metric(metrics['volume']),
                    'Trust Weight': format_metric(metrics['trust']),
                    'Final Weight': format_metric(metrics['final'])
                })
            
            df_weights = pd.DataFrame(rows)
            st.dataframe(df_weights, use_container_width=True, hide_index=True)
            
            st.caption(f"β = {at_fedbn_data['beta']} | Weights sum to 1.0")
            
            st.markdown("""
            **Interpretation**: Clients with higher trust scores (cosine similarity) and
            more training samples receive larger aggregation weights. This protects against
            low-quality or malicious updates while still considering data volume.
            """)
    else:
        st.info('Run Experiment 04 to see trust-based aggregation weights.')

# ============================================================================
# TAB 5: Model Comparison
# ============================================================================

with tab_comparison:
    st.markdown('## Comprehensive Model Comparison')
    
    # Bar chart comparison
    comp_bars = get_image_path('final_comparison_bars.png')
    if comp_bars:
        st.image(comp_bars, use_container_width=True)
    
    # Detailed metrics table
    st.markdown('### Detailed Performance Metrics')
    
    central = load_json('central_baseline.json')
    fedbn = load_json('fedbn_only.json')
    at_fedbn = load_json('at_fedbn.json')
    
    if all([central, fedbn, at_fedbn]):
        rows = []
        
        # Centralized
        rows.append({
            'Strategy': 'Centralized',
            'Accuracy': format_metric(central['best']['accuracy']),
            'Balanced Acc': format_metric(central['best']['balanced_accuracy']),
            'F1': format_metric(central['best']['f1']),
            'ROC-AUC': format_metric(central['best']['roc_auc']),
            'Privacy': '❌ No',
            'Collaboration': '❌ Requires data sharing'
        })
        
        # FedBN
        rows.append({
            'Strategy': 'FedBN',
            'Accuracy': format_metric(fedbn['local']['global']['accuracy']),
            'Balanced Acc': format_metric(fedbn['local']['global']['balanced_accuracy']),
            'F1': format_metric(fedbn['local']['global']['f1']),
            'ROC-AUC': format_metric(fedbn['local']['global']['roc_auc']),
            'Privacy': '✅ Local training',
            'Collaboration': '✅ Federated'
        })
        
        # AT-FedBN
        rows.append({
            'Strategy': 'AT-FedBN',
            'Accuracy': format_metric(at_fedbn['local']['global']['accuracy']),
            'Balanced Acc': format_metric(at_fedbn['local']['global']['balanced_accuracy']),
            'F1': format_metric(at_fedbn['local']['global']['f1']),
            'ROC-AUC': format_metric(at_fedbn['local']['global']['roc_auc']),
            'Privacy': f"✅ Local + DP (σ={cfg['privacy']['noise_multiplier']})",
            'Collaboration': '✅ Federated + Trust'
        })
        
        df_comp = pd.DataFrame(rows)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    # Privacy-Utility Tradeoff
    st.markdown('### Privacy-Utility Tradeoff')
    
    privacy_plot = get_image_path('privacy_utility.png')
    if privacy_plot:
        st.image(privacy_plot, use_container_width=True)
        
        sigma_sweep = load_json('sigma_sweep.json')
        if sigma_sweep:
            st.markdown('#### Differential Privacy Analysis')
            
            rows = []
            for item in sigma_sweep['sweep']:
                epsilon = item['epsilon']
                if epsilon is None:
                    eps_str = '∞ (No DP)'
                    level = 'None'
                elif epsilon >= 100:
                    eps_str = f'{epsilon:,.0f}'
                    level = '❌ Very Weak'
                elif epsilon >= 10:
                    eps_str = f'{epsilon:.1f}'
                    level = '⚠️ Weak'
                else:
                    eps_str = f'{epsilon:.2f}'
                    level = '✅ Moderate' if epsilon > 1 else '✅✅ Strong'
                
                rows.append({
                    'Noise σ': item['sigma'],
                    'Privacy ε': eps_str,
                    'Privacy Level': level,
                    'ROC-AUC': format_metric(item['local']['roc_auc']),
                    'Accuracy': format_metric(item['local']['accuracy'])
                })
            
            df_privacy = pd.DataFrame(rows)
            st.dataframe(df_privacy, use_container_width=True, hide_index=True)
            
            st.warning(f"""
            **Current Configuration**: σ={cfg['privacy']['noise_multiplier']} provides very weak formal privacy
            (ε ≈ 55,000). This is a deliberate design choice prioritizing utility for research
            demonstration. For deployment, increase σ substantially (e.g., σ=0.5-1.0) to achieve
            meaningful privacy guarantees, accepting some accuracy reduction.
            """)
    else:
        st.info('Run sigma sweep experiment to analyze privacy-utility tradeoff.')

# ============================================================================
# TAB 6: Live Demo
# ============================================================================

with tab_demo:
    st.markdown('## Interactive Training Demo')
    
    st.markdown("""
    Run a short 2-round AT-FedBN training session on the real hospital data.
    This demonstrates the complete system: local training, trust calculation,
    and aggregation.
    """)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        beta_demo = st.slider(
            'Trust Weight (β)',
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help='Balance between volume (β=1) and trust (β=0)'
        )
        
        run_demo = st.button('🚀 Run 2-Round Training', type='primary')
    
    with col2:
        st.info(f"""
        **Configuration**: 2 rounds, β={beta_demo}
        
        - β=0.0: Pure trust-based aggregation
        - β=0.7: Hybrid (70% volume, 30% trust)
        - β=1.0: Pure volume-based (standard FedAvg)
        
        Training takes approximately 20-30 seconds.
        """)
    
    if run_demo:
        with st.spinner('Training in progress...'):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    'exp04', 'experiments/04_full_at_fedbn.py'
                )
                exp04 = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(exp04)
                
                strategy, results, payload = exp04.run_at_fedbn(
                    num_rounds=2,
                    beta=beta_demo,
                    tag='ui_live_demo'
                )
                
                st.success('✅ Training complete!')
                
                # Trust scores per round
                st.markdown('### Trust Scores by Round')
                trust_rows = []
                for round_num, scores in sorted(payload['trust_scores'].items()):
                    for client_id, score in scores.items():
                        trust_rows.append({
                            'Round': round_num,
                            'Client': client_id[:16] + '...',
                            'Trust Score': format_metric(score)
                        })
                
                df_trust = pd.DataFrame(trust_rows)
                st.dataframe(df_trust, use_container_width=True, hide_index=True)
                
                # Hospital performance
                st.markdown('### Final Model Performance by Hospital')
                hosp_rows = []
                for hospital, metrics in results['local']['per_hospital'].items():
                    hosp_rows.append({
                        'Hospital': hospital,
                        'ROC-AUC': format_metric(metrics['roc_auc']),
                        'Balanced Acc': format_metric(metrics['balanced_accuracy']),
                        'Accuracy': format_metric(metrics['accuracy']),
                        'Majority Baseline': format_metric(metrics['majority_baseline'])
                    })
                
                df_hosp = pd.DataFrame(hosp_rows)
                st.dataframe(df_hosp, use_container_width=True, hide_index=True)
                
                st.markdown("""
                **Note**: These results are from a very short 2-round training session
                for demonstration purposes. Full experiment results (10 rounds) are
                available in the other tabs.
                """)
                
            except Exception as e:
                st.error(f'Training failed: {e}')
                st.exception(e)
    
    st.markdown('---')
    st.markdown('### Experiment Execution')
    st.markdown("""
    To run the complete experiment suite outside this dashboard:
    
    ```bash
    # Individual experiments
    python experiments/01_central_baseline.py     # Centralized upper bound
    python experiments/03_fedbn_only.py           # FedBN baseline
    python experiments/04_full_at_fedbn.py        # Full AT-FedBN
    
    # Generate comparisons and visualizations
    python experiments/compare_all.py
    
    # Parameter sweeps
    python experiments/05_sweep_experiments.py --sweep sigma
    python experiments/05_sweep_experiments.py --sweep beta
    python experiments/05_sweep_experiments.py --sweep alpha
    ```
    """)

# ============================================================================
# Footer
# ============================================================================

st.markdown('---')

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('**Project Information**')
    st.caption('AT-FedBN Research Project')
    st.caption('Federated Learning + Differential Privacy')

with col2:
    st.markdown('**Dataset**')
    st.caption('Cardiovascular Disease Dataset')
    st.caption('70,000 patient records across 3 hospitals')

with col3:
    st.markdown('**Quick Links**')
    st.caption('[Run Tests](command:python -m pytest tests/)')
    st.caption('[View Code](file:../src/)')

st.caption('Dashboard built with Streamlit | Data remains local to hospitals')
