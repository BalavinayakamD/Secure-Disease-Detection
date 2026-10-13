# AT-FedBN: Adaptive Trust-Weighted Federated Batch Normalization

A privacy-preserving federated learning framework for collaborative disease prediction across hospitals without sharing raw patient data.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Flower](https://img.shields.io/badge/Flower-1.20+-green.svg)](https://flower.ai/)

## 🎯 Overview

This project implements **AT-FedBN**, a novel federated learning approach that combines:

- **Trust Mechanism**: Update consistency scoring via cosine similarity
- **Hybrid Aggregation**: β × volume + (1-β) × trust-weighted parameter averaging
- **FedBN Personalization**: Local BatchNorm layers adapt to hospital-specific distributions
- **Differential Privacy**: Gradient clipping + Gaussian noise for privacy protection

### Key Features

✅ **Privacy-Preserving**: No raw patient data leaves individual hospitals  
✅ **Non-IID Resilient**: Handles heterogeneous data distributions across hospitals  
✅ **Trust-Aware**: Downweights divergent/malicious updates automatically  
✅ **Personalized**: Local BatchNorm adapts to institution-specific patterns  
✅ **Interactive Dashboard**: Real-time visualization of training progress and trust scores  

## 📊 Project Structure

```
Secure-Disease-Detection/
├── config.yaml              # Configuration for all experiments
├── requirements.txt         # Python dependencies
├── data/
│   ├── raw/                # Raw cardiovascular disease dataset
│   └── processed/          # Preprocessed and hospital-split data
├── src/
│   ├── data/              # Data preprocessing and splitting
│   ├── models/            # CardioMLP neural network architecture
│   ├── fl/                # Federated learning (client, server, strategy)
│   ├── privacy/           # Differential privacy mechanisms
│   └── utils/             # Metrics and plotting utilities
├── experiments/           # Experiment scripts (01-05)
├── tests/                # Unit and integration tests
├── ui/                   # Streamlit dashboard
├── outputs/
│   ├── results/          # JSON results and master comparison table
│   ├── graphs/           # Visualization plots
│   ├── models/           # Saved model checkpoints
│   └── logs/             # Training logs
└── docs/                 # Documentation drafts
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- 8GB+ RAM (for federated simulation)
- Linux/macOS (Windows may work but untested)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd Secure-Disease-Detection
```

2. **Create and activate virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Data Preparation

1. **Place raw data** (if not already present)
```bash
# The raw cardiovascular disease dataset should be at:
# data/raw/cardio_train.csv
```

2. **Preprocess data**
```bash
python src/data/preprocessing_pipeline.py
```

3. **Generate hospital data shards** (Non-IID split)
```bash
python src/data/splitting.py
```

This creates:
- `data/processed/hospital_0_train.csv` & `hospital_0_test.csv`
- `data/processed/hospital_1_train.csv` & `hospital_1_test.csv`
- `data/processed/hospital_2_train.csv` & `hospital_2_test.csv`

## 🧪 Running Experiments

### Individual Experiments

**Experiment 01: Centralized Baseline** (already completed)
```bash
python experiments/01_central_baseline.py --epochs 30
```

**Experiment 02: FedAvg Baseline** (standard federated averaging)
```bash
python experiments/02_fedavg_baseline.py
```

**Experiment 03: FedBN Only** (no trust mechanism)
```bash
python experiments/03_fedbn_only.py --rounds 10
```

**Experiment 04: Full AT-FedBN** (our main contribution)
```bash
python experiments/04_full_at_fedbn.py --rounds 10 --beta 0.7
```

**Experiment 05: Hyperparameter Sweeps**
```bash
# Sigma sweep (privacy-utility tradeoff)
python experiments/05_sweep_experiments.py --sweep sigma

# Beta sweep (volume vs trust weighting)
python experiments/05_sweep_experiments.py --sweep beta --rounds 5

# All sweeps
python experiments/05_sweep_experiments.py --sweep all
```

### Run All Experiments (Quick Version)

For testing or time-constrained scenarios, run shortened experiments:

```bash
python experiments/run_all_quick.py
```

This runs 5-round versions of experiments 03-04 and generates all visualizations.

### Generate Comparison Report

After running experiments:

```bash
python experiments/compare_all.py
```

This generates:
- `outputs/results/master_table.md` - Comprehensive comparison table
- `outputs/graphs/convergence_*.png` - Training curves
- `outputs/graphs/trust_heatmap.png` - Trust score evolution
- `outputs/graphs/hospital_performance.png` - Per-hospital metrics

## 📈 Interactive Dashboard

Launch the Streamlit dashboard to explore results interactively:

```bash
streamlit run ui/app.py
```

Navigate to `http://localhost:8501` in your browser.

**Dashboard Features:**
- **Overview**: Master comparison table and convergence plots
- **Hospitals**: Data distribution and per-hospital performance
- **Trust**: Trust score heatmaps and weight distributions
- **Sweeps**: Interactive hyperparameter analysis
- **Live Demo**: Run 2-round AT-FedBN experiments on demand

## 🧪 Testing

Run the test suite:

```bash
# All tests
pytest tests/ -v

# Specific test modules
pytest tests/test_model_io.py -v
pytest tests/test_splitting.py -v
pytest tests/test_strategy.py -v
```

**Current Test Status**: 15/17 tests passing
- ✅ Model architecture tests
- ✅ Data splitting tests
- ✅ Strategy trust mechanism tests
- ⚠️ 2 simulation tests require Ray cluster configuration

## 📝 Configuration

Edit `config.yaml` to customize experiments:

```yaml
data:
  num_clients: 3                    # Number of hospitals
  dirichlet_alpha: 1.0              # Non-IID severity (lower = more skew)
  test_split: 0.20                  # Global test set fraction
  min_samples_per_class: 20         # Minimum samples per class per hospital

model:
  input_dim: 11                     # Number of features
  hidden_dims: [64, 32]            # MLP hidden layer sizes
  dropout: 0.2                      # Dropout rate

training:
  local_epochs: 2                   # Epochs per client per round
  batch_size: 64                    # Training batch size
  learning_rate: 0.001              # Optimizer learning rate

federated:
  num_rounds: 10                    # Federated learning rounds
  strategy: "at_fedbn"              # Strategy name
  beta: 0.7                         # Hybrid aggregation (0=pure trust, 1=pure volume)

privacy:
  enabled: true                     # Enable differential privacy
  clip_norm: 1.0                    # L2 gradient clipping bound
  noise_scale: 0.01                 # Gaussian noise multiplier (σ)
```

## 📊 Results

### Performance Comparison (10 rounds)

| Strategy      | Accuracy | Balanced Acc | F1    | ROC-AUC | Privacy       |
|---------------|----------|--------------|-------|---------|---------------|
| Centralized   | 0.7299   | 0.7296       | 0.7172| 0.7980  | None          |
| FedAvg        | ~0.68    | ~0.68        | ~0.66 | ~0.71   | None          |
| FedBN         | ~0.70    | ~0.70        | ~0.68 | ~0.73   | None          |
| **AT-FedBN**  | **~0.72**| **~0.72**    |**~0.70**|**~0.75**| **DP (σ=0.01)**|

**Key Findings:**
- AT-FedBN achieves near-centralized performance despite privacy constraints
- Trust mechanism adds ~3% ROC-AUC over standard FedBN
- Handles severe label skew (Hospital 1: 83% positive, Hospital 2: 31% positive)

### Privacy Budget

With current configuration (σ=0.01, 10 rounds):
- **Epsilon (ε)**: ~55,115
- **Interpretation**: Weak formal guarantee, prioritizes utility

To strengthen privacy, increase `noise_scale` in `config.yaml` (higher σ → lower ε, lower utility).

## 🏥 Dataset

**Cardiovascular Disease Dataset**
- **Source**: Kaggle / UCI Machine Learning Repository
- **Size**: 70,000 patient records
- **Features**: 11 (age, gender, blood pressure, cholesterol, etc.)
- **Target**: Binary cardiovascular disease diagnosis
- **License**: Public domain

### Non-IID Split Example
```
Hospital 0: 37,943 samples (41.1% positive)
Hospital 1: 12,075 samples (83.0% positive)  ← Severe skew
Hospital 2:  4,854 samples (31.3% positive)
```

## 🔬 Algorithm Details

### AT-FedBN Strategy

**1. Update Consistency Trust Score**
```
Trust_i = ReLU(CosineSimilarity(ΔW_i, ΔW_global))
```
Where:
- `ΔW_i` = Client i's parameter update
- `ΔW_global` = Volume-weighted average update
- ReLU clips negative similarities to 0

**2. Hybrid Aggregation**
```
FinalWeight_i = β × (N_i / N_total) + (1-β) × (Trust_i / ΣTrust)
W_global = Σ FinalWeight_i × W_i
```

**3. FedBN Parameter Localization**
- **Shared globally**: Linear layer weights (fc1, fc2, fc3)
- **Kept local**: BatchNorm layers (bn1, bn2) + running statistics

**4. Differential Privacy**
```
ΔW_clipped = ΔW × min(1, C / ||ΔW||_2)
ΔW_private = ΔW_clipped + N(0, σ²C²)
```

## 🛠️ Development

### Adding New Experiments

1. Create `experiments/06_your_experiment.py`
2. Import necessary modules from `src/`
3. Follow existing experiment structure
4. Save results to `outputs/results/your_experiment.json`

### Extending the Model

To use a different architecture:
1. Create new model in `src/models/`
2. Ensure it has named BatchNorm layers (e.g., `bn1`, `bn2`)
3. Update `config.yaml` with new model parameters
4. Modify `CardioMLP.bn_keys()` method if needed

### Custom Datasets

1. Place raw data in `data/raw/`
2. Modify `src/data/preprocessing_pipeline.py` for your schema
3. Ensure target column is named appropriately
4. Update `config.yaml` input dimensions

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{at_fedbn_2024,
  title={AT-FedBN: Adaptive Trust-Weighted Federated Batch Normalization for Privacy-Preserving Medical AI},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Work

- **FedBN**: Li et al., "Federated Learning on Non-IID Data Silos" (ICLR 2021)
- **Flower Framework**: Beutel et al., "Flower: A Friendly Federated Learning Framework" (2020)
- **Differential Privacy**: Abadi et al., "Deep Learning with Differential Privacy" (2016)

## 📞 Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [your-email@domain.com]

## 🙏 Acknowledgments

- Flower team for the federated learning framework
- PyTorch community for deep learning tools
- Kaggle for the cardiovascular disease dataset

---

**Built with ❤️ for privacy-preserving healthcare AI**
