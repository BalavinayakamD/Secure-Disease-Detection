# AT-FedBN Launch Instructions

## Quick Start (Already Setup)

The project is fully configured and ready to run. All experiments have been executed and results are available.

### Launch the Dashboard

```bash
cd /home/<user>/Projects/Secure-Disease-Detection
source .venv/bin/activate
streamlit run ui/app.py
```

The dashboard will open at: **http://localhost:8501**

---

## Dashboard Overview

The web interface has **6 tabs**:

### 📋 Overview
- Project objectives and methodology
- System architecture
- Performance summary comparing all approaches
- Key findings and metrics

### 🏥 Hospital Data
- Non-IID data distribution visualization
- Hospital-level statistics
- Local model performance by hospital

### 🔄 Training Results
- Convergence plots (ROC-AUC, Balanced Accuracy)
- FedBN vs AT-FedBN comparison
- Training progression details

### 🔒 Trust Mechanism
- Trust-weighted aggregation explanation
- Trust heatmap across rounds
- Final aggregation weights table

### 📊 Model Comparison
- Centralized vs FedBN vs AT-FedBN
- Detailed metrics tables
- Privacy-utility tradeoff analysis

### ⚡ Live Demo
- Run interactive 2-round training
- Adjust trust parameter β
- See real-time results

---

## Re-Running Experiments (Optional)

If you want to regenerate results:

### Individual Experiments
```bash
# Centralized baseline (2 minutes)
python experiments/01_central_baseline.py

# FedBN without trust (1 minute)
python experiments/03_fedbn_only.py --rounds 10

# Full AT-FedBN system (1 minute)
python experiments/04_full_at_fedbn.py --rounds 10

# Generate all visualizations
python experiments/compare_all.py
```

### Parameter Sweeps
```bash
# Privacy-utility analysis (5 minutes)
python experiments/05_sweep_experiments.py --sweep sigma --rounds 10

# Trust parameter exploration (longer)
python experiments/05_sweep_experiments.py --sweep beta --rounds 5

# Non-IID severity analysis (longer)
python experiments/05_sweep_experiments.py --sweep alpha --rounds 5
```

---

## Testing

```bash
# Run complete test suite (23 seconds)
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_strategy.py -v
python -m pytest tests/test_fedavg_baseline.py -v
```

All 17 tests should pass.

---

## Current Results Summary

**Available Experiment Outputs:**
- ✅ `outputs/results/central_baseline.json` - Centralized upper bound
- ✅ `outputs/results/fedbn_only.json` - FedBN baseline
- ✅ `outputs/results/at_fedbn.json` - Full AT-FedBN with trust
- ✅ `outputs/results/sigma_sweep.json` - Privacy-utility analysis
- ✅ `outputs/results/master_table.md` - Formatted comparison table
- ✅ All visualization PNGs in `outputs/graphs/`

**Key Finding:**
- **AT-FedBN ROC-AUC: 0.8056** (outperforms centralized 0.7991)
- FedBN-only ROC-AUC: 0.6621 (shows trust mechanism is critical)
- Centralized Accuracy: 0.7290 (upper bound reference)

---

## Troubleshooting

### Dashboard won't start
```bash
# Check if Streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit
```

### Missing results
```bash
# Check what results exist
ls -lh outputs/results/

# Run missing experiments
python experiments/03_fedbn_only.py --rounds 10
python experiments/04_full_at_fedbn.py --rounds 10
python experiments/compare_all.py
```

### Test failures
```bash
# Run tests with more detail
python -m pytest tests/ -vv --tb=short

# If BCELoss error appears, the fix is already applied in src/fl/client.py
```

---

## Project Structure

```
Secure-Disease-Detection/
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
├── experiments/            # Experiment scripts
│   ├── 01_central_baseline.py
│   ├── 03_fedbn_only.py
│   ├── 04_full_at_fedbn.py
│   ├── 05_sweep_experiments.py
│   └── compare_all.py
├── src/                    # Core implementation
│   ├── data/              # Data preprocessing
│   ├── fl/                # Federated learning (client, server, strategy)
│   ├── models/            # CardioMLP architecture
│   ├── privacy/           # Differential privacy
│   └── utils/             # Metrics and plotting
├── tests/                 # Test suite (17 tests)
├── ui/                    # Web dashboard
│   └── app.py            # Streamlit application
├── data/                  # Datasets
│   ├── raw/              # Original data
│   └── processed/        # Hospital splits
└── outputs/               # Results and visualizations
    ├── results/          # JSON experiment outputs
    ├── graphs/           # PNG visualizations
    └── models/           # Trained model checkpoints
```

---

## Next Steps

### For Demonstration
1. Launch dashboard: `streamlit run ui/app.py`
2. Navigate through tabs to explore results
3. Use Live Demo tab for interactive training

### For Development
1. Modify `config.yaml` for different parameters
2. Run experiments: `python experiments/04_full_at_fedbn.py`
3. Regenerate visualizations: `python experiments/compare_all.py`
4. Run tests: `python -m pytest tests/ -v`

### For Deployment
1. Review `VERIFICATION_REPORT.md` for limitations
2. Increase σ in `config.yaml` for production privacy (σ ≥ 0.5)
3. Test with GPU if available
4. Consider horizontal scaling for more clients

---

## Documentation

- **VERIFICATION_REPORT.md**: Complete verification and testing documentation
- **README.md**: Project overview and setup
- **config.yaml**: All configurable parameters with comments
- **experiments/README.md**: Experiment descriptions

---

## Support

All components are verified and working. The dashboard displays real experimental results.

Current configuration:
- ✅ All tests passing (17/17)
- ✅ All core experiments complete
- ✅ Visualizations generated
- ✅ Dashboard functional
- ✅ Live demo operational

**The system is ready for demonstration and use.**
