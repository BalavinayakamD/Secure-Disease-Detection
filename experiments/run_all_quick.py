"""
Quick experiment runner - runs shorter versions of all experiments for testing.

This script runs:
1. FedBN Only (5 rounds)
2. AT-FedBN (5 rounds)
3. Comparison and visualization generation
4. Simplified sweeps

Usage:
    python experiments/run_all_quick.py
"""

import os
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 100)
print(" " * 35 + "RUNNING ALL EXPERIMENTS")
print("=" * 100)

# Experiment 03: FedBN Only (5 rounds for speed)
print("\n\n" + "=" * 100)
print("EXPERIMENT 03: FedBN ONLY (5 ROUNDS)")
print("=" * 100)

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp03", "experiments/03_fedbn_only.py")
    exp03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp03)
    history03, results03 = exp03.run_fedbn_experiment(num_rounds=5, num_clients=3)
    print("✓ Experiment 03 completed successfully")
except Exception as e:
    print(f"✗ Experiment 03 failed: {e}")
    import traceback
    traceback.print_exc()

# Experiment 04: AT-FedBN (5 rounds for speed)
print("\n\n" + "=" * 100)
print("EXPERIMENT 04: AT-FedBN (5 ROUNDS)")
print("=" * 100)

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp04", "experiments/04_full_at_fedbn.py")
    exp04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp04)
    strategy04, results04, payload04 = exp04.run_at_fedbn(num_rounds=5, num_clients=3, beta=0.7)
    print("✓ Experiment 04 completed successfully")
except Exception as e:
    print(f"✗ Experiment 04 failed: {e}")
    import traceback
    traceback.print_exc()

# Generate comparisons and visualizations
print("\n\n" + "=" * 100)
print("GENERATING COMPARISONS AND VISUALIZATIONS")
print("=" * 100)

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("compare", "experiments/compare_all.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    compare.main()
    print("✓ Comparison and visualization generation completed")
except Exception as e:
    print(f"✗ Comparison generation failed: {e}")
    import traceback
    traceback.print_exc()

# Simplified sweep (sigma only, as it doesn't require re-running FL)
print("\n\n" + "=" * 100)
print("RUNNING SIMPLIFIED SWEEPS")
print("=" * 100)

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("sweep", "experiments/05_sweep_experiments.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)
    sigma_results = sweep.sigma_sweep(num_rounds=10)
    print("✓ Sigma sweep completed")
except Exception as e:
    print(f"✗ Sigma sweep failed: {e}")
    import traceback
    traceback.print_exc()

print("\n\n" + "=" * 100)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 100)
print("\nGenerated outputs:")
print("  📊 outputs/results/central_baseline.json (already exists)")
print("  📊 outputs/results/fedbn_only.json")
print("  📊 outputs/results/at_fedbn.json")
print("  📊 outputs/results/master_table.md")
print("  📊 outputs/results/summary_statistics.json")
print("  📊 outputs/results/sigma_sweep.json")
print("  📈 outputs/graphs/*.png (all visualizations)")
print("\nTo view the dashboard:")
print("  streamlit run ui/app.py")
print("=" * 100)
