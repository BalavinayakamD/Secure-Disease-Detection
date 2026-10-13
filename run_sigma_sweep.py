#!/usr/bin/env python3
"""Quick sigma sweep runner."""
import sys
sys.path.insert(0, '.')
import experiments.sweep_experiments_05 as sweep

if __name__ == '__main__':
    sweep.sigma_sweep(sigma_values=[0.001, 0.01, 0.1, 1.0], num_rounds=10)
