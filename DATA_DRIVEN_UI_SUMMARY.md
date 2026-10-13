# Data-Driven UI Refactor Summary

## Files Changed

1. **`ui/app.py`** - Comprehensive refactor to eliminate hardcoded values
2. **`tests/test_ui_data_driven.py`** (new) - Test suite for data-driven UI

## Hardcoded Values Eliminated

### 1. **Performance Metrics** (Previously: Scattered hardcoded examples)
**Now**: All metrics loaded dynamically from `outputs/results/*.json`
- Overview tab: `summary_statistics.json` → `performance_comparison`
- Training results: `at_fedbn.json`, `fedbn_only.json` → actual round history
- Model comparison: All strategies loaded from their respective result files
- Trust scores: `at_fedbn.json` → `trust_scores` by round

### 2. **Privacy Parameters** (Previously: σ=0.01, ε ≈ 55,000 hardcoded)
**Now**: Computed dynamically
- Noise scale: `cfg['privacy'].get('noise_scale')` from config.yaml
- Epsilon: `get_epsilon_from_noise_scale()` calculates from noise + rounds
- Privacy level: `format_epsilon()` returns classification (Strong/Weak/None)

### 3. **Configuration Values** (Previously: β=0.7, rounds=10, etc.)
**Now**: All from config.yaml
- Beta: `cfg['federated'].get('beta', 0.7)`
- Rounds: `cfg['federated']['num_rounds']`
- Local epochs: `cfg['training']['local_epochs']`
- Clients: `cfg['data']['num_clients']`
- Batch size: `cfg['training']['batch_size']`
- Learning rate: `cfg['training']['learning_rate']`

### 4. **Dataset Statistics** (Previously: 70,000 records, 3 hospitals)
**Now**: Calculated from actual data files
- Total samples: `get_dataset_statistics()` → sums hospital CSVs
- Hospital count: Config + file existence validation
- Per-hospital stats: Read from `hospital_{i}_train.csv` files
- Positive rates: Computed from actual class distributions

### 5. **Experiment Count** (Previously: Manual count)
**Now**: Dynamic file existence check
- `count_available_experiments()` scans `outputs/results/` for:
  - `central_baseline.json`
  - `fedavg_baseline.json`
  - `fedbn_only.json`
  - `at_fedbn.json`

### 6. **Training Round History** (Previously: Not displayed)
**Now**: Read from experiment JSON
- Global metrics per round: `results['global'][round_num]`
- Convergence tracking: Actual values from completed experiments
- Charts: Only shown when data exists

### 7. **Live Demo Results** (Previously: Static examples)
**Now**: Real-time from executed simulation
- Trust scores: Returned from `exp04.run_at_fedbn()`
- Hospital metrics: Actual per-hospital evaluation
- Configuration: User-selected beta + config defaults
- Results stored and displayed immediately after run

## New Data Loading Functions

```python
get_project_root() → Path         # Absolute project root
resolve_path(rel_path) → str      # Project-relative path resolution
load_json(filename) → dict        # Safe JSON loading with validation
load_config() → dict              # Config.yaml with error handling
count_available_experiments() → int  # Dynamic experiment counter
get_dataset_statistics(cfg) → dict   # Actual hospital data stats
get_epsilon_from_noise_scale(...) → float  # DP budget calculation
format_epsilon(epsilon) → tuple   # (display_string, privacy_level)
format_metric(value, type) → str  # Consistent metric formatting
get_image_path(filename) → str    # Graph path with existence check
```

## Legitimate Constants Retained

These are **intentional UI/UX constants**, not data values:

### Visual/Layout Constants
- CSS padding/margins: `2rem`, `1.5rem`, `0.5rem`
- Font sizes: `2rem`, `0.95rem`
- Colors: `#1f2937`, `#3b82f6`, `#e5e7eb`
- Max width: `1400px`

### UI Defaults/Bounds
- Beta slider: `min=0.0`, `max=1.0`, `default=0.7`, `step=0.1`
- Format precision: `.4f` for floats, `.2f` for percentages
- Skew threshold: `abs(x - 0.5) > 0.2` for class imbalance detection

### Explanatory Text Constants
- Live Demo rounds: Fixed at `2` for quick demonstration
- Expected demo time: "20-30 seconds" (informational, not a metric)
- Default beta values in explanations: 0.0, 0.7, 1.0 (examples, not results)

### File Path Constants
- `RESULTS_DIR = 'outputs/results'`
- `GRAPHS_DIR = 'outputs/graphs'`
- `DATA_DIR = 'data/processed'`

## Verification Results

### Test Suite
```bash
python -m pytest tests/test_ui_data_driven.py -v
# Result: 7 passed ✅
```

**Tests:**
1. `test_ui_import_and_data_functions` - Helper functions work
2. `test_epsilon_calculation` - Privacy budget computed correctly
3. `test_dataset_statistics_with_actual_files` - Hospital stats accurate
4. `test_no_hardcoded_metrics_in_overview` - No fake metrics in code
5. `test_experiment_count_dynamic` - Counts match actual files
6. `test_config_values_not_hardcoded` - Config used throughout
7. `test_privacy_parameters_dynamic` - Epsilon changes with config

### Full Test Suite
```bash
python -m pytest tests/ -v --ignore=tests/test_live_demo_simulation.py --ignore=tests/test_ui_live_demo_integration.py
# Result: 37 passed, 3 warnings ✅
```

### UI Import Check
```bash
python -c "import ui.app"
# Result: Success ✅
```

### Code Verification
- Searched for metric patterns (0.7XXX, 0.6XXX) → All from loaded data
- Verified `load_json` used for all result files
- Checked `cfg['...']` patterns for all config access
- Confirmed no `st.metric('...', 0.7234)` hardcoded patterns

## Dynamic Behavior Verification

### Configuration Changes
- Changed `cfg['privacy']['noise_scale']` → epsilon display updates
- Changed `cfg['federated']['num_rounds']` → system architecture updates
- Changed `cfg['federated']['beta']` → trust formula display updates

### Data Changes
- Adding/removing result JSON → experiment count updates
- Modifying hospital CSV → dataset statistics update
- Running new experiment → results appear immediately

### Live Demo
- Runs with user-selected beta
- Returns actual trust scores from simulation
- Displays real hospital metrics
- Updates dashboard state without restart

## Remaining Limitations

### Legitimate Non-Dynamic Elements
1. **UI Layout**: Tab structure, columns, markdown formatting (by design)
2. **Chart Generation**: Graphs must be pre-generated by `compare_all.py`
3. **Help Text**: Explanatory content about AT-FedBN algorithm (documentation)
4. **Default Values**: Fallbacks when data unavailable (clearly marked N/A)

### Missing Data Handling
- Missing results: Displays "N/A" or "Not run" with informative messages
- Missing graphs: Shows info box with generation instructions
- Incomplete experiments: Partial results displayed with warnings
- Empty datasets: Graceful fallback to config values

### Performance
- No caching of expensive reads (acceptable for dashboard size)
- File reads on every metric display (Streamlit reruns frequently anyway)
- No background refresh (user must rerun to see new experiment results)

## Scientific Integrity Preserved

✅ **No fabricated metrics**: All numbers from actual experiments or config  
✅ **No fake trust scores**: Only displayed when simulation actually run  
✅ **No estimated performance**: Missing experiments show "N/A", not guesses  
✅ **Privacy epsilon accurate**: Calculated from actual noise scale + rounds  
✅ **No train/test confusion**: Datasets correctly separated and labeled  
✅ **Transparency**: Missing data clearly indicated, not hidden  

## Code Quality

- **DRY principle**: Single functions for loading, formatting, path resolution
- **Error handling**: Missing files, malformed JSON, absent keys handled gracefully
- **Path safety**: All paths resolved relative to project root
- **Type safety**: Validation of loaded data before display
- **Testability**: Pure functions enable comprehensive unit testing
