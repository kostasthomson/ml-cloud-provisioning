# RL Training Improvements

This document describes the improvements made to the RL training process to achieve better energy efficiency in cloud resource allocation.

## Problem Analysis

Initial benchmark results showed that the PPO agent was not outperforming the Scoring Allocator baseline in terms of energy efficiency:

| Metric | PPO Agent | Scoring Allocator |
|--------|-----------|-------------------|
| Total Energy (kWh) | 1.2459 | 1.1421 |
| Efficiency (kWh/task) | 0.001137 | 0.001043 |

### Identified Issues

1. **Insufficient Energy Penalty**: Energy weight was only 0.5, insufficient to prioritize energy efficiency
2. **Incorrect Energy Baseline**: Baseline was 1.0 kWh while actual values are ~0.001-0.01 kWh
3. **Insufficient Training**: Default 50,000 timesteps was not enough for convergence
4. **No Training Visualization**: Unable to verify if reward curve had plateaued

## Implemented Changes

### 1. Aggressive Reward Shaping (`rl/reward.py`)

#### Previous Configuration
```python
energy_weight: float = 0.5
sla_weight: float = 0.3
rejection_penalty: float = 0.2
energy_baseline: float = 1.0  # Incorrect scale
```

#### New Configuration
```python
energy_weight: float = 0.8      # Increased from 0.5
sla_weight: float = 0.15        # Reduced to prioritize energy
rejection_penalty: float = 0.3  # Increased to discourage unnecessary rejections
energy_baseline: float = 0.005  # Calibrated to actual energy values
energy_excellent_threshold: float = 0.002  # Bonus for very efficient allocations
energy_poor_threshold: float = 0.01        # Penalty for inefficient allocations
```

#### Energy Reward Function Changes

The energy reward function now uses a piecewise linear approach:

| Normalized Energy (E/baseline) | Reward |
|-------------------------------|--------|
| ≤ 0.5 | 1.0 (excellent) |
| 0.5 - 1.0 | 1.0 → 0.5 (good) |
| 1.0 - 2.0 | 0.5 → -0.25 (poor) |
| > 2.0 | -0.25 → -1.5 (very poor) |

Additional bonuses/penalties:
- **+0.3** bonus for energy < 0.002 kWh (excellent efficiency)
- **-0.2** penalty for energy > 0.01 kWh (poor efficiency)

#### Running Statistics
The reward calculator now tracks running mean and standard deviation of energy consumption for future adaptive normalization:

```python
def _update_running_stats(self, energy_kwh: float):
    self._running_energy_count += 1
    self._running_energy_sum += energy_kwh
    self._running_energy_sq_sum += energy_kwh ** 2
```

### 2. Enhanced Training Script (`scripts/train_rl_enhanced.py`)

New training script with:

- **Extended Training Duration**: Default 1,000,000 timesteps (20x increase)
- **Comprehensive Logging**:
  - Episode rewards, lengths
  - Policy, value, and entropy losses
  - Acceptance rate
  - Energy consumption
  - Rolling averages
- **Training Visualization**: Automatic plot generation
- **Checkpointing**: Save model every 100,000 steps
- **Progress Reporting**: Real-time ETA and speed metrics

#### Usage
```bash
# Quick training (100K steps)
python scripts/train_rl_enhanced.py --timesteps 100000

# Full training (1M steps, recommended)
python scripts/train_rl_enhanced.py --timesteps 1000000

# Extended training (5M steps for best results)
python scripts/train_rl_enhanced.py --timesteps 5000000
```

#### Output Files
- `logs/rl_training/<timestamp>/training_metrics.json` - Raw metrics
- `logs/rl_training/<timestamp>/training_curves.png` - Visualization
- `logs/rl_training/<timestamp>/training_summary.json` - Summary
- `logs/rl_training/<timestamp>/checkpoint_*.pth` - Model checkpoints

### 3. Input Normalization Verification

The `StateEncoder` class (`rl/state_encoder.py`) already normalizes all inputs to [0, 1] range:

| Feature Category | Normalization |
|-----------------|---------------|
| Task VMs | /16.0 |
| vCPUs | /64.0 |
| Memory | /512.0 |
| Storage | /10.0 |
| Instructions | log10/15.0 |
| Power (idle/max) | /500.0 |
| Utilization | Already 0-1 |
| Compute capability | log10/10.0 |

This prevents gradient imbalance issues.

## Training Recommendations

### GPU Training Required
Training should be performed on a GPU machine for reasonable performance.

**On GPU Machine, run:**
```bash
cd ml-cloud-provisioning

# Activate virtual environment
source .venv/bin/activate  # Linux
# or
.venv\Scripts\activate     # Windows

# Production training (1M steps, ~2-3 hours on GPU)
python scripts/train_rl_enhanced.py --timesteps 1000000

# Extended training (5M steps, ~10-15 hours on GPU)
python scripts/train_rl_enhanced.py --timesteps 5000000

# Full options
python scripts/train_rl_enhanced.py \
    --timesteps 1000000 \
    --lr 3e-4 \
    --batch-size 64 \
    --epochs 10 \
    --gamma 0.99 \
    --checkpoint-interval 100000 \
    --env-preset medium
```

### Training Duration Guidelines
| Timesteps | Estimated Time (GPU) | Purpose |
|-----------|---------------------|---------|
| 100,000 | ~15 min | Quick testing |
| 500,000 | ~1 hour | Development |
| 1,000,000 | ~2-3 hours | Production |
| 5,000,000 | ~10-15 hours | Best results |

### Hyperparameter Tuning
If reward curve plateaus early:
- Decrease learning rate (try 1e-4)
- Increase batch size (try 128)
- Adjust PPO clip range

## Verification

After training, run the benchmark script:
```bash
python scripts/benchmark_performance.py
```

Expected improvements:
- Lower total energy consumption than baseline
- Better efficiency index (kWh/task)
- Maintained or improved SLA compliance

## Files Modified

| File | Changes |
|------|---------|
| `rl/reward.py` | Aggressive energy penalty, new thresholds, running stats |
| `scripts/train_rl_enhanced.py` | New enhanced training script |
| `scripts/benchmark_performance.py` | Benchmark evaluation script |

## Changelog

### 2026-01-16
- Increased `energy_weight` from 0.5 to 0.8
- Calibrated `energy_baseline` from 1.0 to 0.005 kWh
- Added excellent/poor energy thresholds with bonuses/penalties
- Created enhanced training script with logging and visualization
- Default training timesteps increased to 1,000,000
