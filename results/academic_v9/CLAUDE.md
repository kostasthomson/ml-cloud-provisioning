# Academic Evaluation V9 - Reward Rebalance + Architecture Improvements

## Overview

V9 implements 15 structural changes recommended after V7/V8 analysis, addressing reward asymmetry, training duration, learning rate scheduling, architectural improvements, and disabling scarcity-aware rewards. This is the most comprehensive single-version overhaul in the project history.

**Result: NEW BEST PERFORMANCE.** V9 achieves 39.92% average acceptance (+6.7% vs V4 baseline, +2.1pp vs V7).

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | 2,000,000 (4x increase from V6-V8's ~508k) |
| Evaluation tasks per preset | 5,000 |
| Domain preset | mixed_capacity (small, medium, large) |
| Curriculum | Yes (but no advancement logged) |
| Scarcity-aware rewards | **Disabled** (finally) |
| State encoder | v3 (28 dimensions) |
| Learning rate | 3e-4 with cosine schedule (-> 1e-5) |
| Entropy coefficient | Annealed 0.05 -> 0.001 |
| Rejection penalty | 0.5 (reduced from 0.8) |
| Acceptance bonus | 0.35 (increased from 0.3) |
| Architecture | LayerNorm in TaskEncoder/HWEncoder |

### Key Changes from V8

1. **Reward rebalancing**: Rejection penalty 0.8 -> 0.5, acceptance bonus 0.3 -> 0.35
2. **Scarcity-aware rewards disabled**: No more dynamic scaling
3. **Training duration 4x**: 508k -> 2M timesteps
4. **Entropy annealing**: 0.05 -> 0.001 (encourages exploration early, exploitation late)
5. **Cosine LR schedule**: 3e-4 -> 1e-5 (smooth decay)
6. **LayerNorm**: Added to encoder networks for gradient stability
7. **Continuous scarcity indicator**: 0-1 ramp at 50-90% util (replaces binary)
8. **Adaptive energy baseline**: EMA for reward normalization
9. **Max+mean HW pooling**: For value head
10. **Vectorized HW encoder**: Performance optimization
11. **Execution time floor**: 5-15s -> 1s fixed
12. **Continuous scale bucket**: Replaces discrete categories

## Key Results

### Generalization Results

| Preset | V4 Accept% | V9 Accept% | Absolute Change | Relative Change |
|--------|------------|------------|-----------------|-----------------|
| small | 30.96% | 31.30% | +0.34pp | +1.10% |
| medium | 50.64% | 51.84% | +1.20pp | +2.37% |
| large | 65.28% | 70.18% | +4.90pp | +7.51% |
| high_load | 27.00% | 30.42% | +3.42pp | +12.67% |
| stress_test | 13.20% | 15.94% | +2.74pp | +20.76% |
| **Average** | **37.42%** | **39.94%** | **+2.52pp** | **+6.7%** |

**All presets improved.** The largest gains are on stress_test (+20.8%) and high_load (+12.7%), demonstrating that the combined changes significantly improve constrained environment performance.

### Comparison with Previous Best (V7)

| Preset | V7 Accept% | V9 Accept% | Change |
|--------|------------|------------|--------|
| small | 31.52% | 31.30% | -0.22pp |
| medium | 49.26% | 51.84% | +2.58pp |
| large | 66.96% | 70.18% | +3.22pp |
| high_load | 27.96% | 30.42% | +2.46pp |
| stress_test | 13.42% | 15.94% | +2.52pp |
| **Average** | **37.82%** | **39.94%** | **+2.12pp** |

V9 improves on 4 of 5 presets vs V7. Only small shows a trivial regression (-0.22pp).

### Detailed Per-Preset Metrics

| Preset | Accept% | Energy (kWh) | Energy/task | Cap Rej | Policy Rej | Policy Rej % |
|--------|---------|--------------|-------------|---------|------------|--------------|
| small | 31.30% | 0.1015 | 0.000659 | 1,317 | 2,118 | 61.7% |
| medium | 51.84% | 0.2632 | 0.001022 | 1,358 | 1,050 | 43.6% |
| large | 70.18% | 0.3901 | 0.001115 | 1,134 | 357 | 23.9% |
| high_load | 30.42% | 0.0921 | 0.000635 | 1,526 | 1,953 | 56.1% |
| stress_test | 15.94% | 0.0258 | 0.000328 | 1,343 | 2,860 | 68.0% |

### Utilization Analysis (5 runs of 500 tasks each)

| Preset | Avg Accept% | Std | Avg Cap Rej | Avg Policy Rej | Cap Rej Ratio |
|--------|-------------|-----|-------------|----------------|---------------|
| small | 29.12% | 1.64% | 131.8 | 222.6 | 37.2% |
| medium | 51.52% | 3.44% | 134.8 | 107.6 | 55.7% |
| large | 69.68% | 2.53% | 112.0 | 39.6 | 74.2% |
| high_load | 27.96% | 3.67% | 152.8 | 207.4 | 42.4% |
| stress_test | 16.04% | 1.25% | 139.2 | 280.6 | 33.2% |

### Energy Efficiency Improvement

| Preset | V7 Energy/task | V9 Energy/task | Change |
|--------|----------------|----------------|--------|
| small | 0.000841 | 0.000659 | -21.6% (better) |
| medium | 0.001414 | 0.001022 | -27.7% (better) |
| large | 0.001590 | 0.001115 | -29.9% (better) |
| high_load | 0.000952 | 0.000635 | -33.3% (better) |
| stress_test | 0.000949 | 0.000328 | -65.4% (better) |

V9 achieves dramatically better energy efficiency across all presets, with improvements ranging from 21.6% to 65.4%. This is partly due to the reduced rejection penalty encouraging acceptance of lower-energy tasks that were previously rejected.

## Impact Attribution

The improvement is driven by multiple factors, but the most impactful appear to be:

1. **Reward rebalancing** (biggest impact on medium/large): Reducing rejection penalty from 0.8 to 0.5 makes the agent less conservative, accepting more tasks that were previously rejected despite being feasible
2. **Training duration** (4x increase): 2M timesteps allows full convergence vs the ~508k of V6-V8
3. **Disabling scarcity-aware rewards**: Removes the harmful dynamic that was confounding V7-V8 results
4. **Entropy annealing + cosine LR**: Better exploration/exploitation balance

## Remaining Structural Bottlenecks

Despite the improvement, constrained presets still show high policy rejection rates:
- stress_test: 68.0% policy rejection (vs 71.1% in V7)
- high_load: 56.1% policy rejection (vs 58.6% in V7)

The reduction is modest (~3pp), indicating that the reject head still lacks sufficient information to distinguish capacity-limited vs policy-limited scenarios.

## Files

| File | Purpose |
|------|---------|
| `evaluation_report.json` | Full evaluation report (note: no training section, training was separate) |
| `data/generalization_results.json` | 5-preset generalization metrics |
| `data/utilization_summary.json` | Per-preset utilization analysis |
| `models/` | Trained model weights and checkpoints |
| `logs/train_logs.txt` | Training log output |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Recommendations (for V10)

1. Add capacity information to the reject head (currently only sees 64-dim task embedding)
2. Lower rejection penalty further (0.5 -> 0.3)
3. Try constrained_first domain preset
4. Increase training to 4M timesteps
5. Implement action masking improvements

## Verdict

**NEW BEST (+6.7% vs V4).** V9 achieves the highest acceptance rate to date through comprehensive reward rebalancing, architecture improvements, and proper training duration. Energy efficiency improves dramatically (21-65% better). The remaining bottleneck is the reject head's inability to see infrastructure capacity, motivating V10's capacity-aware reject architecture.
