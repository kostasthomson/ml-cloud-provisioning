# V6 Evaluation Results - Comprehensive Analysis

## Version Summary

| Attribute | V4 | V5 | V6 |
|-----------|----|----|-----|
| Date | 2026-01-31 | 2026-02-02 | 2026-02-02 |
| Training Timesteps | ~100k | ~106k | ~508k |
| Training Time | N/A | 1490s | 6188s (~1.7h) |
| Avg Training Reward | N/A | -85.60 | -57.95 |
| Domain Preset | constrained_first | constrained_first | mixed_capacity |
| Curriculum Learning | No | No | Yes |
| Scarcity-Aware | No | Yes (1.5/2.0) | Yes (1.2/1.5) |
| Throughput | N/A | 71 fps | 82 fps |

---

## Acceptance Rate Comparison

### Raw Data

| Preset | V4 | V5 | V6 | V6 vs V4 | V6 vs V5 |
|--------|----|----|----|---------:|---------:|
| small | 30.96% | 29.98% | 32.42% | **+1.46%** | +2.44% |
| medium | 50.64% | 47.30% | 50.28% | -0.36% | +2.98% |
| large | 65.28% | 61.08% | 65.28% | **0.00%** | +4.20% |
| high_load | 27.00% | 25.30% | 26.16% | -0.84% | +0.86% |
| stress_test | 13.20% | 12.58% | 13.08% | -0.12% | +0.50% |
| **Average** | **37.42%** | **35.25%** | **37.44%** | **+0.02%** | **+2.19%** |

### Interpretation

**Positive Findings:**
- V6 fully recovered from V5 regression (+5.4% vs V5)
- `small` preset shows genuine improvement (+4.7% relative to V4)
- `large` preset perfectly matches V4 baseline (65.28%)
- Training reward improved significantly (-57.95 vs -85.60)

**Negative Findings:**
- Constrained presets (`high_load`, `stress_test`) still slightly below V4
- No overall improvement vs V4 baseline despite 5x more training
- Scarcity-aware rewards provide no measurable benefit

---

## Policy Rejection Analysis

### Raw Data (Policy Rejections / Total Rejections)

| Preset | V4 Policy Rej% | V5 Policy Rej% | V6 Policy Rej% | V6 vs V4 |
|--------|----------------|----------------|----------------|----------|
| small | 62.9% | 64.0% | 61.1% | **-1.8%** ✓ |
| medium | 41.0% | 38.7% | 45.7% | +4.7% ✗ |
| large | 25.5% | 21.8% | 26.0% | +0.5% ~ |
| high_load | 59.4% | 55.9% | 60.6% | +1.2% ~ |
| stress_test | 69.6% | 69.9% | 70.3% | +0.7% ~ |

### Interpretation

**Positive Findings:**
- `small` preset: Policy rejections decreased (model more willing to attempt)
- Overall policy rejection patterns are consistent across versions

**Negative Findings:**
- `medium` preset: Policy rejections increased significantly (+4.7%)
- `stress_test` remains at ~70% policy rejections (model overly conservative)
- The primary goal of reducing policy rejections on constrained environments was NOT achieved

---

## Capacity Rejection Ratio Analysis

The capacity rejection ratio indicates how often the model pushes to physical limits before giving up.
Higher = better (model attempting more before running out of resources).

| Preset | V4 Cap Ratio | V5 Cap Ratio | V6 Cap Ratio | V6 vs V4 |
|--------|--------------|--------------|--------------|----------|
| small | 37.2% | 37.5% | 38.9% | **+1.7%** ✓ |
| medium | 59.1% | 61.5% | 54.3% | -4.8% ✗ |
| large | 74.8% | 78.2% | 74.0% | -0.8% ~ |
| high_load | 40.6% | 43.1% | 39.4% | -1.2% ~ |
| stress_test | 30.4% | 30.8% | 29.7% | -0.7% ~ |

### Interpretation

**Positive Findings:**
- `small` preset shows improvement in capacity utilization

**Negative Findings:**
- `medium` preset shows significant regression in capacity pushing (-4.8%)
- Model is NOT learning to push limits more aggressively
- `stress_test` remains at ~30% capacity ratio (model gives up early)

---

## Energy Efficiency Analysis

| Preset | V4 Energy/Task | V5 Energy/Task | V6 Energy/Task | V6 vs V4 |
|--------|----------------|----------------|----------------|----------|
| small | 0.000872 | 0.000961 | 0.000815 | **-6.5%** ✓ |
| medium | 0.001421 | 0.001433 | 0.001388 | **-2.3%** ✓ |
| large | 0.001546 | 0.001671 | 0.001639 | +6.0% ✗ |
| high_load | 0.001018 | 0.001098 | 0.001096 | +7.7% ✗ |
| stress_test | 0.000845 | 0.001053 | 0.000960 | +13.6% ✗ |

### Interpretation

**Positive Findings:**
- `small` and `medium` presets show improved energy efficiency
- V6 is more energy-efficient than V5 across all presets

**Negative Findings:**
- Constrained presets (`high_load`, `stress_test`) use MORE energy per task than V4
- This suggests the model makes suboptimal HW choices under pressure

---

## Training Dynamics Comparison

| Metric | V5 | V6 | Interpretation |
|--------|----|----|----------------|
| Timesteps | 106k | 508k | 5x more training |
| Episodes | 32 | 56 | More complete episodes |
| Avg Reward | -85.60 | -57.95 | **+32% improvement** |
| Throughput | 71 fps | 82 fps | Better parallelization |
| Domain Preset | constrained_first | mixed_capacity | Easier starting point |
| Curriculum | No | Yes | Gradual difficulty |

### Interpretation

**Positive Findings:**
- Reward improvement suggests better convergence
- Curriculum learning helped stabilize training
- Higher throughput despite longer training

**Negative Findings:**
- Despite 5x more training, no improvement over V4
- Suggests fundamental limitation, not training duration
- Reward of -57.95 still very negative (not converged)

---

## Root Cause Analysis

### Why V6 Matches V4 But Doesn't Exceed It

1. **Feature Gap (Scale Blindness)**
   - State encoder uses RATIOS which hide absolute scale
   - `capacity_ratio = 0.5` means 256 free CPUs on medium, 32 on stress_test
   - Agent cannot distinguish environments by capacity

2. **Task Distribution Mismatch**
   - Same task generator across ALL presets
   - Large tasks (4-15 VMs × 8-32 vCPUs) are common
   - These fit on medium but NOT on stress_test
   - Agent learns one policy that must work everywhere

3. **Scarcity-Aware Rewards Not Effective**
   - Hypothesis: Higher rejection penalty → more acceptance attempts
   - Reality: Model learns to be more selective to avoid penalties
   - Net effect: Neutral to negative

4. **Policy Rejection Dominance**
   - On stress_test: 70% of rejections are policy choices
   - Only 30% are capacity-forced
   - Model "chooses" to reject even when resources exist
   - This is the core problem to solve

---

## Key Metrics Summary

### Acceptance Rate Trend
```
V4: ████████████████████████████████████████ 37.42%
V5: ████████████████████████████████████     35.25% (-5.8%)
V6: ████████████████████████████████████████ 37.44% (+0.0%)
```

### Policy Rejection % on Stress Test
```
V4: ██████████████████████████████████████████████████████████████████████ 69.6%
V5: ██████████████████████████████████████████████████████████████████████ 69.9%
V6: ██████████████████████████████████████████████████████████████████████ 70.3%
```
Target: Reduce to < 50%

### Training Reward Trend
```
V5: ████████████████████████████████████████████████████████████████████████████████████████ -85.60
V6: ██████████████████████████████████████████████████████████ -57.95 (+32%)
```
Target: Positive or near-zero

---

## Conclusions

### V6 Verdict: RECOVERY, NOT IMPROVEMENT

| Outcome | Status |
|---------|--------|
| Recover from V5 regression | ✅ Achieved |
| Beat V4 baseline | ❌ Not achieved |
| Reduce policy rejections | ❌ Not achieved |
| Improve energy efficiency | ⚠️ Mixed (small/medium ✓, constrained ✗) |

### What Worked
- Curriculum learning stabilized training
- Gentler scarcity scaling (1.2/1.5) prevented instability
- Mixed-capacity preset better than constrained-first for initial learning

### What Didn't Work
- Scarcity-aware rewards provide no benefit over V4's flat rewards
- 5x more training did not translate to better generalization
- Policy still overly conservative on constrained environments

### Recommended Next Steps

1. **Diagnostic**: Log raw state vectors to verify normalization
2. **Feature Engineering**: Add absolute capacity signals
3. **Task-Fit Ratio**: Tell agent how big task is relative to capacity
4. **Architecture**: Consider if deeper networks needed

---

## Files Reference

```
results/academic_v6/
├── data/
│   ├── generalization_results.json
│   ├── training_results.json
│   └── utilization_summary.json
├── figures/
│   ├── training_summary.png
│   ├── comparative_utilization.png
│   ├── rejection_analysis.png
│   └── utilization_*_episode_0.png
├── latex/
│   ├── generalization_table.tex
│   └── comparison_table.tex
├── logs/
│   └── train_logs.txt
├── models/
│   ├── model_v5.pth
│   └── model_v5.metrics.json
└── evaluation_report.json
```
