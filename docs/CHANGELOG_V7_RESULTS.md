# V7 Evaluation Results - First Improvement Over Baseline

## Version Information

| Attribute | V6 | V7 |
|-----------|----|----|
| Date | 2026-02-02 | 2026-02-02 |
| Training Timesteps | 508k | 508k |
| Avg Training Reward | -57.95 | -76.43 |
| State Encoder | v2 (22-dim) | **v3 (28-dim)** |
| Capacity Features | No | **Yes** |
| Scarcity-Aware | Yes (1.2/1.5) | Yes (1.5/2.0) ⚠️ |

---

## Key Result: First Model to Beat V4 Baseline

**Average improvement: +1.4%**

---

## Acceptance Rate Comparison

| Preset | V4 (Baseline) | V6 | V7 | V7 vs V4 |
|--------|---------------|----|----|----------|
| small | 30.96% | 32.42% | 31.52% | **+1.8%** ✓ |
| medium | 50.64% | 50.28% | 49.26% | -2.7% ✗ |
| large | 65.28% | 65.28% | **66.96%** | **+2.6%** ✓ |
| high_load | 27.00% | 26.16% | **27.96%** | **+3.6%** ✓ |
| stress_test | 13.20% | 13.08% | **13.42%** | **+1.7%** ✓ |
| **Average** | **37.42%** | **37.44%** | **37.82%** | **+1.4%** ✓ |

---

## Policy Rejection Analysis

| Preset | V4 | V7 | Change |
|--------|----|----|--------|
| small | 62.9% | 62.2% | -0.7% ✓ |
| medium | 41.0% | 46.9% | +5.9% ✗ |
| large | 25.5% | 26.8% | +1.3% ~ |
| high_load | 59.4% | 58.6% | -0.8% ✓ |
| stress_test | 69.6% | 71.1% | +1.5% ~ |

---

## Capacity Rejection Ratio

Higher = model pushing limits before giving up (good)

| Preset | V4 | V7 | Change |
|--------|----|----|--------|
| small | 37.2% | 37.8% | +0.6% ✓ |
| medium | 59.1% | 53.1% | -6.0% ✗ |
| large | 74.8% | 73.2% | -1.6% ~ |
| high_load | 40.6% | 41.4% | +0.8% ✓ |
| stress_test | 30.4% | 28.9% | -1.5% ~ |

---

## Energy Efficiency

| Preset | V4 Energy/Task | V7 Energy/Task | Change |
|--------|----------------|----------------|--------|
| small | 0.000872 | 0.000841 | **-3.6%** ✓ |
| medium | 0.001421 | 0.001414 | -0.5% ~ |
| large | 0.001546 | 0.001590 | +2.8% ~ |
| high_load | 0.001018 | 0.000952 | **-6.5%** ✓ |
| stress_test | 0.000845 | 0.000949 | +12.3% ✗ |

---

## What Worked (Capacity Features)

The v3 state encoder with 6 new capacity features showed improvement:

1. **Constrained environments improved**: high_load +3.6%, stress_test +1.7%
2. **Large environments improved**: large +2.6%
3. **Energy efficiency improved** on small (-3.6%) and high_load (-6.5%)

### V3 Capacity Features
```python
total_system_cpus_normalized    # System scale (0-1)
total_system_memory_normalized  # Memory scale (0-1)
cpu_fit_ratio                   # Task fits in available CPUs
mem_fit_ratio                   # Task fits in available memory
scale_bucket                    # Categorical: 0.1 (stress) → 1.0 (enterprise)
task_relative_size              # Task size vs total capacity
```

---

## What Didn't Work

### 1. Scarcity-Aware Rewards Still Enabled
V7 accidentally used scarcity_aware=True with aggressive scales (1.5/2.0):
```json
"scarcity_aware": true,
"scarcity_rejection_scale": 1.5,
"scarcity_acceptance_scale": 2.0
```
This was proven harmful in V5/V6 and likely caused the medium regression.

### 2. Medium Preset Regression (-2.7%)
- Policy rejection increased from 41.0% to 46.9%
- Capacity rejection ratio dropped from 59.1% to 53.1%
- Model became more conservative on balanced environments

### 3. Training Reward Degradation
- V6: -57.95 → V7: -76.43 (-32% worse)
- Larger state space (28-dim vs 22-dim) may need more training or lower LR

---

## Issues Identified

### Issue 1: GPU Underutilization
Observation: GPU utilization in figures is lower than expected.

**Problem**: GPUs are more energy-efficient per computation despite higher power draw:
- GPU: 300W × 10s = 0.83 Wh (fast completion)
- CPU: 200W × 30s = 1.67 Wh (slow completion)

**Root Cause**: Current reward function penalizes instantaneous power, not total energy.

**Recommendation for V8**: Add compute efficiency features or modify reward to favor throughput.

### Issue 2: Configuration Error
Scarcity-aware rewards should have been disabled for V7 to isolate capacity feature impact.

---

## V8 Recommendations

### Configuration
```bash
torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py \
    --timesteps 500000 \
    --output-dir results/academic_v8 \
    --use-capacity-features \
    --domain-preset mixed_capacity \
    --curriculum \
    --lr 1e-4
```

Note: `--scarcity-aware` now defaults to disabled (fixed in V8 prep).

### Changes from V7
| Parameter | V7 | V8 |
|-----------|----|----|
| scarcity_aware | True (accidental) | **False** (default) |
| scarcity_rejection_scale | 1.5 | N/A |
| scarcity_acceptance_scale | 2.0 | N/A |
| learning_rate | 3e-4 | **1e-4** |

### GPU Utilization Fix
Added hybrid GPU compute model in `environment.py:_estimate_exec_time`:
- For compute-intensive tasks (instructions > 1e11), GPUs provide 30% efficiency boost
- Addresses GPU underutilization observed in V4-V7 figures
- GPUs are faster despite higher power draw, resulting in better total energy

### Rationale
1. **Disable scarcity-aware**: Proven harmful in V5/V6, pollutes capacity feature evaluation
2. **Lower learning rate**: Larger state space (28-dim) needs more stable learning
3. **GPU compute fix**: Agent should now prefer GPUs for large compute tasks

---

## Files Reference

```
results/academic_v7/
├── data/
│   ├── generalization_results.json
│   ├── training_results.json
│   └── utilization_summary.json
├── figures/
│   ├── comparative_utilization.png
│   ├── rejection_analysis.png
│   └── utilization_*_episode_0.png
├── latex/
│   ├── generalization_table.tex
│   └── comparison_table.tex
├── logs/
│   └── train_logs.txt
├── models/
│   └── model_v5.pth
└── evaluation_report.json
```

---

## Version History

| Version | Key Change | Avg Accept | vs V4 | Verdict |
|---------|------------|------------|-------|---------|
| V4 | Domain randomization | 37.42% | Baseline | ✓ Baseline |
| V5 | Scarcity rewards (1.5/2.0) | 35.25% | -5.8% | ✗ Regression |
| V6 | Gentler scaling (1.2/1.5) | 37.44% | +0.0% | ~ Recovery |
| **V7** | **Capacity features (v3)** | **37.82%** | **+1.4%** | **✓ Improvement** |
| V8 | Capacity only (no scarcity) | TBD | TBD | Pending |
