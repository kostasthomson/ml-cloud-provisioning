# V8 Evaluation Results - Lower LR + GPU Compute Fix

## Version Information

| Attribute | V7 | V8 |
|-----------|----|----|
| Date | 2026-02-02 | 2026-02-03 |
| Training Timesteps | 508k | 508k |
| Avg Training Reward | -76.43 | **-74.49** |
| State Encoder | v3 (28-dim) | v3 (28-dim) |
| Capacity Features | Yes | Yes |
| Learning Rate | 3e-4 | **1e-4** |
| Scarcity-Aware | True (1.5/2.0) | True (1.5/2.0) ⚠️ |
| GPU Compute Fix | No | **Yes** |

> **Note**: Scarcity-aware rewards remained enabled despite V7 recommendation to disable.
> Both V7 and V8 configs show `scarcity_aware: true` with scales 1.5/2.0.

---

## Key Result: Regression vs V7 and V4 Baseline

**Average vs V4: -1.8% (regression)**
**Average vs V7: -1.1% (regression)**

---

## Acceptance Rate Comparison

| Preset | V4 (Baseline) | V7 | V8 | V8 vs V7 | V8 vs V4 |
|--------|---------------|----|----|----------|----------|
| small | 30.96% | **31.52%** | 29.38% | -2.14 ✗ | -5.1% ✗ |
| medium | 50.64% | 49.26% | **50.06%** | +0.80 ✓ | -1.1% ~ |
| large | 65.28% | **66.96%** | 64.64% | -2.32 ✗ | -1.0% ✗ |
| high_load | 27.00% | **27.96%** | 26.42% | -1.54 ✗ | -2.1% ✗ |
| stress_test | 13.20% | **13.42%** | 13.28% | -0.14 ~ | +0.6% ~ |
| **Average** | **37.42%** | **37.82%** | **36.76%** | **-1.07** ✗ | **-1.8%** ✗ |

V8 wins only on **medium**. V7 is better on all other presets.

---

## Policy Rejection Analysis

| Preset | V4 | V7 | V8 | V8 vs V7 |
|--------|----|----|-----|----------|
| small | 62.9% | 62.2% | 64.8% | +2.6% ✗ |
| medium | 41.0% | 46.9% | 45.3% | -1.7% ✓ |
| large | 25.5% | 26.8% | 26.4% | -0.5% ~ |
| high_load | 59.4% | 58.6% | 60.2% | +1.6% ✗ |
| stress_test | 69.6% | 71.1% | 70.4% | -0.7% ✓ |

Mixed. V8 more conservative on constrained presets (small, high_load), less conservative on medium/stress_test.

---

## Capacity Rejection Ratio

Higher = model pushing limits before giving up (good)

| Preset | V7 | V8 | Change |
|--------|----|-----|--------|
| small | 37.8% | 35.2% | -2.6% ✗ |
| medium | 53.1% | 54.7% | +1.7% ✓ |
| large | 73.2% | 73.6% | +0.5% ~ |
| high_load | 41.4% | 39.8% | -1.6% ✗ |
| stress_test | 28.9% | 29.6% | +0.7% ~ |

---

## Energy Efficiency

| Preset | V7 Energy/Task | V8 Energy/Task | Change |
|--------|----------------|----------------|--------|
| small | 0.000841 | 0.000985 | **+17.1%** ✗ |
| medium | 0.001414 | 0.001384 | -2.1% ✓ |
| large | 0.001590 | 0.001629 | +2.5% ~ |
| high_load | 0.000952 | 0.001086 | **+14.1%** ✗ |
| stress_test | 0.000949 | 0.000933 | -1.7% ✓ |

V8 is significantly less energy-efficient on constrained presets (small +17%, high_load +14%).

---

## Utilization Stability (std across runs)

| Preset | V7 std | V8 std | Interpretation |
|--------|--------|--------|----------------|
| small | 0.031 | 0.031 | Same |
| medium | 0.035 | **0.013** | V8 more stable |
| large | **0.010** | 0.033 | V7 more stable |
| high_load | 0.019 | 0.018 | Same |
| stress_test | 0.006 | 0.006 | Same |

---

## Analysis

### Why V8 Regressed

**1. Lower learning rate was counterproductive**
The lr reduction from 3e-4 to 1e-4 was motivated by the hypothesis that the larger 28-dim state space needed more stable learning. However:
- Training reward improved (-76.43 → -74.49), suggesting better fit to training distribution
- Generalization worsened across 4/5 presets
- This is a classic sign of **overfitting** — the lower LR caused the model to converge tighter to training environments at the expense of transfer

**2. Scarcity-aware rewards still active**
Despite V7 recommendations to disable scarcity-aware rewards, V8 still ran with `scarcity_aware=true` (scales 1.5/2.0). This confounds the LR change with continued scarcity penalty effects, already proven harmful in V5/V6.

**3. GPU compute fix impact unclear**
The GPU efficiency boost in `environment.py:_estimate_exec_time` was a V8 code change, but its effect cannot be isolated since LR also changed. The energy regression on constrained presets suggests the fix may have introduced noisier reward signals in capacity-limited scenarios where GPUs are less available.

### Why Medium Improved
Medium is the most balanced preset (moderate capacity, moderate load). The lower LR may have provided better fine-grained policy for this "average" scenario while losing adaptability to extremes.

---

## Verdict

**V8: FAIL — regression vs both V7 and V4 baseline.**

| Metric | V8 vs V7 | V8 vs V4 |
|--------|----------|----------|
| Avg acceptance | -1.1% | -1.8% |
| Constrained energy | +15.6% worse | — |
| Policy rejection | Mixed | Worse |

**V7 remains the best version.**

---

## V9 Recommendations

### Priority 1: Disable Scarcity-Aware Rewards
This has been recommended since V6 but never properly applied. Must be confirmed disabled before any other changes.

### Priority 2: Revert Learning Rate
Return to lr=3e-4 which worked well for V7. The 28-dim state space does not benefit from a lower LR.

### Priority 3: Isolate GPU Compute Fix
Run V7 config (lr=3e-4, no scarcity) WITH the GPU compute fix to evaluate its impact independently:
```bash
torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py \
    --timesteps 500000 \
    --output-dir results/academic_v9 \
    --use-capacity-features \
    --no-scarcity-aware \
    --domain-preset mixed_capacity \
    --curriculum \
    --lr 3e-4
```

### Priority 4: Consider Longer Training
V7/V8 both ran 508k timesteps. The 28-dim state space may benefit from 1M+ timesteps to properly learn capacity features without overfitting.

---

## Files Reference

```
results/academic_v8/
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
| **V7** | **Capacity features (v3)** | **37.82%** | **+1.4%** | **✓ Best** |
| V8 | Lower LR (1e-4) + GPU fix | 36.76% | -1.8% | ✗ Regression |
