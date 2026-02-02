# V5 Evaluation Results and V6 Recommendations

## Version Information

| Attribute | V4 | V5 |
|-----------|----|----|
| Date | 2026-01-31 | 2026-02-02 |
| Training Preset | constrained_first | constrained_first |
| Reward System | Flat penalties/bonuses | Scarcity-aware dynamic scaling |
| GPU Tracking | Broken | Fixed |
| Training Timesteps | ~100k | ~106k |
| Avg Training Reward | N/A | -85.60 |

---

## V5 Changes Implemented

### 1. Scarcity-Aware Reward System
- Rejection penalty scales with available capacity (higher when resources free)
- Acceptance bonus scales with scarcity (higher when resources tight)
- Parameters: `scarcity_rejection_scale=1.5`, `scarcity_acceptance_scale=2.0`

### 2. GPU/Accelerator Tracking Fix
- Accelerators now properly consumed on allocation
- Accelerators released when tasks complete
- GPU utilization visible in figures

### 3. Minimum Execution Time
- Tasks now persist 5-15 seconds minimum
- Prevents instant resource release

### 4. Utilization Capture Timing
- Changed from before-step to after-step capture
- Charts now show current state, not stale data

---

## V5 Evaluation Results

### Acceptance Rate Comparison

| Preset | V4 Accept | V5 Accept | Change | Status |
|--------|-----------|-----------|--------|--------|
| small | 30.96% | 29.98% | -0.98% | ❌ Regression |
| medium | 50.64% | 47.30% | -3.34% | ❌ Regression |
| large | 65.28% | 61.08% | -4.20% | ❌ Regression |
| high_load | 27.00% | 25.30% | -1.70% | ❌ Regression |
| stress_test | 13.20% | 12.58% | -0.62% | ❌ Regression |
| **Average** | **37.42%** | **35.25%** | **-5.44%** | ❌ |

### Policy Rejection Comparison

| Preset | V4 Policy Rej% | V5 Policy Rej% | Change | Status |
|--------|----------------|----------------|--------|--------|
| small | 62.9% | 64.0% | +1.1% | ❌ Worse |
| medium | 41.0% | 38.7% | -2.3% | ✓ Better |
| large | 25.5% | 21.8% | -3.7% | ✓ Better |
| high_load | 59.4% | 55.9% | -3.5% | ✓ Better |
| stress_test | 69.6% | 69.9% | +0.3% | ❌ Worse |

### Capacity Rejection Ratio Comparison

| Preset | V4 Cap Ratio | V5 Cap Ratio | Change | Status |
|--------|--------------|--------------|--------|--------|
| small | 37.2% | 37.5% | +0.3% | ~ Neutral |
| medium | 59.1% | 61.5% | +2.4% | ✓ Better |
| large | 74.8% | 78.2% | +3.4% | ✓ Better |
| high_load | 40.6% | 43.1% | +2.5% | ✓ Better |
| stress_test | 30.4% | 30.8% | +0.4% | ~ Neutral |

---

## V5 Verdict: REGRESSION

### Summary
V5 underperformed V4 with an average **-5.44% decline in acceptance rate**. The scarcity-aware reward system did not achieve its intended goal of increasing acceptance on constrained environments.

### What Went Wrong

#### 1. Reward System Backfired
The hypothesis was:
- Higher rejection penalty when resources available → model tries harder to accept
- Higher acceptance bonus under scarcity → model values scarce acceptances more

Reality:
- Model learned to be MORE conservative to avoid the higher rejection penalties
- Result: Lower overall acceptance, not higher

#### 2. Insufficient Training
- Only 106k timesteps completed
- Average reward of -85.60 indicates model did not converge
- New reward dynamics require more exploration time

#### 3. No Curriculum Learning
- Model faced hardest environments (stress_test, high_load) immediately
- No gradual difficulty increase to build baseline competence

#### 4. Aggressive Scaling Parameters
- `scarcity_rejection_scale=1.5` may be too punitive
- `scarcity_acceptance_scale=2.0` creates reward instability

---

## V6 Recommendations

### Hypothesis Revision
Instead of scaling penalties higher, consider:
1. **Curriculum learning** to establish baseline behavior on easier presets
2. **Gentler scaling** to avoid reward instability
3. **Longer training** to allow proper convergence

### Recommended V6 Configuration

```bash
torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py \
    --timesteps 500000 \
    --domain-preset mixed_capacity \
    --curriculum \
    --scarcity-aware \
    --scarcity-rejection-scale 1.2 \
    --scarcity-acceptance-scale 1.5 \
    --output-dir results/academic_v6
```

### Parameter Changes from V5

| Parameter | V5 Value | V6 Recommended | Rationale |
|-----------|----------|----------------|-----------|
| timesteps | 100k | 500k | Allow proper convergence |
| domain-preset | constrained_first | mixed_capacity | Start with easier mix |
| curriculum | False | True | Gradual difficulty increase |
| scarcity-rejection-scale | 1.5 | 1.2 | Less aggressive penalty |
| scarcity-acceptance-scale | 2.0 | 1.5 | More stable rewards |

### Alternative V6 Approaches

#### Option A: Disable Scarcity-Aware (Baseline Reset)
```bash
torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py \
    --timesteps 500000 \
    --domain-preset constrained_first \
    --no-scarcity-aware \
    --output-dir results/academic_v6_baseline
```

#### Option B: Inverted Scaling Logic
Instead of penalizing rejection when resources available, BONUS acceptance when resources available:
- Current: Higher penalty for rejection when capacity exists
- Alternative: Higher bonus for acceptance when capacity exists (encourage greedy behavior)

---

## Files Generated (V5)

```
results/academic_v5/
├── data/
│   ├── generalization_results.json
│   └── utilization_summary.json
├── figures/
│   ├── comparative_utilization.png/pdf
│   ├── rejection_analysis.png/pdf
│   ├── utilization_small_episode_0.png/pdf
│   ├── utilization_medium_episode_0.png/pdf
│   ├── utilization_large_episode_0.png/pdf
│   ├── utilization_high_load_episode_0.png/pdf
│   └── utilization_stress_test_episode_0.png/pdf
├── latex/
│   ├── generalization_table.tex
│   └── comparison_table.tex
├── utilization_analysis/
│   └── analysis_summary.json
└── evaluation_report.json
```

---

## Metrics Definitions

| Metric | Description |
|--------|-------------|
| Acceptance Rate | Tasks accepted / Total tasks |
| Policy Rejection % | Rejections by model choice / Total rejections |
| Capacity Rejection % | Rejections due to resource limits / Total rejections |
| Capacity Rejection Ratio | Capacity rejections / Total rejections (higher = model pushing limits) |
| Energy per Task | Total energy / Accepted tasks |

---

## Historical Comparison

| Version | Key Change | Avg Acceptance | Outcome |
|---------|------------|----------------|---------|
| V4 | Domain randomization, GPU fix | 37.42% | Baseline |
| V5 | Scarcity-aware rewards | 35.25% | -5.44% regression |
| V6 | TBD | TBD | Pending |

---

## Next Steps

1. Run V6 training with recommended parameters
2. Compare V6 results against V4 baseline (not V5)
3. If V6 fails, consider Option B (inverted scaling) or disable scarcity-aware entirely
4. Document learnings for future reward shaping experiments
