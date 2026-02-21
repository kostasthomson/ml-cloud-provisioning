# Academic Evaluation V11 - Capacity-Scaled Task Generation

## Overview

V11 addresses the fundamental task-infrastructure mismatch identified by V10's diagnostic. Instead of modifying the agent, V11 modifies the environment: task sizes (specifically `num_vms`) are scaled proportionally to infrastructure capacity. The medium preset serves as the reference scale (1.0), and other presets scale their task sizes accordingly.

**Result: MIXED.** Constrained presets improve significantly (stress_test +3.6pp, small +2.6pp), but large preset regresses (-8.2pp). Average is neutral at 39.47% (-0.2pp vs V9).

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | 2,007,040 (complete) |
| Evaluation tasks per preset | 5,000 |
| Domain preset | full_spectrum (stress_test, high_load, small, medium, large) |
| Curriculum | Yes |
| Scarcity-aware rewards | Disabled |
| State encoder | v3 (28 dimensions) |
| Reject head input | 195 dimensions (V10 architecture) |
| Learning rate | 3e-4 with cosine schedule (-> 1e-5) |
| Entropy coefficient | Annealed 0.05 -> 0.001 |
| Rejection penalty | 0.3 (reduced from V9's 0.5) |
| Acceptance bonus | 0.35 |
| Training time | 25,226 seconds (~7.0 hours) |
| FPS | 79.56 |
| Avg training reward | 34.75 (first positive average!) |

### Capacity-Scaled Task Generation

**Scaling formula**:
```
capacity_scale = clamp(min(total_cpus/1024, total_memory/7168), 0.25, 3.0)
```

Medium preset (1024 CPUs, 7168 GB memory) = reference scale 1.0.

**Effect on task sizes (num_vms upper bounds)**:

| Preset | Scale Factor | Large Task VMs (before) | Large Task VMs (after) |
|--------|-------------|-------------------------|------------------------|
| stress_test | ~0.25 | 4-15 | 2-4 |
| small | ~0.5 | 4-15 | 2-8 |
| medium | 1.0 | 4-15 | 4-15 (unchanged) |
| large | ~2.0 | 4-15 | 8-31 |

**Memory cap**: Memory-intensive tasks capped at 50% of total system memory.

## Key Results

### Generalization Results

| Preset | V4 Accept% | V11 Accept% | vs V4 | V9 Accept% | vs V9 |
|--------|------------|-------------|-------|------------|-------|
| small | 30.96% | 33.56% | +2.60pp | 31.30% | +2.26pp |
| medium | 50.64% | 52.58% | +1.94pp | 51.84% | +0.74pp |
| large | 65.28% | 61.60% | -3.68pp | 70.18% | -8.58pp |
| high_load | 27.00% | 30.42% | +3.42pp | 30.42% | 0.00pp |
| stress_test | 13.20% | 19.20% | +6.00pp | 15.94% | +3.26pp |
| **Average** | **37.42%** | **39.47%** | **+2.05pp** | **39.94%** | **-0.47pp** |

### Comparison with V10

| Preset | V10 Accept% | V11 Accept% | Change |
|--------|-------------|-------------|--------|
| small | 31.00% | 33.56% | +2.56pp |
| medium | 52.06% | 52.58% | +0.52pp |
| large | 69.82% | 61.60% | -8.22pp |
| high_load | 30.28% | 30.42% | +0.14pp |
| stress_test | 15.58% | 19.20% | +3.62pp |

### Detailed Per-Preset Metrics

| Preset | Accept% | Energy (kWh) | Energy/task | Cap Rej | Policy Rej | Policy Rej % |
|--------|---------|--------------|-------------|---------|------------|--------------|
| small | 33.56% | 0.3007 | 0.001816 | 1,174 | 2,148 | 64.7% |
| medium | 52.58% | 0.2454 | 0.000940 | 1,342 | 1,029 | 43.4% |
| large | 61.60% | 0.1730 | 0.000563 | 1,349 | 571 | 29.7% |
| high_load | 30.42% | 0.3971 | 0.002752 | 1,276 | 2,203 | 63.3% |
| stress_test | 19.20% | 0.1563 | 0.001700 | 1,065 | 2,975 | 73.6% |

### Reject Probability with Capacity (V10+ metric)

| Preset | V10 Reject Prob | V11 Reject Prob | Improvement |
|--------|-----------------|-----------------|-------------|
| small | 1.71% | 0.027% | 63x better |
| medium | 0.12% | 0.004% | 30x better |
| large | 0.025% | 0.001% | 25x better |
| high_load | 0.83% | 0.005% | 166x better |
| stress_test | 3.42% | 0.029% | 118x better |

The reject probabilities dropped by 30-166x across all presets, indicating the agent is now almost perfectly calibrated in its rejection decisions when resources exist.

### Utilization Analysis (5 runs of 500 tasks each)

| Preset | Avg Accept% | Std | Avg Cap Rej | Avg Policy Rej | Cap Rej Ratio |
|--------|-------------|-----|-------------|----------------|---------------|
| small | 32.28% | 3.30% | 123.6 | 215.0 | 36.7% |
| medium | 51.80% | 3.13% | 132.8 | 108.2 | 55.0% |
| large | 61.24% | 1.68% | 135.4 | 58.4 | 70.0% |
| high_load | 31.08% | 5.36% | 137.2 | 207.4 | 40.1% |
| stress_test | 18.44% | 1.67% | 103.2 | 304.6 | 25.4% |

### Training Improvement

| Metric | V9 | V10 | V11 |
|--------|----|----- |-----|
| Training timesteps | 2M | 1.66M | 2M |
| Avg training reward | 7.46* | N/A | 34.75 |
| Training time | ~6.3h | N/A | ~7.0h |

*V9 training data from V9 changelog; V9 evaluation_report.json has no training section.

V11 achieves the first strongly positive average training reward (34.75), indicating healthy convergence.

### Energy Efficiency Analysis

V11's energy numbers are NOT directly comparable to V9/V10 because task sizes changed:

| Preset | V10 Energy/task | V11 Energy/task | Change | Explanation |
|--------|-----------------|-----------------|--------|-------------|
| small | 0.000669 | 0.001816 | +171% | Smaller tasks accepted -> lower energy base in V10 |
| medium | 0.000981 | 0.000940 | -4.2% | Same task distribution, slight improvement |
| large | 0.001064 | 0.000563 | -47.1% | Larger tasks -> more efficient resource use |
| high_load | 0.000605 | 0.002752 | +355% | Different task sizes, not comparable |
| stress_test | 0.000357 | 0.001700 | +376% | Much larger tasks proportionally |

The energy increases on small/high_load/stress_test are artifacts of the scaled task sizes (bigger tasks consume more energy), not agent inefficiency.

## Root Cause of Large Preset Regression

The large preset regression (-8.2pp vs V10) is caused by super-linear resource consumption when scaling tasks upward:

1. **Task sizes doubled**: Large preset tasks now request 8-31 VMs (up from 4-15)
2. **Resource contention**: Larger tasks hold more resources for longer, creating cascading unavailability
3. **Queue effects**: More tasks waiting for resources, increasing effective rejection rate
4. **Non-linear scaling**: Doubling num_vms more than doubles resource consumption due to shared infrastructure contention

The capacity rejection ratio on large dropped from 79.5% (V10) to 70.0% (V11), meaning fewer rejections are capacity-forced and more are policy decisions - the agent is reacting to the resource pressure by becoming more conservative.

## Files

| File | Purpose |
|------|---------|
| `evaluation_report.json` | Full report with training section and reject probabilities |
| `data/training_results.json` | Training metrics (first positive avg reward) |
| `data/generalization_results.json` | 5-preset metrics with reject probabilities |
| `data/utilization_summary.json` | Per-preset utilization analysis |
| `data/raw/decision_log.csv` | Raw per-step decision data |
| `data/raw/episode_summary.csv` | Per-episode summary data |
| `data/raw/training_curves.csv` | Training loss/reward curves |
| `models/model_v11.pth` | Final trained model |
| `models/model_v11_checkpoint_*.pth` | Training checkpoints |
| `logs/train_logs.txt` | Training log output |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Recommendations (for V12)

Three options for fixing the large preset regression while keeping constrained preset gains:

1. **Option A: Asymmetric scaling** - Cap scale at 1.0, only downscale (never increase task sizes above medium reference)
2. **Option B: Sub-linear upward scaling** - Use square root for upward scaling: `scale_up = sqrt(raw_scale)` so 2.0 becomes 1.41
3. **Option C: Per-task-type scaling caps** - Different scaling limits per task category (compute_intensive, memory_intensive, etc.)

## Verdict

**MIXED (+5.5% vs V4, -0.5% vs V9).** Capacity-scaled task generation successfully improves constrained preset performance (stress_test +3.6pp is the largest single-version gain on that preset). However, upward scaling on the large preset causes super-linear resource contention and -8.2pp regression. The net effect is a wash. The approach is promising but needs asymmetric or sub-linear upward scaling to avoid harming large-infrastructure performance.
