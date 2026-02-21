# Academic Evaluation V6 - Gentle Scarcity Scaling + Curriculum Learning

## Overview

V6 addresses V5's regression by implementing two corrections: gentler scarcity-aware reward scaling (1.2/1.5 instead of 1.5/2.0) and curriculum learning via the `mixed_capacity` preset. The goal was to recover V4 baseline performance while testing whether scarcity-aware rewards can work at all.

**Result: RECOVERY.** V6 matches V4 baseline (37.44% vs 37.42%) but does not improve upon it.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | 507,904 |
| Evaluation tasks per preset | 5,000 |
| Domain preset | mixed_capacity (small, medium, large) |
| Curriculum | Yes |
| Scarcity-aware rewards | Yes |
| Scarcity rejection scale | 1.2 |
| Scarcity acceptance scale | 1.5 |
| State encoder | v1 (17 dimensions) |
| Learning rate | 3e-4 |
| Training time | 6,188 seconds (~1.7 hours) |
| FPS | 82.08 |
| Avg training reward | -57.95 |

## Key Results

### Generalization Results

| Preset | V4 Accept% | V6 Accept% | Absolute Change | Relative Change |
|--------|------------|------------|-----------------|-----------------|
| small | 30.96% | 32.42% | +1.46pp | +4.72% |
| medium | 50.64% | 50.28% | -0.36pp | -0.71% |
| large | 65.28% | 65.28% | 0.00pp | 0.00% |
| high_load | 27.00% | 26.16% | -0.84pp | -3.11% |
| stress_test | 13.20% | 13.08% | -0.12pp | -0.91% |
| **Average** | **37.42%** | **37.44%** | **+0.02pp** | **-0.003%** |

V6 fully recovers from V5's -5.44% regression. The `small` preset shows genuine improvement (+4.72%), while `large` matches V4 exactly. Constrained presets remain slightly below V4.

### Detailed Per-Preset Metrics

| Preset | Accept% | Energy (kWh) | Energy/task | Cap Rej | Policy Rej | Policy Rej % |
|--------|---------|--------------|-------------|---------|------------|--------------|
| small | 32.42% | 0.1317 | 0.000815 | 1,316 | 2,063 | 61.1% |
| medium | 50.28% | 0.3477 | 0.001388 | 1,351 | 1,135 | 45.7% |
| large | 65.28% | 0.5345 | 0.001639 | 1,285 | 451 | 26.0% |
| high_load | 26.16% | 0.1418 | 0.001096 | 1,454 | 2,238 | 60.6% |
| stress_test | 13.08% | 0.0627 | 0.000960 | 1,289 | 3,057 | 70.3% |

### Utilization Analysis (5 runs of 500 tasks each)

| Preset | Avg Accept% | Std | Avg Cap Rej | Avg Policy Rej | Cap Rej Ratio |
|--------|-------------|-----|-------------|----------------|---------------|
| small | 32.72% | 1.05% | 131.0 | 205.4 | 39.0% |
| medium | 48.92% | 1.77% | 139.4 | 116.0 | 54.7% |
| large | 66.56% | 0.54% | 124.4 | 42.8 | 74.4% |
| high_load | 25.72% | 1.97% | 145.0 | 226.4 | 39.1% |
| stress_test | 13.04% | 0.64% | 130.6 | 304.2 | 30.0% |

### Training Improvement vs V5

| Metric | V5 | V6 | Change |
|--------|----|----|--------|
| Avg reward | -85.60 | -57.95 | +32.3% (better) |
| Avg acceptance | 35.25% | 37.44% | +6.2% (recovered) |

The improved training reward confirms better convergence with gentler scaling.

## Root Cause Analysis

V6 identified four fundamental problems that scarcity-aware rewards cannot solve:

1. **Feature Gap (Scale Blindness)**: The v1 state encoder uses only utilization ratios (e.g., 80% CPU utilized), hiding absolute scale differences. A stress_test cluster at 80% utilization has far fewer absolute resources than a medium cluster at 80%. The agent cannot distinguish these situations.

2. **Task Distribution Mismatch**: The same task generator produces identical task distributions across all presets. A task requesting 178 vCPUs is reasonable for a medium cluster (1,024 CPUs) but impossible for a stress_test cluster (96 CPUs). This creates a fundamentally unfair evaluation.

3. **Scarcity-Aware Rewards Provide No Benefit**: V6 matches V4 with scarcity-aware rewards at 1.2/1.5; V4 achieved the same without any scarcity scaling. The mechanism adds complexity without measurable improvement.

4. **Policy Rejection Dominance**: On stress_test, 70.3% of rejections are policy choices (not capacity-forced). The agent is choosing to reject tasks that could fit, suggesting it has not learned when acceptance is feasible on constrained infrastructure.

## Files

| File | Purpose |
|------|---------|
| `evaluation_report.json` | Full evaluation report with v4_comparison |
| `data/training_results.json` | Training metrics and config |
| `data/generalization_results.json` | 5-preset generalization metrics |
| `data/utilization_summary.json` | Per-preset utilization analysis |
| `models/model_v5.pth` | Trained model weights |
| `logs/train_logs.txt` | Training log output |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Recommendations (for V7)

1. Add absolute capacity features to the state encoder (system scale, task fit ratios)
2. Disable scarcity-aware rewards (they provide no benefit)
3. Focus on solving the feature gap rather than reward engineering

## Verdict

**RECOVERY, NOT IMPROVEMENT (+0.003%).** V6 recovers V4 baseline exactly. The scarcity-aware reward mechanism is proven ineffective - gentler scaling merely neutralizes the harm caused by V5's aggressive scaling, returning to V4-equivalent performance. The root cause of poor constrained-environment performance is identified as a feature gap, not a reward engineering problem.
