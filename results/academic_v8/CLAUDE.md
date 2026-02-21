# Academic Evaluation V8 - Lower Learning Rate + GPU Fix

## Overview

V8 attempts to improve upon V7 by lowering the learning rate from 3e-4 to 1e-4 (hypothesized to help with the larger 28-dimensional state space) and adding a GPU compute efficiency fix. V7's recommendations also included disabling scarcity-aware rewards, but this was not done in V8.

**Result: REGRESSION.** V8 drops to -1.75% vs V4 baseline (36.76% vs 37.42%), losing V7's gains.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | 507,904 |
| Evaluation tasks per preset | 5,000 |
| Domain preset | mixed_capacity (small, medium, large) |
| Curriculum | Yes |
| Scarcity-aware rewards | Yes (still enabled at 1.5/2.0 - NOT fixed) |
| State encoder | v3 (28 dimensions) |
| Learning rate | 1e-4 (reduced from V7's 3e-4) |
| Training time | 6,190 seconds (~1.7 hours) |
| FPS | 82.05 |
| Avg training reward | -74.49 |

## Key Results

### Generalization Results

| Preset | V4 Accept% | V8 Accept% | Absolute Change | Relative Change |
|--------|------------|------------|-----------------|-----------------|
| small | 30.96% | 29.38% | -1.58pp | -5.10% |
| medium | 50.64% | 50.06% | -0.58pp | -1.15% |
| large | 65.28% | 64.64% | -0.64pp | -0.98% |
| high_load | 27.00% | 26.42% | -0.58pp | -2.15% |
| stress_test | 13.20% | 13.28% | +0.08pp | +0.61% |
| **Average** | **37.42%** | **36.76%** | **-0.66pp** | **-1.75%** |

V8 regresses on 4 of 5 presets. Only stress_test shows marginal improvement (+0.08pp). The small preset is hit hardest (-5.10%), consistent with the lower LR causing underfitting on the most challenging environments.

### Comparison with V7

| Preset | V7 Accept% | V8 Accept% | Change |
|--------|------------|------------|--------|
| small | 31.52% | 29.38% | -2.14pp |
| medium | 49.26% | 50.06% | +0.80pp |
| large | 66.96% | 64.64% | -2.32pp |
| high_load | 27.96% | 26.42% | -1.54pp |
| stress_test | 13.42% | 13.28% | -0.14pp |
| **Average** | **37.82%** | **36.76%** | **-1.06pp** |

V8 wins only on medium (+0.80pp) and loses on all other presets.

### Detailed Per-Preset Metrics

| Preset | Accept% | Energy (kWh) | Energy/task | Cap Rej | Policy Rej | Policy Rej % |
|--------|---------|--------------|-------------|---------|------------|--------------|
| small | 29.38% | 0.1390 | 0.000985 | 1,242 | 2,289 | 64.8% |
| medium | 50.06% | 0.3452 | 0.001384 | 1,367 | 1,130 | 45.3% |
| large | 64.64% | 0.5245 | 0.001629 | 1,302 | 466 | 26.4% |
| high_load | 26.42% | 0.1424 | 0.001086 | 1,465 | 2,214 | 60.2% |
| stress_test | 13.28% | 0.0619 | 0.000933 | 1,282 | 3,054 | 70.4% |

### Utilization Analysis (5 runs of 500 tasks each)

| Preset | Avg Accept% | Std | Avg Cap Rej | Avg Policy Rej | Cap Rej Ratio |
|--------|-------------|-----|-------------|----------------|---------------|
| small | 30.08% | 3.11% | 126.4 | 223.2 | 36.2% |
| medium | 51.48% | 1.34% | 137.4 | 105.2 | 56.7% |
| large | 62.60% | 3.35% | 136.8 | 50.2 | 73.2% |
| high_load | 26.24% | 1.79% | 148.8 | 220.0 | 40.4% |
| stress_test | 13.28% | 0.60% | 128.8 | 304.8 | 29.7% |

### Energy Efficiency Regression

| Preset | V7 Energy/task | V8 Energy/task | Change |
|--------|----------------|----------------|--------|
| small | 0.000841 | 0.000985 | +17.1% (worse) |
| medium | 0.001414 | 0.001384 | -2.1% (better) |
| large | 0.001590 | 0.001629 | +2.5% (worse) |
| high_load | 0.000952 | 0.001086 | +14.1% (worse) |
| stress_test | 0.000949 | 0.000933 | -1.7% (better) |

Energy efficiency worsened significantly on constrained presets (small +17.1%, high_load +14.1%).

## Root Cause Analysis

1. **Lower learning rate caused overfitting to training distribution**: The 3x reduction in LR (3e-4 -> 1e-4) reduced the agent's ability to explore and generalize. The training reward improved slightly (-74.49 vs -76.43), but this represents better fitting to the training environments at the cost of generalization.

2. **Scarcity-aware rewards still enabled**: V7 recommended disabling these (at 1.5/2.0 scales), but V8 kept them active. This confounds the analysis - the LR change cannot be isolated from the ongoing scarcity reward distortion.

3. **GPU compute fix impact unclear**: The GPU compute efficiency change was bundled with the LR reduction, making it impossible to attribute improvements or regressions to either change independently.

## Files

| File | Purpose |
|------|---------|
| `evaluation_report.json` | Full evaluation report with v4_comparison |
| `data/training_results.json` | Training metrics (confirms lr=0.0001) |
| `data/generalization_results.json` | 5-preset generalization metrics |
| `data/utilization_summary.json` | Per-preset utilization analysis |
| `models/model_v5.pth` | Trained model weights |
| `logs/train_logs.txt` | Training log output |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Recommendations (for V9)

1. Revert learning rate to 3e-4
2. Disable scarcity-aware rewards entirely
3. Isolate the GPU compute fix in a separate experiment
4. Address reward structure fundamentally (reduce rejection penalty, increase acceptance bonus)
5. Significantly increase training timesteps

## Verdict

**REGRESSION (-1.75% vs V4, -2.8% vs V7).** The lower learning rate causes overfitting and reduces generalization. Combined with still-enabled harmful scarcity rewards, V8 loses all of V7's gains. The experiment highlights the importance of isolating changes (one variable at a time) and following through on recommendations (disabling scarcity rewards).
