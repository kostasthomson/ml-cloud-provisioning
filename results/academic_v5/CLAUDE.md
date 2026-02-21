# Academic Evaluation V5 - Scarcity-Aware Rewards (Aggressive)

## Overview

V5 introduces scarcity-aware reward scaling to address the high policy rejection rates on constrained environments identified in V4. The hypothesis was that dynamically adjusting rejection penalties and acceptance bonuses based on resource utilization would make the agent more context-sensitive. V5 uses aggressive scaling parameters (1.5x rejection, 2.0x acceptance).

**Result: REGRESSION.** V5 shows -5.44% average decline vs V4 baseline.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | ~508k (actual), target was higher |
| Evaluation tasks per preset | 5,000 |
| Domain preset | mixed_capacity (small, medium, large) |
| Curriculum | No |
| Scarcity-aware rewards | Yes |
| Scarcity rejection scale | 1.5 |
| Scarcity acceptance scale | 2.0 |
| State encoder | v1 (17 dimensions) |
| Learning rate | 3e-4 |

### Scarcity-Aware Reward Mechanism

The reward scaling modifies rejection penalties and acceptance bonuses based on current resource utilization:
- When resources are abundant (low utilization): rejection penalty multiplied by up to 1.5x (more punished for rejecting)
- When resources are scarce (high utilization): acceptance bonus multiplied by up to 2.0x (more rewarded for accepting under pressure)

## Key Results

### Generalization Results

| Preset | V4 Accept% | V5 Accept% | Absolute Change | Relative Change |
|--------|------------|------------|-----------------|-----------------|
| small | 30.96% | 29.98% | -0.98pp | -3.17% |
| medium | 50.64% | 47.30% | -3.34pp | -6.60% |
| large | 65.28% | 61.08% | -4.20pp | -6.43% |
| high_load | 27.00% | 25.30% | -1.70pp | -6.30% |
| stress_test | 13.20% | 12.58% | -0.62pp | -4.70% |
| **Average** | **37.42%** | **35.25%** | **-2.17pp** | **-5.44%** |

**All presets regressed.** The largest absolute drops are on medium (-3.34pp) and large (-4.20pp), the environments with the most capacity. This is counterintuitive - the scarcity mechanism was expected to help constrained environments but instead harmed all environments.

### Detailed Per-Preset Metrics

| Preset | Accept% | Energy (kWh) | Energy/task | Cap Rej | Policy Rej | Policy Rej % |
|--------|---------|--------------|-------------|---------|------------|--------------|
| small | 29.98% | 0.1366 | 0.000961 | 1,261 | 2,240 | 64.0% |
| medium | 47.30% | 0.3386 | 0.001433 | 1,616 | 1,019 | 38.7% |
| large | 61.08% | 0.5085 | 0.001671 | 1,522 | 424 | 21.8% |
| high_load | 25.30% | 0.1376 | 0.001098 | 1,646 | 2,089 | 55.9% |
| stress_test | 12.58% | 0.0660 | 0.001053 | 1,314 | 3,057 | 69.9% |

### Utilization Analysis (5 runs of 500 tasks each)

| Preset | Avg Accept% | Std | Avg Cap Rej | Avg Policy Rej | Cap Rej Ratio |
|--------|-------------|-----|-------------|----------------|---------------|
| small | 31.32% | 3.92% | 128.4 | 215.0 | 37.5% |
| medium | 47.00% | 2.75% | 163.2 | 101.8 | 61.5% |
| large | 58.24% | 3.19% | 161.8 | 47.0 | 77.7% |
| high_load | 25.08% | 1.57% | 161.2 | 213.4 | 43.1% |
| stress_test | 12.64% | 0.82% | 134.4 | 302.4 | 30.8% |

## Root Cause Analysis

1. **Reward system backfired**: The scarcity-aware penalties made the agent MORE conservative, not less. When resources are abundant, the 1.5x rejection penalty creates a strong incentive to accept, but on constrained presets where resources quickly become scarce, the reduced penalty allows more rejection. The net effect is the agent learns that rejection is less costly than expected.

2. **Insufficient training convergence**: The average training reward of -85.60 (from V6 report comparison) indicates the policy had not converged. With only ~508k timesteps and negative average reward, the complex scarcity signal may have destabilized learning.

3. **No curriculum learning**: The agent faced all difficulty levels simultaneously without progressive difficulty scheduling, making it harder to learn the relationship between scarcity and reward.

4. **Aggressive scaling parameters**: The 1.5/2.0 scales may have been too extreme, creating reward signal instability.

## Files

| File | Purpose |
|------|---------|
| `evaluation_report.json` | Full evaluation report with v4_comparison |
| `data/generalization_results.json` | 5-preset generalization metrics |
| `data/utilization_summary.json` | Per-preset utilization analysis |
| `figures/comparative_utilization.png` | Utilization comparison across presets |
| `figures/rejection_analysis.png` | Rejection type breakdown |
| `figures/utilization_*.png` | Per-preset utilization over time |
| `latex/generalization_table.tex` | LaTeX generalization table |
| `latex/comparison_table.tex` | LaTeX comparison table |

## Recommendations (for V6)

1. Reduce scaling parameters from 1.5/2.0 to 1.2/1.5
2. Add curriculum learning (start with easier presets, progress to harder ones)
3. Increase training timesteps to 500k+
4. Consider disabling scarcity-aware rewards entirely

## Verdict

**REGRESSION (-5.44%).** Scarcity-aware rewards with aggressive parameters cause universal regression across all presets. The mechanism that was designed to improve constrained environment performance instead degrades all environments.
