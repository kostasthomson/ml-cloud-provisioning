# Academic Evaluation V4 - Domain Randomization Baseline

## Overview

V4 introduces domain randomization training, where the agent trains across multiple infrastructure presets (`small`, `medium`, `large`) simultaneously instead of a single preset. This version establishes the baseline for all subsequent experiments (V5-V11) by demonstrating that domain randomization improves generalization, particularly on constrained environments. V4 also includes a direct comparison between single-preset and domain-randomized training.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | 100,000 |
| Evaluation episodes per preset | 5 (utilization) |
| Training preset | mixed_capacity (small, medium, large) |
| Curriculum | No |
| State encoder | v1 (17 dimensions) |
| Total runtime | ~6,387 seconds (~1.8 hours) training |
| Learning rate | 3e-4 |
| Batch size | 64 |

## Key Results

### Training Metrics

| Metric | Value |
|--------|-------|
| Final reward | 563.77 |
| Avg reward (last 100) | 441.91 |
| Episodes completed | 48 |
| Training time | 6,387 seconds |

### Domain Randomization vs Single-Preset Comparison

This is the critical experiment that justifies domain randomization for all future versions.

| Preset | Single-Preset Accept% | Domain-Random Accept% | Improvement |
|--------|------------------------|-----------------------|-------------|
| small | 29.24% | 33.48% | +14.5% |
| medium | 53.12% | 51.04% | -3.9% |
| large | 69.44% | 68.40% | -1.5% |
| high_load | 27.24% | 28.28% | +3.8% |
| stress_test | 15.04% | 16.48% | +9.6% |
| **Average** | **38.82%** | **39.54%** | **+4.5%** |

**Training reward**: Single-preset 412.2 vs Domain-random 494.4 (+19.9%)

**Interpretation**: Domain randomization provides substantial gains on constrained presets (small +14.5%, stress_test +9.6%) at a small cost to the training distribution (medium -3.9%, large -1.5%). The net effect is +4.5% average improvement across all presets. This validates domain randomization as the preferred training strategy.

### Generalization Results (Domain-Randomized Model)

| Preset | Accept% | Energy/task (kWh) | Total Tasks | Accepted | Capacity Rej | Policy Rej | Cap Rej Ratio |
|--------|---------|-------------------|-------------|----------|--------------|------------|---------------|
| small | 33.48% | 0.000436 | 2,500 | 837 | 119.8 | 212.8 | - |
| medium | 51.04% | 0.001081 | 2,500 | 1,276 | 137.8 | 107.0 | - |
| large | 68.40% | 0.001249 | 2,500 | 1,710 | 117.2 | 40.8 | - |
| high_load | 28.28% | 0.000888 | 2,500 | 707 | 158.2 | 200.4 | - |
| stress_test | 16.48% | 0.000505 | 2,500 | 412 | 130.4 | 287.2 | - |

### V4 Reference Acceptance Rates (5,000 tasks per preset)

These are the canonical V4 baseline numbers used by all subsequent versions for comparison:

| Preset | Acceptance Rate |
|--------|-----------------|
| small | 30.96% |
| medium | 50.64% |
| large | 65.28% |
| high_load | 27.00% |
| stress_test | 13.20% |
| **Average** | **37.42%** |

Note: The comparison results (2,500 tasks) and the canonical baselines (5,000 tasks) differ slightly due to sample size. The 5,000-task numbers are used as the official V4 baseline.

## Interpretation

V4 marks a paradigm shift from single-preset to multi-preset training. The domain randomization approach forces the agent to learn a more general policy that transfers better across infrastructure scales. However, the agent still suffers from scale blindness - it cannot distinguish between environments of different sizes because the state encoder uses only relative features (utilization ratios). The constrained presets (stress_test 16.5%, high_load 28.3%) remain far below the medium/large performance, motivating the search for solutions in V5-V7.

The policy rejection analysis reveals a structural problem: on stress_test, 68.8% of rejections are policy decisions (not capacity-forced), meaning the agent is choosing to reject tasks it could theoretically accept. This suggests the policy learned on mixed presets is still biased toward medium-scale behavior.

## Files

| File | Purpose |
|------|---------|
| `academic_evaluation_report.json` | Full evaluation report (note: identical structure to V3) |
| `data/training_results.json` | Domain-randomized training metrics and loss curves |
| `data/comparison_results.json` | Single-preset vs domain-random comparison |
| `data/generalization_results.json` | Generalization test results |
| `data/ablation_results.json` | Ablation study |
| `utilization_analysis/` | Detailed per-preset utilization metrics |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Verdict

**BASELINE ESTABLISHED.** V4 becomes the reference point for all future versions. Domain randomization provides +4.5% average improvement over single-preset training. All subsequent versions measure themselves against V4's 37.42% average acceptance rate.
