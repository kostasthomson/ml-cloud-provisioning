# Academic Evaluation V3 - Medium-Scale Evaluation

## Overview

V3 is a medium-scale evaluation (50k timesteps, 5 seeds, 20 eval episodes) that provides a more statistically robust assessment than V2. It tests all 6 environment presets and 6 ablation configurations with increased evaluation episodes for tighter confidence intervals.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | 50,000 |
| Evaluation episodes | 20 |
| Seeds | 5 |
| Training preset | medium |
| Total runtime | ~12.5 hours |
| State encoder | v1 (17 dimensions) |

## Key Results

### Multi-Seed Training

| Metric | Mean |
|--------|------|
| Energy/task (kWh) | 0.001112 |
| Acceptance rate | 57.92% |
| SLA compliance | 92.97% |

Performance recovers compared to V2 (56.13%) with additional training steps, approaching V1 levels despite 4x fewer timesteps.

### Generalization Test (6 presets)

| Preset | Accept% | Energy/task | SLA% | Reward | Energy Gap | Accept Gap |
|--------|---------|-------------|------|--------|------------|------------|
| small | 34.4% | 0.000683 | 93.17% | -7.20 | -30.9% | -39.8% |
| medium | 57.1% | 0.000989 | 95.18% | 40.01 | baseline | baseline |
| large | 70.9% | 0.001236 | 94.28% | 68.35 | +24.9% | +24.1% |
| enterprise | 75.6% | 0.001358 | 95.11% | 78.35 | +37.2% | +32.4% |
| high_load | 35.4% | 0.000313 | 95.05% | -5.68 | -68.4% | -38.1% |
| stress_test | 18.0% | 0.000344 | 93.04% | -42.34 | -65.2% | -68.6% |

**Average generalization gaps**: Energy -20.5%, Acceptance -18.0%

**Interpretation**: The generalization gaps are worse than V1 (which only tested 4 presets). The constrained presets remain disastrous: stress_test at 18.0% acceptance and high_load at 35.4% confirm the scale blindness problem. The agent achieves negative rewards on 3 of 6 presets (small, high_load, stress_test), meaning it would be better to reject all tasks than use this policy on those environments.

### Ablation Study (6 configurations)

| Config | Energy/task | Accept% | SLA% | Reward | vs Full |
|--------|-------------|---------|------|--------|---------|
| full | 0.001049 | 59.25% | 95.19% | 44.45 | baseline |
| no_energy | 0.000976 | 57.05% | 95.35% | 39.89 | energy -7.0% |
| no_sla | 0.001234 | 53.90% | 93.32% | 33.24 | accept -9.0% |
| no_rejection_penalty | 0.001308 | 52.00% | 92.79% | 29.03 | accept -12.2% |
| no_acceptance_bonus | 0.001052 | 57.65% | 94.71% | 41.19 | accept -2.7% |
| aggressive_accept | 0.001126 | 57.65% | 93.76% | 40.94 | accept -2.7% |

**Key findings**:
- Removing rejection penalty now causes -12.2% acceptance drop (vs V2's -10.4%), with +24.7% energy increase - the most impactful ablation
- Removing SLA causes -9.0% acceptance and +17.6% energy, showing SLA signals help both acceptance and efficiency
- Removing energy weight actually reduces energy by 7.0%, counterintuitively - the no_energy model may avoid high-energy hardware types more consistently
- Aggressive accept still shows no meaningful improvement over baseline, confirming that parameter scaling alone cannot solve structural issues

## Interpretation

V3 confirms V1's findings with higher statistical confidence and expands the test matrix to include constrained environments. The average generalization gaps (-20.5% energy, -18.0% acceptance) across 5 out-of-distribution presets quantify the severity of the scale blindness problem. The ablation study shows that the rejection penalty is the single most important reward component for maintaining both acceptance rates and energy efficiency.

## Files

| File | Purpose |
|------|---------|
| `academic_evaluation_report.json` | Full evaluation report |
| `data/generalization_results.json` | 6-preset generalization test |
| `data/ablation_results.json` | 6-config ablation study |
| `data/multi_seed_results.json` | 5-seed training results |
| `data/pareto_results.json` | Pareto analysis |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Verdict

**CONFIRMS BASELINE.** Validates V1 findings with broader preset coverage. The constrained environment results motivate the domain randomization approach introduced in V4.
