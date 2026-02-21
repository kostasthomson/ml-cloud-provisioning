# Academic Evaluation V1 - Baseline PPO Agent

## Overview

V1 is the inaugural full-scale academic evaluation of the PPO-based RL agent for energy-aware cloud resource allocation. It establishes the foundational baseline using a single training preset (`medium`) and evaluates across generalization, ablation, multi-seed stability, and Pareto optimality dimensions.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | 200,000 |
| Evaluation episodes | 50 |
| Seeds | 10 (42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066) |
| Training preset | medium |
| Total runtime | ~79.6 hours |
| State encoder | v1 (17 dimensions: 12 task + 5 global) |

## Key Results

### Multi-Seed Training (10 seeds)

Demonstrates training stability and reproducibility.

| Metric | Mean | Std | 95% CI |
|--------|------|-----|--------|
| Energy/task (kWh) | 0.001075 | 4.20e-05 | [0.001043, 0.001107] |
| Acceptance rate | 57.04% | 0.97% | [56.31%, 57.77%] |
| SLA compliance | 93.78% | 0.54% | [93.38%, 94.19%] |
| Avg reward | 65.23 | 1.62 | [64.01, 66.46] |

All 10 seeds converged to similar final rewards (range: 1059-1083), confirming stable learning. Training completed 97 episodes per seed in ~12,800 seconds each.

### Per-Seed Evaluation Breakdown

| Seed | Energy/task | Accept% | SLA% | Reward |
|------|-------------|---------|------|--------|
| 42 | 0.001070 | 56.84% | 93.46% | 64.92 |
| 123 | 0.001060 | 59.02% | 94.34% | 68.47 |
| 456 | 0.001115 | 55.22% | 93.44% | 62.16 |
| 789 | 0.001064 | 57.58% | 93.85% | 66.37 |
| 1011 | 0.001123 | 56.96% | 94.00% | 65.10 |
| 2022 | 0.001011 | 57.20% | 94.37% | 65.41 |
| 3033 | 0.001077 | 55.82% | 94.37% | 63.17 |
| 4044 | 0.001034 | 57.52% | 93.50% | 66.09 |
| 5055 | 0.001155 | 57.24% | 92.56% | 65.29 |
| 6066 | 0.001041 | 56.98% | 93.93% | 65.38 |

### Generalization Test (trained on medium, tested on 4 presets)

| Preset | Accept% | Energy/task | SLA% | Reward | Energy Gap | Accept Gap |
|--------|---------|-------------|------|--------|------------|------------|
| small | 35.32% | 0.000677 | 92.53% | 29.43 | -36.1% | -40.7% |
| medium | 59.60% | 0.001059 | 94.03% | 69.44 | baseline | baseline |
| large | 71.42% | 0.001185 | 94.20% | 88.61 | +12.0% | +19.8% |
| enterprise | 78.06% | 0.001473 | 93.85% | 99.40 | +39.1% | +31.0% |

**Average generalization gaps**: Energy +5.0%, Acceptance +3.4%, SLA -0.5%

**Interpretation**: The agent generalizes reasonably well to larger infrastructures (large, enterprise) where more capacity is available, but struggles significantly on smaller environments. The 40.7% acceptance gap on `small` reveals scale blindness - the agent trained on medium-sized infrastructure cannot adapt its policy to constrained environments. This problem persists across all future versions until V7 introduces capacity features.

### Pareto Analysis (6 energy weights)

| Energy Weight | Energy/task | Accept% | SLA% | Reward | Pareto Optimal |
|---------------|-------------|---------|------|--------|----------------|
| 0.50 | 0.001053 | 59.28% | 94.57% | 68.89 | Yes |
| 0.60 | 0.001108 | 59.52% | 93.51% | 69.26 | Yes |
| 0.70 | 0.000937 | 57.78% | 94.32% | 66.53 | Yes (best energy) |
| 0.80 | 0.001149 | 58.40% | 92.95% | 67.29 | No |
| 0.90 | 0.001029 | 57.34% | 94.77% | 65.75 | No |
| 0.95 | 0.001049 | 58.38% | 93.83% | 67.35 | Yes |

4 of 6 configurations are Pareto-optimal. Energy weight 0.7 achieves the lowest energy per task (0.000937 kWh) while weight 0.6 achieves the highest acceptance rate (59.52%). The Pareto frontier shows a moderate trade-off between energy efficiency and task acceptance.

### Ablation Study (5 configurations)

| Config | Energy/task | Accept% | SLA% | Reward | vs Full |
|--------|-------------|---------|------|--------|---------|
| full | 0.000990 | 58.12% | 94.60% | 67.09 | baseline |
| no_energy | 0.001028 | 57.34% | 94.52% | 65.70 | energy +3.9% |
| no_sla | 0.001101 | 58.92% | 93.41% | 68.31 | SLA -1.3% |
| no_rejection_penalty | 0.001079 | 58.74% | 94.04% | 67.99 | accept +1.1% |
| energy_only | 0.001012 | 56.90% | 94.87% | 65.01 | accept -2.1% |

**Key findings**:
- Removing the energy component increases energy consumption by only 3.9%, suggesting the SLA and rejection signals partially encode energy-relevant behavior
- Removing SLA has the largest energy impact (+11.2%), indicating SLA compliance correlates with energy-efficient decisions
- Removing the rejection penalty increases acceptance by 1.1% but degrades energy efficiency by 9.0%
- The full configuration provides the best balance across all metrics

## Interpretation

V1 establishes that PPO can learn meaningful resource allocation policies for energy-aware cloud provisioning. The agent achieves 57% acceptance with 0.001075 kWh/task energy efficiency, with high stability across seeds (std < 1%). However, the single-preset training creates a policy that overfits to medium infrastructure scale. The generalization gaps (-40.7% acceptance on small, +31% on enterprise) reveal the fundamental scale blindness problem that becomes the focus of subsequent versions.

## Files

| File | Purpose |
|------|---------|
| `academic_evaluation_report.json` | Full evaluation report with all experiments |
| `data/multi_seed_results.json` | 10-seed training and evaluation details |
| `data/ablation_results.json` | 5-config ablation study |
| `data/pareto_results.json` | 6-weight Pareto analysis |
| `data/generalization_results.json` | Cross-preset generalization test |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Verdict

**BASELINE ESTABLISHED.** V1 provides a solid foundation demonstrating PPO viability. The main limitation is single-preset training causing poor generalization to different infrastructure scales.
