# Academic Evaluation V2 - Quick Iteration Test

## Overview

V2 is a rapid iteration run with reduced scale (30k timesteps, 3 seeds, 10 eval episodes) to test evaluation pipeline changes. It introduces two new test presets (`high_load`, `stress_test`) and adds a new ablation config (`aggressive_accept`). This is not a full academic evaluation but a validation checkpoint.

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | 30,000 |
| Evaluation episodes | 10 |
| Seeds | 3 (42, 123, 456) |
| Training preset | medium |
| Total runtime | ~3.7 hours |
| State encoder | v1 (17 dimensions) |

## Key Results

### Multi-Seed Training (3 seeds)

| Metric | Mean |
|--------|------|
| Energy/task (kWh) | 0.001147 |
| Acceptance rate | 56.13% |
| SLA compliance | 92.25% |

Performance is slightly below V1 due to 6.7x fewer training timesteps (30k vs 200k). The reduced training budget means the policy has not fully converged.

### Generalization Test (6 presets, including new high_load and stress_test)

| Preset | Accept% | Energy/task | SLA% | Reward | Energy Gap | Accept Gap |
|--------|---------|-------------|------|--------|------------|------------|
| small | 34.9% | 0.000431 | 91.98% | -6.35 | -64.8% | -35.3% |
| medium | 53.9% | 0.001226 | 91.84% | 33.21 | baseline | baseline |
| large | 67.7% | 0.001564 | 93.80% | 61.62 | +27.5% | +25.6% |
| enterprise | 79.8% | 0.001707 | 92.48% | 86.87 | +39.2% | +48.1% |
| high_load | 30.7% | 0.000778 | 92.18% | -15.24 | -36.6% | -43.0% |
| stress_test | 15.3% | 0.000600 | 91.50% | -47.94 | -51.0% | -71.6% |

**Average generalization gaps**: Energy -17.1%, Acceptance -15.3%

**Interpretation**: V2 reveals the severity of the generalization problem with the introduction of constrained presets. The stress_test preset (15.3% acceptance, -71.6% gap) and high_load preset (30.7% acceptance, -43.0% gap) demonstrate that the medium-trained agent fails catastrophically on resource-constrained infrastructures. Negative rewards on small, high_load, and stress_test indicate the agent is actively harmful (worse than random rejection) in these environments.

### Ablation Study (6 configurations, including new aggressive_accept)

| Config | Energy/task | Accept% | SLA% | Reward | vs Full |
|--------|-------------|---------|------|--------|---------|
| full | 0.001181 | 53.0% | 95.28% | 31.38 | baseline |
| no_energy | 0.001232 | 52.4% | 94.66% | 29.99 | energy +4.3% |
| no_sla | 0.001126 | 55.5% | 94.05% | 36.70 | accept +4.7% |
| no_rejection_penalty | 0.001518 | 47.5% | 92.84% | 19.61 | accept -10.4% |
| no_acceptance_bonus | 0.000983 | 56.2% | 93.42% | 38.22 | energy -16.7% |
| aggressive_accept | 0.001190 | 53.5% | 94.39% | 32.29 | accept +0.9% |

**Key findings**:
- Removing rejection penalty causes the largest acceptance drop (-10.4%) and energy increase (+28.6%), confirming its importance
- Removing acceptance bonus unexpectedly improves energy efficiency by 16.7% and increases acceptance by 6.0% - this suggests the bonus may cause suboptimal behavior at low training budgets
- The aggressive_accept config (rejection_penalty=1.2, acceptance_bonus=0.5) shows only marginal improvement (+0.9%), indicating aggressive parameters don't help with short training

## Interpretation

V2 serves primarily as a pipeline validation run and introduces the critical constrained environment presets that expose the generalization crisis. The stress_test and high_load results confirm that single-preset training produces agents that fail on different infrastructure scales. The reduced training budget (30k steps) means results should not be compared directly to V1, but the relative patterns are informative.

## Files

| File | Purpose |
|------|---------|
| `academic_evaluation_report.json` | Full evaluation report |
| `data/generalization_results.json` | 6-preset generalization test (includes high_load, stress_test) |
| `data/ablation_results.json` | 6-config ablation study (includes aggressive_accept) |
| `data/multi_seed_results.json` | 3-seed training results |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Verdict

**PIPELINE VALIDATION.** Not a production evaluation due to reduced scale, but successfully introduces constrained presets that reveal the generalization crisis driving future version development.
