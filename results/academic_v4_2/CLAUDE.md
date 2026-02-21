# Academic Evaluation V4_2 - V4 Re-run / Validation

## Overview

V4_2 is a duplicate/re-run of the V4 evaluation, sharing the same structure and configuration as V3/V4. It was run to validate the V4 results and ensure reproducibility.

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

These results match V3/V4 closely, confirming reproducibility of the training pipeline.

### Evaluation Components

The report includes the same comprehensive evaluation suite as V3/V4:
- Multi-seed training (5 seeds)
- Pareto analysis (3 energy weights: 0.5, 0.7, 0.9)
- Ablation study (6 configurations)
- Generalization test (6 presets)
- Baseline comparison (PPO vs Scoring vs Random)

## Interpretation

V4_2 serves as a validation checkpoint confirming that the evaluation pipeline produces consistent results across runs. The metrics match V3/V4 within expected variance, establishing confidence in the experimental methodology.

## Files

| File | Purpose |
|------|---------|
| `academic_evaluation_report.json` | Full evaluation report |
| `data/` | All experimental data (ablation, generalization, multi-seed, pareto) |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Verdict

**VALIDATION RUN.** Confirms reproducibility of V3/V4 results. Not a distinct version - use V4 as the canonical baseline.
