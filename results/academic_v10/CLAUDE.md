# Academic Evaluation V10 - Capacity-Aware Reject Head

## Overview

V10 modifies the PPO agent's architecture to give the reject head visibility into hardware capacity. In V9, the reject head only received the 64-dimensional task embedding, lacking any information about whether hardware types could actually accommodate the task. V10 expands the reject head input from 64 to 195 dimensions by concatenating mean and max hardware embeddings plus a 3-scalar capacity summary.

**Result: NEUTRAL.** V10 matches V9's performance within noise (39.74% vs 39.94%, -0.2pp).

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Training timesteps | 1,660,000 (41.5% of planned 4M) |
| Evaluation tasks per preset | 5,000 |
| Domain preset | full_spectrum (stress_test, high_load, small, medium, large) |
| Curriculum | Yes |
| Scarcity-aware rewards | Disabled |
| State encoder | v3 (28 dimensions) |
| Reject head input | 195 dimensions (64 task_emb + 64 mean_hw_emb + 64 max_hw_emb + 3 capacity_summary) |
| Learning rate | 3e-4 with cosine schedule |
| Rejection penalty | 0.5 |
| Acceptance bonus | 0.35 |
| Backward compatibility | V9 checkpoints loaded with strict=False |

### Reject Head Architecture Change

**V9 (64-dim input)**:
```
task_embedding (64) -> reject_head -> reject_probability
```

**V10 (195-dim input)**:
```
[task_embedding (64) | mean_hw_emb (64) | max_hw_emb (64) | capacity_summary (3)] -> reject_head -> reject_probability
```

Capacity summary scalars:
1. Number of valid (allocatable) hardware types
2. Average available CPU ratio across HW types
3. Average available memory ratio across HW types

## Key Results

### Generalization Results

| Preset | V4 Accept% | V10 Accept% | vs V4 | V9 Accept% | vs V9 |
|--------|------------|-------------|-------|------------|-------|
| small | 30.96% | 31.00% | +0.04pp | 31.30% | -0.30pp |
| medium | 50.64% | 52.06% | +1.42pp | 51.84% | +0.22pp |
| large | 65.28% | 69.82% | +4.54pp | 70.18% | -0.36pp |
| high_load | 27.00% | 30.28% | +3.28pp | 30.42% | -0.14pp |
| stress_test | 13.20% | 15.58% | +2.38pp | 15.94% | -0.36pp |
| **Average** | **37.42%** | **39.75%** | **+2.33pp** | **39.94%** | **-0.19pp** |

V10 matches V9 within statistical noise across all presets. The differences are all < 0.4pp and within standard deviations.

### Reject Probability with Capacity (New V10 Metric)

This metric measures how often the reject head outputs a high rejection probability when resources ARE available (i.e., the valid_mask contains at least one allocatable HW type).

| Preset | Avg Reject Prob with Capacity |
|--------|-------------------------------|
| small | 1.71% |
| medium | 0.12% |
| large | 0.025% |
| high_load | 0.83% |
| stress_test | 3.42% |

**Interpretation**: The reject head almost never rejects when capacity exists (0.025-3.42%). This proves the reject head is NOT the bottleneck - the high policy rejection rates seen in V9 are caused by the valid_mask being all-False (no HW type can accommodate the task), which means the environment itself forces rejection due to insufficient infrastructure capacity.

### Detailed Per-Preset Metrics

| Preset | Accept% | Energy (kWh) | Energy/task | Cap Rej | Policy Rej | Policy Rej % |
|--------|---------|--------------|-------------|---------|------------|--------------|
| small | 31.00% | 0.1020 | 0.000669 | 1,302 | 2,148 | 62.3% |
| medium | 52.06% | 0.2543 | 0.000981 | 1,376 | 1,021 | 42.6% |
| large | 69.82% | 0.3709 | 0.001064 | 1,196 | 313 | 20.7% |
| high_load | 30.28% | 0.0885 | 0.000605 | 1,520 | 1,966 | 56.4% |
| stress_test | 15.58% | 0.0275 | 0.000357 | 1,313 | 2,908 | 68.9% |

### Utilization Analysis (5 runs of 500 tasks each)

| Preset | Avg Accept% | Std | Avg Cap Rej | Avg Policy Rej | Cap Rej Ratio |
|--------|-------------|-----|-------------|----------------|---------------|
| small | 29.04% | 1.53% | 131.4 | 223.4 | 37.0% |
| medium | 51.16% | 2.63% | 138.2 | 106.0 | 56.7% |
| large | 70.24% | 2.17% | 118.2 | 30.6 | 79.5% |
| high_load | 28.84% | 3.40% | 154.0 | 201.8 | 43.3% |
| stress_test | 15.80% | 1.32% | 137.0 | 284.0 | 32.6% |

### Training Status

V10 was evaluated at 1.66M of a planned 4M timesteps (41.5% complete). The training was cut short because initial results showed no improvement trajectory that would justify the additional compute cost.

## Key Insight

**The reject head was never the bottleneck.** V10's most important contribution is diagnostic: the near-zero reject probabilities when capacity exists (0.025-3.42%) prove that the agent already knows how to accept tasks when resources are available. The high policy rejection percentages (56-69% on constrained presets) are caused by infrastructure exhaustion - the valid_mask is all-False because no hardware type has enough resources for the task.

This redirects attention from the agent's decision-making (which is actually quite good) to the environment's task-infrastructure mismatch: the task distribution generates tasks that are physically impossible to serve on small infrastructure.

## Files

| File | Purpose |
|------|---------|
| `evaluation_report.json` | Full report with avg_reject_prob_with_capacity (new V10 metric) |
| `data/generalization_results.json` | 5-preset metrics with reject probabilities |
| `data/utilization_summary.json` | Per-preset utilization analysis |
| `data/raw/decision_log.csv` | Raw per-step decision data |
| `data/raw/episode_summary.csv` | Per-episode summary data |
| `data/raw/training_curves.csv` | Training loss/reward curves |
| `models/` | Model weights and checkpoints |
| `logs/train_logs.txt` | Training log output |
| `figures/` | Visualization plots |
| `latex/` | LaTeX-formatted tables |

## Recommendations (for V11)

1. Scale task generation to match infrastructure capacity (the core bottleneck)
2. Use medium preset as reference scale (1.0), scale num_vms proportionally
3. Cap memory-intensive tasks at 50% of system memory
4. Lower rejection penalty further (0.5 -> 0.3)

## Verdict

**NEUTRAL (-0.2pp vs V9).** The capacity-aware reject head adds no measurable benefit because the reject head was already performing optimally. V10's diagnostic value is its most important contribution: proving that the performance ceiling is caused by task-infrastructure mismatch, not agent decision quality. This directly motivates V11's capacity-scaled task generation.
