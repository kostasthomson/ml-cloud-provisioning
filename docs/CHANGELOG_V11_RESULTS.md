# V11 Evaluation Results

## Training Summary

| Parameter | Value |
|-----------|-------|
| Timesteps | 2,007,040 |
| Episodes | 240 |
| Avg Reward | 34.75 |
| Training Time | 25,226s (~7h) |
| Throughput | 80 steps/sec |
| Domain preset | full_spectrum (stress_test, high_load, small, medium, large) |
| Curriculum | enabled |
| LR schedule | 3e-4 -> 1e-5 (cosine) |
| Entropy schedule | 0.05 -> 0.001 (linear) |
| Rejection penalty | 0.3 |
| Acceptance bonus | 0.35 |
| Scarcity-aware | disabled |
| Architecture | Same as V10 (capacity-aware reject head, 195-dim input) |
| Environment change | Capacity-scaled task generation (V11) |

## V11 Change Summary

Task `num_vms` now scales proportionally to infrastructure capacity using `medium` (1024 CPUs, 7168 GB) as reference (scale=1.0). Memory-intensive task `memory_per_vm` choices are capped at 50% of total system memory.

| Preset | Total CPU | Scale | Large num_vms | Medium num_vms |
|--------|-----------|-------|---------------|----------------|
| stress_test | 96 | 0.25 | 2-4 (was 4-15) | 1-2 (was 2-7) |
| high_load | 256 | 0.25 | 2-4 | 1-2 |
| small | 384 | 0.38 | 2-6 | 1-3 |
| medium | 1024 | 1.00 | 4-15 (unchanged) | 2-7 (unchanged) |
| large | 2250 | 2.09 | 8-31 (scaled up) | 4-14 |

## Generalization Results (10 episodes/preset, 500 steps)

| Preset | Acceptance | Energy/Task (kWh) | Avg Energy (kWh) | Policy Rej% | Cap Rej Ratio | Reject Prob w/ Capacity |
|--------|-----------|-------------------|-------------------|-------------|---------------|------------------------|
| small | 33.6% | 0.001816 | 0.301 +/- 0.070 | 64.7% | 35.3% | 0.027% |
| medium | 52.6% | 0.000940 | 0.245 +/- 0.032 | 43.4% | 56.6% | 0.004% |
| large | 61.6% | 0.000563 | 0.173 +/- 0.018 | 29.7% | 70.3% | 0.001% |
| high_load | 30.4% | 0.002752 | 0.397 +/- 0.045 | 63.3% | 36.7% | 0.005% |
| stress_test | 19.2% | 0.001700 | 0.156 +/- 0.019 | 73.6% | 26.4% | 0.029% |
| **Average** | **39.5%** | **0.001554** | **0.254** | **54.9%** | -- | -- |

## Utilization Analysis (5 episodes/preset)

| Preset | Avg Accept | Std | Cap Rej/ep | Policy Rej/ep | Cap Ratio |
|--------|-----------|-----|-----------|---------------|-----------|
| small | 32.3% | 3.3% | 123.6 | 215.0 | 36.7% |
| medium | 51.8% | 3.1% | 132.8 | 108.2 | 55.0% |
| large | 61.2% | 1.7% | 135.4 | 58.4 | 70.0% |
| high_load | 31.1% | 5.4% | 137.2 | 207.4 | 40.1% |
| stress_test | 18.4% | 1.7% | 103.2 | 304.6 | 25.4% |

## Version Comparison: Acceptance Rate

| Preset | V4 | V7 | V9 | V10 | V11 | V11 vs V10 |
|--------|------|------|------|------|------|------------|
| small | 34.4% | 31.5% | 31.3% | 31.0% | 33.6% | **+2.6pp** |
| medium | 57.1% | 49.3% | 51.8% | 52.1% | 52.6% | +0.5pp |
| large | 70.9% | 67.0% | 70.2% | 69.8% | 61.6% | **-8.2pp** |
| high_load | 35.4% | 28.0% | 30.4% | 30.3% | 30.4% | +0.1pp |
| stress_test | 18.0% | 13.4% | 15.9% | 15.6% | 19.2% | **+3.6pp** |
| **Average** | **43.1%** | **37.8%** | **39.9%** | **39.7%** | **39.5%** | **-0.2pp** |

Note: V4 used single-preset training. V7-V11 use domain randomization.

## Version Comparison: Policy Rejection %

| Preset | V7 | V9 | V10 | V11 | V11 vs V10 |
|--------|------|------|------|------|------------|
| small | 62.2% | 61.7% | 62.3% | 64.7% | +2.4pp |
| medium | 46.9% | 43.6% | 42.6% | 43.4% | +0.8pp |
| large | 26.8% | 23.9% | 20.7% | 29.7% | +9.0pp |
| high_load | 58.6% | 56.1% | 56.4% | 63.3% | +6.9pp |
| stress_test | 71.1% | 68.0% | 68.9% | 73.6% | +4.7pp |

## Version Comparison: Reject Probability With Capacity

| Preset | V10 | V11 | Improvement |
|--------|-----|-----|-------------|
| stress_test | 3.42% | 0.029% | 118x lower |
| high_load | 0.83% | 0.005% | 166x lower |
| small | 1.71% | 0.027% | 63x lower |
| medium | 0.12% | 0.004% | 30x lower |
| large | 0.03% | 0.001% | 30x lower |

## Version Comparison: Energy Per Task

| Preset | V10 | V11 | Delta |
|--------|-----|-----|-------|
| stress_test | 0.000357 | 0.001700 | +376% |
| high_load | 0.000605 | 0.002752 | +355% |
| small | 0.000669 | 0.001816 | +171% |
| medium | 0.000981 | 0.000940 | -4.2% |
| large | 0.001064 | 0.000563 | -47% |

Energy per task is not directly comparable between V10 and V11 because the task distributions changed. V11 constrained-preset tasks are physically larger (more VMs), so accepted tasks consume more energy. V11 large-preset tasks are also larger but fewer are accepted.

## Training Dynamics

- **Episodes 1-80** (curriculum: stress_test, high_load): Volatile rewards, avg ~0, many negative episodes as the agent learns on constrained presets
- **Episodes 80-120** (transition to easier presets): Rewards jump to 30-60 range
- **Episodes 120-240** (full mix): Stable at avg 35-55 with occasional dips to single digits
- Policy loss converged smoothly from -0.02 to -0.001
- Entropy settled at 0.15-0.17 (healthy exploration maintained)
- Value loss ranged 3.5-7.0 throughout training

## Interpretation

### Verdict: Mixed -- Constrained Presets Improved, Large Regressed

The capacity-scaled task generation successfully addressed the core hypothesis: physically impossible tasks on constrained presets were eliminated, leading to meaningful acceptance gains on stress_test (+3.6pp) and small (+2.6pp). However, the symmetric upward scaling on the `large` preset caused an unexpected regression (-8.2pp).

### What worked

1. **Downward scaling on constrained presets**: stress_test and small both showed the strongest single-version improvements since V4. Tasks now fit within physical capacity, allowing the agent to learn useful allocation policies instead of learning to reject everything.

2. **Reject head convergence**: The `avg_reject_prob_with_capacity` metric improved by 30-166x across all presets. The reject head now outputs near-zero rejection probability (0.001-0.029%) when capacity exists. This is functionally perfect reject-head behavior.

3. **Medium preset stability**: With scale=1.0, medium results were unchanged (+0.5pp), confirming the scaling formula correctly uses medium as the neutral reference point.

### What didn't work

1. **Upward scaling on `large` preset** (scale=2.09): Tasks scaled to num_vms 8-31 (was 4-15) consume resources too aggressively. Each larger task holds resources longer, creating compounding contention. The capacity rejection count rose from 1196 to 1349 despite the same infrastructure.

2. **Policy rejection % increased everywhere**: This is misleading. The raw reject probability with capacity is near-zero (proven by the metric). The "policy rejections" are actually cases where `valid_mask` is all-False for the task's compatible HW types -- effectively capacity exhaustion that the current classification counts as policy rejection. V11's larger tasks exhaust capacity faster, producing more of these edge cases.

3. **Higher variance on constrained presets**: stress_test std rose from 1.3% to 1.7%, high_load from 3.4% to 5.4%. The agent's behavior is less deterministic, suggesting the policy hasn't fully adapted to the new task distributions.

### Root cause of `large` regression

Resource consumption scales super-linearly with num_vms because each task blocks resources until completion. A 31-VM task blocks ~31x the resources of a 1-VM task but also runs proportionally longer, preventing other tasks from being accepted. The linear scaling formula `int(15 * 2.09) = 31` overshoots the sustainable task size for the large preset.

## Recommendations for V12

### Option A: Asymmetric scaling (recommended)

Only scale downward, never upward. Cap `capacity_scale` at 1.0:
```python
return max(0.25, min(scale, 1.0))
```
This preserves the V11 gains on constrained presets while reverting large/enterprise to original task distributions. Expected impact: recover the 8.2pp regression on large while keeping the +3.6pp stress_test gain.

### Option B: Sub-linear upward scaling

Use square root for upward scaling to dampen the effect:
```python
if scale > 1.0:
    scale = 1.0 + (scale - 1.0) ** 0.5
```
This would give large a scale of ~1.45 instead of 2.09.

### Option C: Per-task-type scaling caps

Different task types may need different scaling behavior. Large tasks are the most sensitive since they already have high per-VM resource requirements (8-32 vCPUs, 64-256 GB).

## Files

| File | Content |
|------|---------|
| `results/academic_v11/evaluation_report.json` | Full evaluation report |
| `results/academic_v11/data/generalization_results.json` | Per-preset generalization metrics |
| `results/academic_v11/data/utilization_summary.json` | Utilization analysis (5 episodes) |
| `results/academic_v11/data/training_results.json` | Training config and summary |
| `results/academic_v11/data/raw/decision_log.csv` | 25,000 individual decisions |
| `results/academic_v11/data/raw/episode_summary.csv` | 50 episode summaries |
| `results/academic_v11/data/raw/training_curves.csv` | 240 episodes of training metrics |
| `results/academic_v11/figures/comparative_utilization.png` | Utilization plots per preset |
| `results/academic_v11/figures/rejection_analysis.png` | Rejection breakdown bar charts |
| `results/academic_v11/figures/training_summary.png` | Training summary card |
| `results/academic_v11/figures/utilization_*_episode_0.png` | Per-preset utilization timeseries |
| `results/academic_v11/latex/generalization_table.tex` | LaTeX table for paper |
| `results/academic_v11/models/model_v11.pth` | Final model weights |
| `results/academic_v11/models/model_v11.metrics.json` | Training curves data |
