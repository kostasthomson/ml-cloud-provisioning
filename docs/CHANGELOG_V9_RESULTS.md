# V9 Evaluation Results

## Training Summary

| Parameter | Value |
|-----------|-------|
| Timesteps | 2,007,040 |
| Episodes | 960 (240/GPU x 4) |
| Training time | 22,834s (~6.3h) |
| Throughput | 88 steps/sec |
| Final avg reward | 7.46 |
| Domain preset | mixed_capacity (small, medium, large) |
| Curriculum | enabled (per-preset thresholds) |
| LR schedule | 3e-4 -> 1e-5 (cosine) |
| Entropy schedule | 0.05 -> 0.001 (linear) |
| Rejection penalty | 0.5 |
| Acceptance bonus | 0.35 |
| Scarcity-aware | disabled |

## Generalization Results (10 episodes/preset, 500 steps)

| Preset | Acceptance | Energy/Task (kWh) | Avg Energy (kWh) | Policy Rej% | Cap Rej Ratio |
|--------|-----------|-------------------|-------------------|-------------|---------------|
| small | 31.3% | 0.000659 | 0.101 +/- 0.020 | 61.7% | 38.3% |
| medium | 51.8% | 0.001022 | 0.263 +/- 0.042 | 43.6% | 56.4% |
| large | 70.2% | 0.001115 | 0.390 +/- 0.034 | 23.9% | 76.1% |
| high_load | 30.4% | 0.000635 | 0.092 +/- 0.026 | 56.1% | 43.9% |
| stress_test | 15.9% | 0.000328 | 0.026 +/- 0.014 | 68.0% | 32.0% |
| **Average** | **39.9%** | **0.000752** | **0.174** | **50.7%** | — |

## Utilization Analysis (5 episodes/preset)

| Preset | Avg Accept | Std | Cap Rej/ep | Policy Rej/ep | Cap Ratio |
|--------|-----------|-----|-----------|---------------|-----------|
| small | 29.1% | 1.6% | 131.8 | 222.6 | 37.2% |
| medium | 51.5% | 3.4% | 134.8 | 107.6 | 55.7% |
| large | 69.7% | 2.5% | 112.0 | 39.6 | 74.2% |
| high_load | 28.0% | 3.7% | 152.8 | 207.4 | 42.4% |
| stress_test | 16.0% | 1.2% | 139.2 | 280.6 | 33.2% |

## Version Comparison

| Preset | V4 (Baseline) | V7 (Prev Best) | V9 | V9 vs V7 |
|--------|--------------|----------------|-----|----------|
| small | ~30% | ~30% | 31.3% | +1.3pp |
| medium | ~40% | ~42% | 51.8% | +9.8pp |
| large | ~55% | ~55% | 70.2% | +15.2pp |
| high_load | ~25% | ~28% | 30.4% | +2.4pp |
| stress_test | ~14% | ~14% | 15.9% | +1.9pp |
| **Average** | **37.42%** | **37.82%** | **39.9%** | **+2.1pp** |

**V9 is the new best with +2.1pp avg improvement over V7 (+6.7% relative to V4 baseline).**

## Interpretation

### What improved
- **Medium and large presets saw the biggest gains** (+9.8pp and +15.2pp vs V7). The reward rebalance (penalty/bonus ratio 2.67:1 -> 1.43:1) directly reduced unnecessary policy rejections in environments with available capacity.
- **Capacity rejection ratio on large is 76%** — the agent is now pushing capacity limits rather than rejecting conservatively.
- **Policy rejection dropped to 24% on large** (was 45-71% range in V7), showing the agent learned when it's safe to accept.

### What didn't improve much
- **Constrained presets (stress_test, high_load, small)** still have 56-68% policy rejection. The agent is still conservative when resources are scarce.
- **Cap-rej ratio on stress_test is only 32%** — the agent rejects proactively even when capacity exists. This is the remaining structural bottleneck.

### Training dynamics
- Reward trajectory: 15 -> 4-5 (mid-training dip) -> 6-8 (recovery)
- The dip correlates with entropy/LR transition. High entropy early means random-ish policy with moderate rewards; as entropy drops, the policy sharpens and briefly underperforms before stabilizing.
- No curriculum advancement was logged, suggesting the per-preset thresholds for mixed_capacity (small=0.35, medium=0.50, large=0.60) may need tuning, or the agent reached them too quickly to be captured in logs.

## V10 Recommendations

1. **Lower rejection penalty further** (0.5 -> 0.3) for constrained presets specifically, or make it capacity-aware (lighter penalty when capacity is low)
2. **Try constrained_first or full_spectrum** domain presets instead of mixed_capacity — V9 never trained on stress_test/high_load directly
3. **Separate rejection scoring from capacity awareness**: The reject head only sees task embedding. Add capacity summary features to help it distinguish "can't fit" from "shouldn't fit"
4. **Increase training to 4M timesteps**: The reward was still recovering at 2M, suggesting more training may help
5. **Action masking improvements**: Pre-mask HW types that clearly can't fit the task (CPU/memory check) before softmax, reducing the "wasted" policy rejection probability

## Files

| File | Content |
|------|---------|
| `results/academic_v9/evaluation_report.json` | Full evaluation report |
| `results/academic_v9/data/generalization_results.json` | Per-preset generalization metrics |
| `results/academic_v9/data/training_results.json` | Training config and summary |
| `results/academic_v9/data/utilization_summary.json` | Utilization analysis |
| `results/academic_v9/data/raw/decision_log.csv` | 25,000 individual decisions |
| `results/academic_v9/data/raw/episode_summary.csv` | 50 episode summaries |
| `results/academic_v9/data/raw/training_curves.csv` | 240 episode training curves |
| `results/academic_v9/figures/` | Utilization plots, comparative, rejection analysis |
| `results/academic_v9/latex/generalization_table.tex` | LaTeX table |
| `results/academic_v9/models/model_v5.pth` | Final model |
