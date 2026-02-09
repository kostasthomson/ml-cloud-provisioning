# V9 Changelog and Recommendations

## Summary

V9 targets the structural bottlenecks identified in V7/V8 analysis: reward asymmetry driving conservative policy (45-71% policy rejection), lack of LR/entropy schedules, weak state encoding signals, and network architecture gaps.

**Baseline**: V7 (best, +1.4% over V4, avg acceptance 37.82%)
**V8 regression**: -1.8% due to lower LR (1e-4) confounded with GPU fix

## Changes

### 1. Reward Asymmetry (reward.py)

**Problem**: Rejection penalty (0.8) >> acceptance bonus (0.3) creates asymmetric risk landscape. Agent learns that rejecting is "safer" than accepting.

**Solution**: Reduce rejection penalty 0.8 -> 0.5, increase acceptance bonus 0.3 -> 0.35. Ratio changes from 2.67:1 to 1.43:1.

**Expected impact**: Less conservative policy, lower policy rejection rates.

### 2. Scarcity-Aware Default Off (reward.py)

**Problem**: Scarcity-aware rewards proved harmful in V5 (-5.8%) and showed no benefit in V6 recovery.

**Solution**: Default `scarcity_aware=False`.

**Expected impact**: Simpler, more stable reward signal.

### 3. Default Timesteps 2M (scripts)

**Problem**: 100K timesteps insufficient for policy convergence with domain randomization across multiple presets.

**Solution**: Default timesteps 100K -> 2M in both training scripts.

**Expected impact**: Better convergence, more stable final policy.

### 4. Entropy Coefficient Annealing (distributed_trainer.py)

**Problem**: Fixed entropy coefficient doesn't balance exploration vs exploitation across training phases.

**Solution**: Linear annealing from 0.05 (high exploration early) to 0.001 (exploitation late).

**Expected impact**: Better exploration early, more decisive policy late in training.

### 5. LayerNorm in Encoders (agent.py)

**Problem**: TaskEncoder and HWEncoder lack normalization, leading to internal covariate shift during training.

**Solution**: Add LayerNorm after each Linear layer (before ReLU) in both encoders.

**Expected impact**: More stable training gradients, faster convergence.

### 6. Cosine LR Schedule (distributed_trainer.py)

**Problem**: Fixed LR throughout training. V8 showed that simply lowering LR hurts; need adaptive schedule.

**Solution**: CosineAnnealingLR from 3e-4 to 1e-5 over total training updates.

**Expected impact**: High LR for fast early learning, low LR for fine-tuning late.

### 7. Continuous Scarcity Indicator (state_encoder.py)

**Problem**: Binary scarcity indicator (0 or 1 at 80% threshold) loses gradient information.

**Solution**: Continuous ramp: 0.0 at <=50% util, linearly increasing to 1.0 at >=90% util.
Formula: `max(0, min(1, (max_util - 0.5) / 0.4))`

**Expected impact**: Smoother gradient flow, better scarcity awareness when enabled.

### 8. Adaptive Energy Baseline (reward.py)

**Problem**: Fixed energy baseline (0.05 kWh) doesn't adapt to actual energy distribution across presets.

**Solution**: Exponential Moving Average (EMA) with alpha=0.01. After 20 observations, use EMA instead of fixed baseline for energy reward normalization.

**Expected impact**: Better calibrated energy rewards across different capacity environments.

### 9. Max+Mean HW Pooling for Value Head (agent.py)

**Problem**: Value head only sees mean HW embedding, missing information about best/worst HW types.

**Solution**: Concatenate task_emb + mean_hw_emb + max_hw_emb for value head input (embed_dim * 3).

**Expected impact**: Better value estimation, especially in heterogeneous environments.

### 10. Vectorized HW Encoder Forward Pass (agent.py)

**Problem**: Sequential loop over HW types in forward pass is inefficient.

**Solution**: Batch all HW vectors into a single tensor, process through encoder in one call. Compute scores via batched scorer network.

**Expected impact**: Faster forward pass, better GPU utilization during training.

### 11. Per-Preset Curriculum Thresholds (environment.py)

**Problem**: Fixed curriculum threshold (0.6) too high for constrained presets (stress_test can barely achieve 15% acceptance).

**Solution**: Per-preset thresholds: stress_test=0.15, high_load=0.30, small=0.35, medium=0.50, large=0.60, enterprise=0.65.

**Expected impact**: Curriculum actually advances through stages instead of getting stuck.

### 12. Reduced Execution Time Floor (environment.py)

**Problem**: Random floor `5.0 + uniform(0, 10)` adds noise that obscures HW type differences.

**Solution**: Fixed floor of 1.0 second.

**Expected impact**: Cleaner energy signal, HW type differences more visible to agent.

### 13. Continuous Scale Bucket (state_encoder.py)

**Problem**: Discrete scale bucket (5 values) loses information between thresholds.

**Solution**: Continuous: `min(total_cpus / 2000.0, 1.0)`.

**Expected impact**: Smoother gradient through scale feature, better generalization across capacity scales.

### 14. CLI Parameter Plumbing (scripts)

**Problem**: New reward and training params need CLI exposure.

**Solution**: Added `--acceptance-bonus`, `--entropy-start`, `--entropy-end`, `--lr-min` to both training scripts. All defaults baked in.

### 15. Documentation (this file)

All changes documented with problem, solution, and expected impact.

## V9 Run Command

```bash
torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py \
    --timesteps 2000000 \
    --output-dir results/academic_v9 \
    --use-capacity-features \
    --domain-preset mixed_capacity \
    --curriculum \
    --lr 3e-4
```

All V9 defaults are baked in -- no extra flags needed for:
- Reward asymmetry (0.5/0.35)
- Entropy annealing (0.05 -> 0.001)
- Cosine LR (3e-4 -> 1e-5)
- Continuous scarcity/scale
- Per-preset curriculum thresholds
- Reduced exec time floor

## Verification Checklist

1. `python -c "from rl.reward import RewardCalculator; r = RewardCalculator(); print(r.get_config())"` -- verify defaults
2. `python -c "from rl.agent import PolicyNetwork; p = PolicyNetwork(task_dim=28); print(p)"` -- verify LayerNorm + value_head
3. `python scripts/test_rl_module.py` -- existing tests pass
4. `python scripts/train_rl_distributed.py --timesteps 5000` -- verify entropy/LR schedule logging
5. Full evaluation with run command above

## Version History

| Version | Key Change | Avg Acceptance | vs V4 | Verdict |
|---------|------------|----------------|-------|---------|
| V4 | Domain randomization, GPU fix | 37.42% | Baseline | Baseline |
| V5 | Scarcity-aware rewards (1.5/2.0) | 35.25% | -5.8% | Regression |
| V6 | Gentler scaling (1.2/1.5), curriculum | 37.44% | +0.0% | Recovery |
| V7 | Capacity features (v3 encoder) | 37.82% | +1.4% | Best |
| V8 | Lower LR (1e-4) + GPU fix | 36.76% | -1.8% | Regression |
| **V9** | **Reward rebalance, schedules, architecture** | **TBD** | **TBD** | **TBD** |
