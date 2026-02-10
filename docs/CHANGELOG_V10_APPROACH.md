# V10: Capacity-Aware Reject Head

## Problem Analysis (V9)

V9 achieves 39.9% avg acceptance (+6.7% over V4 baseline), but the reject head only sees `task_emb` (64-dim) -- zero visibility into HW capacity. This causes 56-68% policy rejection on constrained presets even when resources exist.

The value head already receives `[task_emb, mean_hw_emb, max_hw_emb]` (192-dim), proving aggregated HW info is available. The reject head was the only component making decisions without infrastructure context.

### Evidence

| Component | Input | Sees Capacity? |
|-----------|-------|----------------|
| Scorer | task_emb + hw_emb | Yes (per-HW) |
| Value head | task_emb + mean_hw + max_hw | Yes (aggregated) |
| **Reject head (V9)** | **task_emb only** | **No** |

## V10 Architecture Change

### Reject Head Input: 64 -> 195 dimensions

```
task_emb (64)           - Task context
mean_hw_emb (64)        - Average HW state across all types
max_hw_emb (64)         - Best HW state (max pooled)
capacity_summary (3)    - Scalar capacity indicators:
  - max_hw_score        - Best allocation score (from scorer)
  - mean_hw_score       - Average allocation score
  - num_valid_ratio     - Fraction of HW types with capacity
```

### Reject Head Architecture

```
V9:  Linear(64, 32) -> ReLU -> Linear(32, 1)
V10: Linear(195, 64) -> ReLU -> Linear(64, 1)
```

### Backward Compatibility

- V9 checkpoints detected by absence of `reject_head_version: v2`
- On load: strip old reject_head weights, load with `strict=False`
- Warning logged to user about reject head reinitialization

## Files Modified

| File | Change |
|------|--------|
| `rl/agent.py` | Reject head 64->195, forward() wiring, save/load compat |
| `rl/distributed_trainer.py` | Reject prob logging, save/load compat |
| `scripts/test_rl_module.py` | New assertions + edge case tests |
| `scripts/run_academic_evaluation_v5.py` | --version-tag, reject prob metric |
| `scripts/utilization_analysis.py` | Track reject probs with capacity |

## Training Command (Linux, 4x GPU)

```bash
torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py \
    --timesteps 4000000 --output-dir results/academic_v10 \
    --use-capacity-features --domain-preset full_spectrum --curriculum \
    --lr 3e-4 --rejection-penalty 0.3 --acceptance-bonus 0.35 \
    --version-tag v10
```

### Key V10 vs V9 training changes

- 4M timesteps (was 2M -- V9 reward still recovering)
- `full_spectrum` preset (was `mixed_capacity` -- includes stress_test/high_load)
- `rejection-penalty 0.3` (was 0.5 -- further reward rebalance)

## Expected Impact

- Reject head can now see that resources exist before deciding to reject
- Policy rejection % on constrained presets should drop significantly
- Acceptance rate expected to improve, especially on stress_test and high_load

## Roadmap to Production

1. **V10**: Capacity-aware reject head (this version)
2. **V11**: Evaluate longer training (8M steps) if V10 shows improvement
3. **V12**: Consider attention-based HW aggregation if mean/max pooling is insufficient
