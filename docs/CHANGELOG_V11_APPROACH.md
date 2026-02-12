# V11: Capacity-Scaled Task Generation

## Problem

V4-V10 optimization plateaued at ~39.9% average acceptance. The bottleneck is the environment, not the policy.

Task sizes are generated from fixed distributions regardless of infrastructure capacity:
- A `large` task averages 178 vCPUs / 1,418 GB but `stress_test` only has 96 vCPUs / 384 GB total
- A `memory_intensive` task can need 1,536 GB (4x stress_test's total memory)
- Task types are sampled uniformly (20% each), so ~40% of tasks are physically impossible on constrained presets

In a real cloud, a 96-CPU cluster would never receive 15-VM / 32-vCPU jobs. Task scale correlates with infrastructure scale.

## Approach

Scale `num_vms` upper bounds proportionally to infrastructure capacity. VM sizes (`vcpus_per_vm`, `memory_per_vm`) stay fixed since they represent standardized VM types.

### Reference Point

`medium` preset (1024 total CPUs, 7168 GB) = scale 1.0.

### Scaling Formula

```
total_cpus = sum(cfg.total_cpus for all hw_configs)
total_memory = sum(cfg.total_memory for all hw_configs)
capacity_scale = min(total_cpus / 1024, total_memory / 7168)
capacity_scale = clamp(capacity_scale, 0.25, 3.0)
```

### Memory Cap

For `memory_intensive` tasks, `memory_per_vm` choices are filtered to at most 50% of total system memory. Prevents impossible memory requests on constrained presets.

## Resulting Scale Factors

| Preset | Total CPU | Total Mem | Scale | Large num_vms | Medium num_vms |
|--------|-----------|-----------|-------|---------------|----------------|
| stress_test | 96 | 384 | 0.25 | 2-4 (was 4-15) | 1-2 (was 2-7) |
| high_load | 256 | 1792 | 0.25 | 2-4 | 1-2 |
| small | 384 | 3072 | 0.38 | 2-6 | 1-3 |
| medium | 1024 | 7168 | 1.00 | 4-15 (unchanged) | 2-7 (unchanged) |
| large | 2250 | 15000 | 2.09 | 8-31 | 4-14 |

## Files Modified

| File | Change |
|------|--------|
| `rl/environment.py` | `_compute_capacity_scale()`, scaled `_generate_task()`, memory cap, `DomainRandomizedEnv.reset()` update |
| `scripts/test_rl_module.py` | `test_capacity_scaled_tasks()` |

## Expected Impact

- Eliminate physically impossible tasks on constrained presets
- Acceptance rate improvement primarily on `stress_test` and `high_load`
- Agent learns allocation policy rather than memorizing "reject everything on small clusters"
- No change to state encoder, reward function, or policy architecture

## Training Command

```bash
torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py \
    --timesteps 2000000 --output-dir results/academic_v11 \
    --use-capacity-features --domain-preset full_spectrum --curriculum \
    --lr 3e-4 --rejection-penalty 0.3 --acceptance-bonus 0.35 \
    --version-tag v11
```
