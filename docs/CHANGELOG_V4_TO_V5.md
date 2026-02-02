# Changelog: Version 4 to Version 5

## Version Information

| Attribute | v4 | v5 |
|-----------|----|----|
| Date | 2026-01-31 | 2026-02-02 |
| Training Preset | constrained_first | constrained_first (with scarcity-aware rewards) |
| Reward System | Flat penalties/bonuses | Scarcity-aware dynamic scaling |
| GPU Tracking | Broken (never consumed) | Fixed (proper allocation/release) |
| Utilization Capture | Before step | After step |

---

## Analysis That Drove Changes

### Utilization Analysis Results (v4)

The following data from `results/academic_v4/utilization_analysis/analysis_summary.json` revealed critical issues:

```json
{
  "small": {
    "avg_acceptance_rate": 0.3096,
    "avg_capacity_rejections": 128.0,
    "avg_policy_rejections": 217.2,
    "capacity_rejection_ratio": 0.372
  },
  "medium": {
    "avg_acceptance_rate": 0.4888,
    "avg_capacity_rejections": 143.0,
    "avg_policy_rejections": 112.6,
    "capacity_rejection_ratio": 0.559
  },
  "large": {
    "avg_acceptance_rate": 0.6448,
    "avg_capacity_rejections": 135.8,
    "avg_policy_rejections": 41.8,
    "capacity_rejection_ratio": 0.767
  },
  "high_load": {
    "avg_acceptance_rate": 0.2628,
    "avg_capacity_rejections": 150.8,
    "avg_policy_rejections": 217.8,
    "capacity_rejection_ratio": 0.409
  },
  "stress_test": {
    "avg_acceptance_rate": 0.13,
    "avg_capacity_rejections": 134.2,
    "avg_policy_rejections": 300.8,
    "capacity_rejection_ratio": 0.308
  }
}
```

### Key Findings

#### 1. Policy Rejections Dominate on Constrained Environments

| Preset | Total Rejections | Policy Rejections | Policy Rej % |
|--------|------------------|-------------------|--------------|
| stress_test | 435 | 300.8 | **69.1%** |
| small | 345.2 | 217.2 | **62.9%** |
| high_load | 368.6 | 217.8 | **59.1%** |
| medium | 255.6 | 112.6 | 44.1% |
| large | 177.6 | 41.8 | 23.5% |

**Interpretation**: On constrained environments (stress_test, small, high_load), the model chose to reject tasks (policy rejection) even when resources were available. This indicates the model learned to be overly conservative.

#### 2. Capacity Rejection Ratio Pattern

Higher ratio means the model attempts allocations until hitting actual capacity limits:

- **large: 76.7%** - Model pushes capacity (good behavior)
- **medium: 55.9%** - Moderate
- **high_load: 40.9%** - Model gives up too early
- **small: 37.2%** - Model gives up too early
- **stress_test: 30.8%** - Model gives up far too early

**Interpretation**: On larger presets, most rejections are due to actual capacity limits. On smaller presets, the model preemptively rejects before trying.

#### 3. GPU Utilization Always Zero

Visual inspection of utilization figures showed GPU utilization flat at 0% for all GPU-capable hardware types, despite tasks being allocated to them.

**Root Cause**: Code inspection revealed accelerators were checked for availability but never consumed or released:

```python
# v4 Bug: Accelerators checked but not consumed
if can_allocate and task.requires_accelerator:
    can_allocate = hw_state['available_accelerators'] >= task.num_vms
    # BUG: available_accelerators never decremented!

if can_allocate:
    hw_state['available_cpus'] -= cpus_needed
    hw_state['available_memory'] -= mem_needed
    # BUG: No accelerator decrement here!
```

#### 4. Utilization Captured at Wrong Time

```python
# v4: Captured BEFORE the step (shows previous state)
hw_utils_before = self._capture_hw_utilizations(state)
next_state, reward, done, truncated, info = self.env.step(selected_hw_id)
```

This meant utilization shown in charts was always one step behind actual resource consumption.

---

## Changes Implemented

### 1. GPU/Accelerator Tracking Fix

**File**: `rl/environment.py`

**v4 Code** (lines 229-247):
```python
if can_allocate:
    accepted = True
    hw_state['available_cpus'] -= cpus_needed
    hw_state['available_memory'] -= mem_needed
    hw_state['running_tasks'] += 1
    self.accepted_count += 1

    cfg = next(c for c in self.hw_configs if c.hw_type_id == action)
    exec_time = self._estimate_exec_time(task, cfg)
    energy = self._estimate_energy(task, cfg, exec_time)
    self.total_energy += energy

    self.running_tasks.append({
        'hw_type_id': action,
        'cpus': cpus_needed,
        'memory': mem_needed,
        'remaining_time': exec_time,
        'energy': energy
    })
```

**v5 Code**:
```python
if can_allocate:
    accepted = True
    hw_state['available_cpus'] -= cpus_needed
    hw_state['available_memory'] -= mem_needed
    hw_state['running_tasks'] += 1
    self.accepted_count += 1

    # NEW: Consume accelerators
    accs_needed = task.num_vms if task.requires_accelerator else 0
    if accs_needed > 0:
        hw_state['available_accelerators'] -= accs_needed

    cfg = next(c for c in self.hw_configs if c.hw_type_id == action)
    exec_time = self._estimate_exec_time(task, cfg)
    energy = self._estimate_energy(task, cfg, exec_time)
    self.total_energy += energy

    self.running_tasks.append({
        'hw_type_id': action,
        'cpus': cpus_needed,
        'memory': mem_needed,
        'accelerators': accs_needed,  # NEW: Track accelerators
        'remaining_time': exec_time,
        'energy': energy
    })
```

**v4 Release Code** (lines 480-485):
```python
for i in reversed(completed):
    task = self.running_tasks.pop(i)
    hw = self.hw_states[task['hw_type_id']]
    hw['available_cpus'] += task['cpus']
    hw['available_memory'] += task['memory']
    hw['running_tasks'] -= 1
```

**v5 Release Code**:
```python
for i in reversed(completed):
    task = self.running_tasks.pop(i)
    hw = self.hw_states[task['hw_type_id']]
    hw['available_cpus'] += task['cpus']
    hw['available_memory'] += task['memory']
    hw['available_accelerators'] += task.get('accelerators', 0)  # NEW
    hw['running_tasks'] -= 1
```

### 2. Minimum Execution Time

**File**: `rl/environment.py`

**Problem**: Tasks completed in < 0.1 seconds while time steps were 0.5-2.0 seconds, causing instant resource release.

**v4 Code**:
```python
def _estimate_exec_time(self, task: TaskState, cfg: HWTypeConfig) -> float:
    if task.requires_accelerator and cfg.accelerator_compute > 0:
        compute = cfg.accelerator_compute * task.num_vms * task.accelerator_rho
    else:
        compute = cfg.compute_capability * task.num_vms * task.vcpus_per_vm

    base_time = task.instructions / max(compute * 1e6, 1)
    # Returns ~0.01 seconds for small tasks!
    return base_time
```

**v5 Code**:
```python
def _estimate_exec_time(self, task: TaskState, cfg: HWTypeConfig) -> float:
    if task.requires_accelerator and cfg.accelerator_compute > 0:
        compute = cfg.accelerator_compute * task.num_vms * task.accelerator_rho
    else:
        compute = cfg.compute_capability * task.num_vms * task.vcpus_per_vm

    base_time = task.instructions / max(compute * 1e6, 1)

    # NEW: Minimum 5-15 seconds so tasks persist across multiple steps
    min_exec_time = 5.0 + np.random.uniform(0, 10.0)
    base_time = max(base_time, min_exec_time)

    return base_time
```

### 3. Scarcity-Aware Reward System

**File**: `rl/reward.py`

**Rationale**: The v4 model learned to reject tasks because the rejection penalty (-0.8) was fixed regardless of whether resources were actually available. This caused conservative behavior on constrained environments.

**v4 Reward Logic**:
```python
def compute_reward(self, outcome: TaskOutcome, state: Optional[RLState] = None) -> float:
    if not outcome.accepted:
        return -self.rejection_penalty  # Always -0.8, regardless of resource state

    reward = self.acceptance_bonus  # Always +0.3
    # ... energy and SLA calculations
```

**v5 Reward Logic**:
```python
def compute_reward(self, outcome: TaskOutcome, state: Optional[RLState] = None) -> float:
    scarcity = self._compute_scarcity(state) if state and self.scarcity_aware else 0.5

    if not outcome.accepted:
        base_penalty = self.rejection_penalty
        if self.scarcity_aware and state:
            # When resources available (low scarcity), penalty is higher
            # When resources scarce (high scarcity), penalty is lower
            available_capacity = 1.0 - scarcity
            penalty_scale = 1.0 + available_capacity * (self.scarcity_rejection_scale - 1.0)
            return -base_penalty * penalty_scale
        return -base_penalty

    # Acceptance bonus scales with scarcity
    base_bonus = self.acceptance_bonus
    if self.scarcity_aware:
        # Higher bonus for accepting when resources are scarce
        bonus_scale = 1.0 + scarcity * (self.scarcity_acceptance_scale - 1.0)
        reward = base_bonus * bonus_scale
    else:
        reward = base_bonus
    # ... rest of reward calculation

def _compute_scarcity(self, state: RLState) -> float:
    """Compute resource scarcity (0 = abundant, 1 = scarce)."""
    if not state or not state.hw_types:
        return 0.5

    cpu_utils = [hw.utilization_cpu for hw in state.hw_types]
    mem_utils = [hw.utilization_memory for hw in state.hw_types]

    avg_cpu_util = sum(cpu_utils) / len(cpu_utils)
    avg_mem_util = sum(mem_utils) / len(mem_utils)

    # Scarcity is the max of CPU/memory utilization
    scarcity = max(avg_cpu_util, avg_mem_util)
    return min(1.0, max(0.0, scarcity))
```

**Reward Behavior Comparison**:

| Scenario | v4 Reward | v5 Reward | Change |
|----------|-----------|-----------|--------|
| Reject (20% utilization) | -0.80 | -1.12 | -0.32 (more penalty) |
| Reject (90% utilization) | -0.80 | -0.84 | -0.04 (similar) |
| Accept (20% utilization) | +0.30 | +0.36 | +0.06 |
| Accept (90% utilization) | +0.30 | +0.57 | +0.27 (more bonus) |

### 4. Utilization Capture Timing Fix

**File**: `scripts/utilization_analysis.py`

**v4 Code**:
```python
hw_utils_before = self._capture_hw_utilizations(state)  # BEFORE step
next_state, reward, done, truncated, info = self.env.step(selected_hw_id)
# ...
step_metrics = StepMetrics(
    hw_utilizations=hw_utils_before,  # Shows previous state
    # ...
)
```

**v5 Code**:
```python
next_state, reward, done, truncated, info = self.env.step(selected_hw_id)
hw_utils_after = self._capture_hw_utilizations(next_state)  # AFTER step
# ...
step_metrics = StepMetrics(
    hw_utilizations=hw_utils_after,  # Shows current state
    # ...
)
```

### 5. Training Script Updates

**File**: `scripts/train_rl_distributed.py`

**New Arguments**:
```python
parser.add_argument('--scarcity-aware', action='store_true', default=True,
                    help='Enable scarcity-aware rewards (default: True)')
parser.add_argument('--rejection-penalty', type=float, default=0.8,
                    help='Base rejection penalty (default: 0.8)')
parser.add_argument('--scarcity-rejection-scale', type=float, default=1.5,
                    help='Rejection penalty scale when resources available (default: 1.5)')
parser.add_argument('--scarcity-acceptance-scale', type=float, default=2.0,
                    help='Acceptance bonus scale under scarcity (default: 2.0)')
```

**Reward Config Passed to Trainer**:
```python
reward_config = {
    'scarcity_aware': args.scarcity_aware,
    'rejection_penalty': args.rejection_penalty,
    'scarcity_rejection_scale': args.scarcity_rejection_scale,
    'scarcity_acceptance_scale': args.scarcity_acceptance_scale,
}

results = trainer.train(
    # ... other args
    reward_config=reward_config,
)
```

### 6. Environment Configuration Updates

**File**: `rl/environment.py`

**New Parameter**:
```python
def __init__(
    self,
    # ... existing params
    reward_config: Optional[Dict[str, Any]] = None  # NEW
):
    # ...
    self.reward_calculator = RewardCalculator(**(reward_config or {}))
```

### 7. Distributed Trainer Updates

**File**: `rl/distributed_trainer.py`

**New Parameter in train()**:
```python
def train(
    self,
    # ... existing params
    reward_config: Optional[Dict[str, Any]] = None,  # NEW
) -> Dict[str, Any]:
    # ...
    if domain_randomization:
        vec_env = VectorizedEnv(
            num_envs,
            domain_randomization=True,
            domain_preset=domain_preset,
            curriculum=curriculum,
            reward_config=reward_config  # NEW
        )
    else:
        vec_env = VectorizedEnv(num_envs, preset=env_preset, reward_config=reward_config)
```

---

## Expected Impact

### Acceptance Rate Improvements

Based on the reward changes, we expect:

1. **Constrained environments (stress_test, high_load, small)**:
   - Reduced policy rejections due to higher penalty when resources available
   - Model will attempt more allocations before giving up

2. **All environments**:
   - Better GPU utilization tracking
   - More realistic utilization patterns in analysis

### Quantitative Predictions

| Preset | v4 Acceptance | Expected v5 Acceptance | Expected Change |
|--------|---------------|------------------------|-----------------|
| stress_test | 13.0% | 18-22% | +5-9% |
| small | 31.0% | 38-45% | +7-14% |
| high_load | 27.0% | 32-38% | +5-11% |
| medium | 50.6% | 52-58% | +2-7% |
| large | 65.3% | 66-70% | +1-5% |

---

## Training Command (v5)

```bash
python scripts/train_rl_distributed.py \
    --timesteps 100000 \
    --domain-randomization \
    --domain-preset constrained_first \
    --scarcity-aware \
    --scarcity-rejection-scale 1.5 \
    --scarcity-acceptance-scale 2.0 \
    --save-path models/rl/ppo/model_v5.pth
```

---

## Files Modified

| File | Changes |
|------|---------|
| `rl/environment.py` | GPU tracking fix, min exec time, reward_config param |
| `rl/reward.py` | Scarcity-aware reward system |
| `rl/distributed_trainer.py` | reward_config parameter passing |
| `scripts/train_rl_distributed.py` | New CLI arguments for reward config |
| `scripts/utilization_analysis.py` | Capture timing fix, GPU display, layout fixes |

---

## Verification

Scarcity-aware rewards verified with test:

```
Rejection (abundant): -1.120 (higher penalty)
Rejection (scarce):   -0.840 (lower penalty)
Acceptance (scarce):  +1.570 (higher bonus)
Acceptance (abundant):+1.360 (normal bonus)

Rejection penalty difference: -0.280
Acceptance bonus difference:  +0.210
```
