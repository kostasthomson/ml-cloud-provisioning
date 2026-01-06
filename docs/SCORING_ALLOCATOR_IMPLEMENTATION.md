# Multi-Objective Scoring Allocator Implementation Log

## Overview

Implementation of a multi-objective scoring-based resource allocation system that evaluates hardware types using a weighted sum of 5 metrics: energy consumption, execution time, network utilization, RAM utilization, and storage utilization.

## Implementation Date
2024-12-27

## Files Created/Modified

### New Files

1. **`entities/allocator/scoring_allocator.py`**
   - `ExecutionTimeEstimator`: Estimates execution time with partial allocation model
   - `EnergyEstimator`: Calculates energy consumption based on power model
   - `UtilizationEstimator`: Calculates post-allocation utilization for network, RAM, storage
   - `ScoringAllocator`: Main allocator class with weighted scoring logic

2. **`tests/test_scoring_allocator.py`**
   - Integration tests for all new endpoints
   - Tests for weights configuration, multi-implementation scoring, ongoing tasks

### Modified Files

1. **`entities/schemas.py`** - Added new Pydantic models:
   - `ResourceUsage`: Resource consumption by a task
   - `OngoingTask`: Currently running task information
   - `HardwareTypeStatus`: Extended HW status with ongoing tasks
   - `ScoringWeights`: Configurable weights (must sum to 1.0)
   - `ScoringTaskImplementation`: Task implementation for scoring
   - `ScoringAllocationRequest`: Request schema
   - `MetricBreakdown`: Per-metric breakdown (raw, normalized, weighted)
   - `HWScore`: Score for a (impl, hw) combination
   - `ScoringAllocationResponse`: Response schema with full breakdown

2. **`entities/__init__.py`** - Added exports for new schemas and `ScoringAllocator`

3. **`entities/allocator/__init__.py`** - Added `ScoringAllocator` export

4. **`main.py`** - Added:
   - `scoring_allocator` global instance initialization
   - `GET /weights` - Get global weights
   - `PUT /weights` - Set global weights
   - `POST /allocate_scoring` - Multi-objective scoring allocation
   - `GET /scoring_statistics` - Get scoring allocator statistics

## Architecture

### Execution Time Estimation (Partial Allocation Model)

```
Available Capacity Timeline:
├─ T=0: Limited capacity (ongoing tasks using resources)
├─ T=T1: Task 1 completes, capacity increases
├─ T=T2: Task 2 completes, capacity increases further
└─ Continue until new task's instructions fully processed

Effective Rate = min(available_compute, task_compute_need)
```

The estimator:
1. Builds a timeline of when ongoing tasks complete
2. Tracks available compute capacity over time
3. Simulates phased execution as capacity changes
4. Returns total execution time

### Multi-Objective Scoring

```
For each candidate (implementation, hardware):
1. Check compatibility (impl requires accelerator → HW must have accelerators)
2. Check resource sufficiency (CPU, RAM, storage, network, accelerators)
3. Calculate raw metrics:
   - Energy consumption (kWh)
   - Execution time (seconds)
   - Post-allocation network utilization
   - Post-allocation RAM utilization
   - Post-allocation storage utilization
4. Normalize metrics across all feasible candidates [0, 1]
5. Apply weights: score = Σ(weight_i × normalized_metric_i)
6. Select candidate with lowest score
```

### Weight Configuration

Two-tier system:
- **Global weights**: Set via `PUT /weights`, used when no per-request weights provided
- **Per-request weights**: Optional in `ScoringAllocationRequest`, override global

Default weights:
- energy: 0.25
- exec_time: 0.25
- network: 0.20
- ram: 0.15
- storage: 0.15

## API Endpoints

### GET /weights
Returns current global scoring weights.

### PUT /weights
Updates global weights. Request body:
```json
{
  "energy": 0.30,
  "exec_time": 0.30,
  "network": 0.15,
  "ram": 0.15,
  "storage": 0.10
}
```

### POST /allocate_scoring
Main scoring allocation endpoint. Request:
```json
{
  "timestamp": 100.0,
  "task_id": "task_001",
  "num_vms": 4,
  "implementations": [
    {
      "impl_id": 1,
      "instructions": 1e10,
      "vcpus_per_vm": 4,
      "memory_per_vm": 16.0,
      "requires_accelerator": false
    }
  ],
  "hw_types": [
    {
      "hw_type_id": 1,
      "hw_type_name": "CPU-only",
      "num_servers": 100,
      "total_cpus": 2000,
      "total_memory": 12800,
      "total_storage": 100,
      "total_network": 40,
      "available_cpus": 1500,
      "available_memory": 10000,
      "available_storage": 80,
      "available_network": 30,
      "compute_capability_per_cpu": 4400,
      "cpu_idle_power": 163,
      "cpu_max_power": 220,
      "ongoing_tasks": []
    }
  ],
  "weights": null
}
```

Response:
```json
{
  "success": true,
  "selected_hw_type_id": 1,
  "selected_impl_id": 1,
  "total_score": 0.0,
  "estimated_exec_time_sec": 142045.45,
  "estimated_energy_kwh": 7.0117,
  "all_scores": [...],
  "weights_used": {...},
  "reason": "Optimal score 0.0000: HW1 with impl1",
  "timestamp": 100.0
}
```

### GET /scoring_statistics
Returns scoring allocator statistics including allocation count, rejections, and current global weights.

## Testing

### Unit Tests (Direct Allocator)
```python
from entities import ScoringAllocator, ScoringAllocationRequest, ...
allocator = ScoringAllocator()
result = allocator.allocate(request)
```

### Integration Tests
```bash
cd ml-cloud-provisioning
python tests/test_scoring_allocator.py
```

Tests cover:
1. Get/set weights
2. Basic single-implementation allocation
3. Multi-implementation with multiple HW types
4. Allocation with ongoing tasks
5. Per-request weight override
6. Statistics endpoint

## Formulas

### Execution Time
```
effective_rate = min(available_compute, task_compute_need)
time_to_complete = remaining_instructions / effective_rate
```

### Energy Consumption
```
cpu_power = idle_power + utilization × (max_power - idle_power)
acc_power = idle_power + rho × (max_power - idle_power) × num_vms
energy_kwh = (cpu_power + acc_power) × exec_time_sec / 3,600,000
```

### Utilization (post-allocation)
```
util = (current_used + task_requirement) / total_capacity
```

### Normalization
```
normalized = (value - min) / (max - min)
```
If all values equal: normalized = 0 (no preference)

### Weighted Score
```
score = Σ(weight_i × normalized_i)
```
Lower score = better choice

## Notes

- The service is system-agnostic; all state comes from request payload
- Ongoing tasks list allows time-aware execution estimation
- Implementation compatibility: impl requiring accelerator only matches HW with accelerators
- Pydantic v2 used with `model_dump()` instead of deprecated `dict()`
