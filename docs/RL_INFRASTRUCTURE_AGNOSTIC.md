# Infrastructure-Agnostic RL Model

This document describes the redesign of the RL model to support any number of hardware types with any configuration, making it truly infrastructure-agnostic.

## Problem Statement

The original RL model was hardcoded for exactly 4 hardware types (CPU, GPU, DFE, MIC):

```python
# Original constraints
NUM_HW_TYPES = 4
STATE_DIM = 81  # Fixed: 12 task + 4×16 HW + 5 global
action_dim = 5  # Fixed: 4 HW types + reject
hw_type_id: int = Field(..., ge=1, le=4)  # Constrained to 1-4
```

This meant:
- Could not add new hardware types (e.g., TPU, FPGA variants)
- Could not use different HW type IDs (e.g., 10, 20, 30)
- Could not handle infrastructures with fewer than 4 types
- Model had to be retrained if infrastructure changed

## Solution: Per-HW-Type Scoring Architecture

The new architecture evaluates each hardware type independently using shared weights, allowing the same model to work with any infrastructure configuration.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Infrastructure-Agnostic Policy Network                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Task Features (17 dims)                                                 │
│  ├── num_vms, vcpus, memory, storage, network (5)                       │
│  ├── instructions, requires_acc, acc_rho (3)                            │
│  ├── num_compatible, has_deadline, deadline_norm (3)                    │
│  ├── resource_intensity (1)                                             │
│  └── Global: power, queue, acceptance_rate, avg_energy, time (5)        │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────┐                                                     │
│  │  TaskEncoder    │  Linear(17→64) + ReLU + Linear(64→64) + ReLU       │
│  │  (shared)       │                                                     │
│  └────────┬────────┘                                                     │
│           │                                                              │
│           ▼                                                              │
│     task_embedding (64 dims)                                             │
│           │                                                              │
│           ├────────────────┬────────────────┬─────── ... ───┐           │
│           │                │                │                │           │
│           ▼                ▼                ▼                ▼           │
│  HW₁ Features (16)  HW₂ Features (16)  HW₃ Features (16)   HWₙ Features │
│           │                │                │                │           │
│           ▼                ▼                ▼                ▼           │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │                    HWEncoder (shared weights)                │        │
│  │          Linear(16→64) + ReLU + Linear(64→64) + ReLU        │        │
│  └─────────────────────────────────────────────────────────────┘        │
│           │                │                │                │           │
│           ▼                ▼                ▼                ▼           │
│      hw_emb₁ (64)     hw_emb₂ (64)     hw_emb₃ (64)      hw_embₙ (64)   │
│           │                │                │                │           │
│           ▼                ▼                ▼                ▼           │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │                    Scorer (shared weights)                   │        │
│  │   Input: concat(task_emb, hw_emb) = 128 dims                │        │
│  │   Linear(128→64) + ReLU + Linear(64→32) + ReLU + Linear(32→1)│       │
│  └─────────────────────────────────────────────────────────────┘        │
│           │                │                │                │           │
│           ▼                ▼                ▼                ▼           │
│        score₁           score₂           score₃           scoreₙ        │
│           │                │                │                │           │
│           └────────────────┴────────────────┴────────────────┘           │
│                                    │                                     │
│                                    ▼                                     │
│                    ┌───────────────────────────────┐                    │
│  task_embedding ──►│        RejectHead             │                    │
│                    │  Linear(64→32) + ReLU + Linear(32→1)               │
│                    └───────────────┬───────────────┘                    │
│                                    │                                     │
│                                    ▼                                     │
│                              reject_score                                │
│                                    │                                     │
│                                    ▼                                     │
│              softmax([score₁, score₂, ..., scoreₙ, reject_score])       │
│                                    │                                     │
│                                    ▼                                     │
│              Action: argmax or sample → HW_type_id or -1 (reject)       │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                              Value Head                                  │
│  Input: concat(task_emb, mean(hw_embs)) = 128 dims                      │
│  Linear(128→64) + ReLU + Linear(64→1) → State Value V(s)                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why This Works

1. **Shared Weights for Generalization**
   - `HWEncoder` uses the same weights for all HW types
   - `Scorer` uses the same weights for all task-HW pairs
   - The model learns "what makes a good allocation" abstractly, not per-HW-type

2. **Variable Action Space**
   - Actions are HW type IDs (any integer) or -1 (reject)
   - Number of actions = number of HW types + 1
   - Softmax computed over available options only

3. **Fixed Feature Dimensions**
   - Task features: always 17 dimensions
   - HW features: always 16 dimensions per type
   - Model weights are fixed regardless of how many HW types exist

4. **Mask-Based Feasibility**
   - Valid actions determined by resource availability
   - Invalid actions masked to -∞ before softmax
   - Ensures only feasible allocations are selected

## Files Changed

### `rl/schemas.py`

**Before:**
```python
hw_type_id: int = Field(..., ge=1, le=4)  # Constrained
action: int = Field(..., ge=0, le=4)      # 0-4 fixed
hw_types: List[HWTypeState] = Field(..., max_length=4)
```

**After:**
```python
hw_type_id: int = Field(..., ge=1)  # Any positive integer
action: int = Field(...)            # -1 for reject, or hw_type_id
hw_types: List[HWTypeState] = Field(..., min_length=1)  # No max

class RLAction:
    action: int           # -1 or hw_type_id
    hw_type_id: Optional[int]  # Explicit HW type
    action_probs: Dict[int, float]  # Probs keyed by hw_type_id
```

### `rl/state_encoder.py`

**Before:**
```python
def encode(self, state: RLState) -> np.ndarray:
    # Returns fixed 81-dim vector
    # Loops over [1, 2, 3, 4] explicitly
```

**After:**
```python
def encode(self, state: RLState) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    # Returns:
    # - task_vec: (17,) task + global features
    # - hw_list: [(hw_type_id, hw_vec), ...] for each HW type
    # Works with any number of HW types
```

### `rl/agent.py`

**Before:**
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=81, action_dim=5):
        self.shared = nn.Sequential(...)  # Fixed input
        self.actor = nn.Linear(hidden, 5)  # Fixed output
```

**After:**
```python
class TaskEncoder(nn.Module):
    # Encodes task → 64-dim embedding

class HWEncoder(nn.Module):
    # Encodes single HW → 64-dim embedding (shared)

class Scorer(nn.Module):
    # Scores (task_emb, hw_emb) → scalar (shared)

class PolicyNetwork(nn.Module):
    def forward(self, task_vec, hw_vecs, valid_mask):
        # Variable number of hw_vecs
        # Returns variable-length action_probs
```

### `rl/trainer.py`

**Before:**
```python
class PPOBuffer:
    # Fixed-size arrays for 5 actions
    self.masks = np.zeros((buffer_size, 5), dtype=bool)
```

**After:**
```python
@dataclass
class Experience:
    hw_vecs: List[np.ndarray]  # Variable length
    hw_type_ids: List[int]     # Actual IDs
    action_idx: int            # Index into hw_type_ids list

class PPOBuffer:
    experiences: List[Experience]  # Variable-length experiences
```

### `rl/environment.py`

**Before:**
```python
# Hardcoded 4 HW types
for hw_id in [1, 2, 3, 4]:
    hw_types.append(...)
```

**After:**
```python
REALISTIC_HW_CONFIGS = {
    'small': [2 types],
    'medium': [3 types],
    'large': [4 types],
    'enterprise': [6 types],
}

class CloudProvisioningEnv:
    def __init__(self, hw_configs=None, preset='medium'):
        # Accepts any HWTypeConfig list
```

## Feature Encoding

### Task Features (17 dimensions)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | num_vms | / 16 |
| 1 | vcpus_per_vm | / 64 |
| 2 | memory_per_vm | / 512 |
| 3 | storage_per_vm | / 10 |
| 4 | network_per_vm | / 1.0 |
| 5 | log10(instructions) | / 15 |
| 6 | requires_accelerator | 0 or 1 |
| 7 | accelerator_rho | [0, 1] |
| 8 | num_compatible_types | / 10 |
| 9 | has_deadline | 0 or 1 |
| 10 | deadline | / 3600 |
| 11 | resource_intensity | normalized |
| 12-16 | Global state (5 features) | various |

### HW Type Features (16 dimensions per type)

| Index | Feature | Normalization |
|-------|---------|---------------|
| 0-4 | Utilization (cpu, mem, storage, net, acc) | [0, 1] |
| 5-9 | Availability ratios | [0, 1] |
| 10 | log10(compute_capability) | / 10 |
| 11 | log10(acc_compute_capability) | / 10 |
| 12 | power_idle | / 500 |
| 13 | power_max | / 500 |
| 14 | num_running_tasks | / 100 |
| 15 | avg_remaining_time | / 3600 |

## Action Space

| Action | Meaning |
|--------|---------|
| -1 | Reject task |
| hw_type_id (1, 2, ..., N) | Allocate to that HW type |

The action returned includes:
- `action`: -1 or hw_type_id
- `hw_type_id`: Explicit HW type ID or None
- `action_name`: "REJECT" or "ALLOCATE_HW_{id}"
- `action_probs`: Dict mapping each hw_type_id and -1 to probability

## Test Results

All 9 tests pass, validating the infrastructure-agnostic design:

```
======================================================================
INFRASTRUCTURE-AGNOSTIC RL MODEL TEST SUITE
======================================================================

Encoder with Variable HW Types: PASSED
  - small: 2 HW types encoded successfully
  - medium: 3 HW types encoded successfully
  - large: 4 HW types encoded successfully
  - enterprise: 6 HW types encoded successfully

Agent Inference with Variable HW Types: PASSED
  - small (2 types): HW_2, conf=0.341, value=-0.035
  - medium (3 types): HW_2, conf=0.255, value=-0.036
  - large (4 types): HW_2, conf=0.203, value=-0.038
  - enterprise (6 types): HW_4, conf=0.144, value=-0.037

Stochastic Action Selection: PASSED
  - Action distribution over 100 samples:
  - REJECT: 25 (25.0%)
  - HW_1: 25 (25.0%)
  - HW_2: 22 (22.0%)
  - HW_3: 28 (28.0%)

Valid HW Types Detection: PASSED
  - Valid HW types: [2, 3]
  - Expected: [2, 3] (type 1 lacks resources)

Custom HW Configuration: PASSED
  - Custom config with 5 types (IDs: 10,20,30,40,50)
  - Action: REJECT, confidence: 0.205
  - State value: -0.019, inference time: 1.00ms

Model Save/Load: PASSED
  - Model saved and loaded successfully
  - Loaded model produces identical outputs

Edge Cases: PASSED
  - Single HW type: action=REJECT, value=0.057
  - High utilization: action=REJECT
  - Low utilization: action=REJECT

Training with Variable HW Types: PASSED
  - small: 4 episodes, avg_reward=1.347
  - medium: 4 episodes, avg_reward=11.393
  - large: 4 episodes, avg_reward=19.154

Expected Behavior Patterns: PASSED
  - Util 20%: action=ALLOCATE_HW_2, value=2.687
  - Util 50%: action=ALLOCATE_HW_2, value=2.273
  - Util 80%: action=ALLOCATE_HW_2, value=1.935

======================================================================
SUMMARY: 9 passed, 0 failed out of 9
======================================================================
```

## Usage Examples

### Basic Inference

```python
from rl.agent import RLAgent
from rl.schemas import RLState, TaskState, HWTypeState, GlobalState

agent = RLAgent()

# Create state with 3 HW types
state = RLState(
    task=TaskState(
        task_id="web-server-001",
        num_vms=4,
        vcpus_per_vm=8,
        memory_per_vm=32.0,
        instructions=1e11,
        compatible_hw_types=[1, 2, 3]
    ),
    hw_types=[
        HWTypeState(hw_type_id=1, utilization_cpu=0.3, ...),
        HWTypeState(hw_type_id=2, utilization_cpu=0.5, ...),
        HWTypeState(hw_type_id=3, utilization_cpu=0.7, ...),
    ],
    global_state=GlobalState(timestamp=43200.0)
)

action, value, time_ms = agent.predict(state, deterministic=True)
print(f"Allocate to HW type: {action.hw_type_id}")  # 1, 2, 3, or None
print(f"Confidence: {action.confidence:.3f}")
print(f"State value: {value:.3f}")
```

### Custom Infrastructure

```python
from rl.environment import HWTypeConfig, CloudProvisioningEnv

# Define custom infrastructure
custom_hw = [
    HWTypeConfig(10, "ARM-Edge", 64, 256, 1, 1, 0, 2000, 0, 15, 25),
    HWTypeConfig(20, "x86-Standard", 256, 1024, 10, 10, 0, 4400, 0, 163, 220),
    HWTypeConfig(30, "GPU-Inference", 128, 512, 5, 5, 8, 4400, 100000, 200, 350, 50, 300),
    HWTypeConfig(40, "TPU-Training", 64, 256, 2, 2, 4, 4400, 420000, 250, 450, 100, 400),
]

# Train with custom infrastructure
env = CloudProvisioningEnv(hw_configs=custom_hw, episode_length=100)
```

### API Usage

```bash
# Health check
curl http://localhost:8000/rl/health

# Predict action
curl -X POST http://localhost:8000/rl/predict \
  -H "Content-Type: application/json" \
  -d '{
    "state": {
      "task": {
        "task_id": "task_001",
        "num_vms": 2,
        "vcpus_per_vm": 4,
        "memory_per_vm": 16.0,
        "instructions": 1e10,
        "compatible_hw_types": [1, 2]
      },
      "hw_types": [
        {"hw_type_id": 1, "utilization_cpu": 0.3, ...},
        {"hw_type_id": 2, "utilization_cpu": 0.5, ...}
      ],
      "global_state": {"timestamp": 0}
    },
    "deterministic": true
  }'
```

## Model Characteristics

| Property | Value |
|----------|-------|
| Task embedding dim | 64 |
| HW embedding dim | 64 |
| Total trainable parameters | ~25,000 |
| Inference time | ~1ms per prediction |
| Min HW types | 1 |
| Max HW types | Unlimited |
| HW type ID range | Any positive integer |

## Limitations

1. **No temporal dependencies between HW types**: Each HW type is scored independently; the model doesn't consider relationships between types.

2. **Fixed feature dimensions**: If new features are needed, the encoder must be updated and the model retrained.

3. **Cold start**: Untrained model has random policy; requires training on target infrastructure for optimal performance.

## Future Improvements

1. **Attention mechanism**: Replace mean pooling with attention for HW aggregation
2. **Graph neural network**: Model relationships between HW types
3. **Meta-learning**: Faster adaptation to new infrastructure configurations
4. **Curriculum learning**: Train on simple infrastructures first, then complex ones
