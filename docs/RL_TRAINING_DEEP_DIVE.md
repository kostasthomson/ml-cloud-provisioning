# RL Training System - Deep Dive

This document provides an in-depth conceptual explanation of how the Reinforcement Learning training system works in this project. It covers data collection, the learning process, model architecture, state/action spaces, and the encoding pipeline.

## Table of Contents

1. [Data Source & Collection](#1-data-source--collection)
2. [Learning Process (PPO Algorithm)](#2-learning-process-ppo-algorithm)
3. [Model Architecture](#3-model-architecture)
4. [State and Action Space](#4-state-and-action-space)
5. [Encoders](#5-encoders)
6. [Episodes and Rewards](#6-episodes-and-rewards)
7. [Summary](#7-summary)

---

## 1. Data Source & Collection

### The Fundamental Concept: No Static Dataset

Unlike supervised learning (which the EnergyAwareNN uses), reinforcement learning **does not train on a pre-existing dataset**. Instead, the agent learns by **interacting with an environment** and receiving feedback.

### Key Terminology

| Term | Definition | Implementation |
|------|------------|----------------|
| **Environment** | A simulation that the agent interacts with | `CloudProvisioningEnv` (`environment.py:82`) |
| **State** | A snapshot of the world at a moment in time | `RLState` containing task + HW states + global info |
| **Action** | A decision the agent makes | Allocate to HW type N, or reject (-1) |
| **Reward** | Numerical feedback signal after taking an action | Computed by `RewardCalculator` |
| **Transition** | One (state, action, reward, next_state, done) tuple | `Experience` dataclass (`trainer.py:24-34`) |
| **Trajectory** | A sequence of transitions from start to episode end | Stored in `PPOBuffer` |

### How Data is Generated

The system uses **simulation-based data generation**. The `CloudProvisioningEnv` acts as a "world" that:

1. **Generates tasks** with realistic characteristics (`environment.py:269-333`)
   - 5 task types: small, medium, large, GPU-intensive, memory-intensive
   - Random sampling of VMs (1-16), vCPUs (2-32), memory (4-512 GB)
   - Compute work expressed in instructions (1e8 to 1e14 FLOPS)

2. **Maintains infrastructure state** (`environment.py:122-134`)
   - Tracks available/used resources per hardware type
   - Updates utilization as tasks are allocated and completed
   - Simulates task completion over time

3. **Computes energy consumption** (`environment.py:411-419`)
   ```
   Energy = (P_idle + utilization_increase × (P_max - P_idle)) × exec_time / 3600000
   ```

### The Data Collection Loop (Rollouts)

At training time, data is collected through **rollouts** - the agent interacts with the environment to gather experiences:

```
┌────────────────────────────────────────────────────────────────────┐
│                         ROLLOUT COLLECTION                         │
│                                                                    │
│   state₀ ──[encode]──► policy ──[sample]──► action₀               │
│      │                                          │                  │
│      └──────────────── env.step(action₀) ◄──────┘                  │
│                              │                                     │
│                              ▼                                     │
│                    (reward₀, state₁, done)                         │
│                              │                                     │
│                              ▼                                     │
│                     Store in PPOBuffer                             │
│                              │                                     │
│                              ▼                                     │
│                         Repeat...                                  │
└────────────────────────────────────────────────────────────────────┘
```

This happens in `trainer.py:160-226`. The agent collects 2048 transitions before each policy update.

### Alternative: External Experience Submission

The system also supports **offline RL** via the API (`rl/api.py`):
- External systems (like the C++ simulator) can submit experiences via `POST /rl/experience`
- These are stored and used to create an environment that replays them (`trainer.py:346-364`)

---

## 2. Learning Process (PPO Algorithm)

### What is Reinforcement Learning?

RL is about learning a **policy** π(a|s) - a function that maps states to actions (or action probabilities). The goal is to find a policy that maximizes **cumulative reward** over time.

### Why PPO (Proximal Policy Optimization)?

PPO is a **policy gradient** algorithm that directly optimizes the policy. The implementation uses PPO because:

1. **Stability**: It prevents destructive large policy updates
2. **Sample efficiency**: Reuses collected data for multiple gradient updates
3. **Simplicity**: Easier to tune than older algorithms like TRPO

### The Actor-Critic Architecture

The `PolicyNetwork` (`agent.py:74-167`) is an **Actor-Critic**:

| Component | Role | Output |
|-----------|------|--------|
| **Actor** (Policy) | Decides what action to take | Action probabilities |
| **Critic** (Value) | Estimates how good the current state is | State value V(s) |

Both share the same encoder but have separate "heads":
- Actor: `scorer` + `reject_head` → softmax → action probabilities
- Critic: `value_head` → scalar value estimate

### Key Concepts in the Training Loop

#### 1. Temporal Difference (TD) Error

The TD error measures the "surprise" between predicted and actual outcomes:

```
δₜ = rₜ + γ × V(sₜ₊₁) - V(sₜ)
```

Where:
- `rₜ` = reward at time t
- `γ` (gamma) = discount factor (0.99) - how much to value future rewards
- `V(s)` = value function estimate

#### 2. Generalized Advantage Estimation (GAE)

GAE (`trainer.py:51-76`) computes how much better an action was compared to average:

```python
# From PPOBuffer.finish_path()
deltas = rewards[:-1] + gamma * values[1:] * (1 - dones) - values[:-1]

# Compute advantages by walking backward through trajectory
for t in reversed(range(n)):
    advantages[t] = deltas[t] + gamma * lam * (1 - dones[t]) * last_adv
```

The **λ (lambda)** parameter (0.95) controls bias-variance tradeoff:
- λ = 0: High bias, low variance (only uses immediate TD error)
- λ = 1: Low bias, high variance (uses full Monte Carlo return)

#### 3. PPO Clipped Objective

The core innovation of PPO is the **clipped surrogate objective** (`trainer.py:291-301`):

```python
ratio = exp(new_log_prob - old_log_prob)  # How much policy changed

surr1 = ratio * advantage
surr2 = clamp(ratio, 1 - clip_range, 1 + clip_range) * advantage

policy_loss = -min(surr1, surr2)  # Take the pessimistic bound
```

This prevents the policy from changing too much in one update. If the ratio goes outside [0.8, 1.2] (with clip_range=0.2), the gradient is zeroed.

**Why this matters**: Without clipping, a single bad update could destroy a good policy. PPO ensures gradual, stable improvement.

#### 4. The Complete Loss Function

```python
# trainer.py:314
loss = policy_loss + 0.5 * value_loss + entropy_loss
```

| Component | Purpose | Coefficient |
|-----------|---------|-------------|
| Policy loss | Improve action selection | 1.0 |
| Value loss | Improve state value estimates | 0.5 |
| Entropy loss | Encourage exploration | -0.01 |

The **entropy bonus** (`trainer.py:306-307`) prevents the policy from becoming too deterministic too quickly:
```python
entropy = -(action_probs * log(action_probs)).sum()
entropy_loss = -0.01 * entropy  # Negative because we maximize entropy
```

### The Training Loop Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                              │
│                                                                 │
│  while timesteps < total_timesteps:                             │
│      │                                                          │
│      ├── 1. COLLECT 2048 transitions (rollout)                  │
│      │       └── Store: (state, action, reward, value, logprob) │
│      │                                                          │
│      ├── 2. COMPUTE ADVANTAGES (GAE-Lambda)                     │
│      │       └── Walk backward, accumulate discounted TD errors │
│      │                                                          │
│      ├── 3. PPO UPDATE (10 epochs)                              │
│      │       └── For each mini-batch (64 samples):              │
│      │           ├── Forward pass → new probs, values           │
│      │           ├── Compute clipped policy loss                │
│      │           ├── Compute value loss (MSE)                   │
│      │           ├── Compute entropy bonus                      │
│      │           ├── Backprop combined loss                     │
│      │           └── Gradient clip (max norm 0.5)               │
│      │                                                          │
│      └── 4. REPEAT                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Training Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | 3e-4 | Adam optimizer step size |
| Gamma (γ) | 0.99 | Discount factor for future rewards |
| GAE Lambda (λ) | 0.95 | Advantage estimation smoothing |
| Batch size | 64 | Samples per gradient update |
| PPO epochs | 10 | Update passes per rollout |
| Clip range (ε) | 0.2 | Policy change limit |
| Entropy coefficient | 0.01 | Exploration bonus weight |
| Value coefficient | 0.5 | Critic loss weight |
| Max gradient norm | 0.5 | Gradient clipping threshold |

---

## 3. Model Architecture

### Design Philosophy: Infrastructure-Agnostic

The model is designed to work with **any number of hardware types** without retraining. This is achieved through **modular encoding with shared weights**.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PolicyNetwork                                      │
│                                                                              │
│  ┌──────────────┐                                                            │
│  │  TaskEncoder │ ◄── task_vec (17 features)                                 │
│  │  17 → 64 → 64│                                                            │
│  └──────┬───────┘                                                            │
│         │ task_emb (64)                                                      │
│         │                                                                    │
│         ├─────────────────┬─────────────────┬─────────────────┐              │
│         │                 │                 │                 │              │
│         ▼                 ▼                 ▼                 ▼              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   HWEncoder  │  │   HWEncoder  │  │   HWEncoder  │  │  RejectHead  │     │
│  │  (shared!)   │  │  (shared!)   │  │  (shared!)   │  │   64 → 32 → 1│     │
│  │  16 → 64 → 64│  │  16 → 64 → 64│  │  16 → 64 → 64│  └──────┬───────┘     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │              │
│         │ hw_emb₁         │ hw_emb₂         │ hw_emb₃         │ reject_score │
│         │                 │                 │                 │              │
│         ▼                 ▼                 ▼                 │              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │              │
│  │    Scorer    │  │    Scorer    │  │    Scorer    │         │              │
│  │  (shared!)   │  │  (shared!)   │  │  (shared!)   │         │              │
│  │ 128 → 64 → 1 │  │ 128 → 64 → 1 │  │ 128 → 64 → 1 │         │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │              │
│         │ score₁          │ score₂          │ score₃          │              │
│         │                 │                 │                 │              │
│         └────────────────┬┴─────────────────┴─────────────────┘              │
│                          │                                                   │
│                          ▼                                                   │
│                    [score₁, score₂, score₃, reject_score]                    │
│                          │                                                   │
│                          ▼                                                   │
│                      softmax (with action masking)                           │
│                          │                                                   │
│                          ▼                                                   │
│              action_probs: [p₁, p₂, p₃, p_reject]                            │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  VALUE HEAD (Critic):                                                        │
│                                                                              │
│        task_emb ──┬── mean(hw_embs) ──► [concat] ──► ValueHead ──► V(s)     │
│                   │                        │         128 → 64 → 1            │
│                   └────────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why Shared Weights?

The `HWEncoder` and `Scorer` use the **same weights for all hardware types** (`agent.py:137-141`):

```python
for hw_vec in hw_vecs:
    hw_emb = self.hw_encoder(hw_vec)  # Same encoder for all HW types
    score = self.scorer(task_emb, hw_emb)  # Same scorer for all pairs
```

**Benefits:**
1. **Generalization**: What the model learns about evaluating one HW type transfers to others
2. **Scalability**: Add new HW types without changing the model
3. **Parameter efficiency**: Fewer parameters than having separate networks per HW type

### Action Masking

Not all actions are valid for every state. The `valid_mask` tensor (`agent.py:158-160`) sets invalid action scores to `-inf` before softmax:

```python
if valid_mask is not None:
    all_scores = all_scores.masked_fill(~full_mask, float('-inf'))
action_probs = softmax(all_scores)  # Invalid actions get probability ≈ 0
```

This ensures the agent never selects impossible actions (e.g., allocating to a HW type with insufficient resources).

### Network Layer Dimensions

| Layer | Input → Output | Purpose |
|-------|---------------|---------|
| TaskEncoder | 17 → 64 → 64 | Compress task + global features |
| HWEncoder | 16 → 64 → 64 | Compress per-HW-type features |
| Scorer | 128 → 64 → 32 → 1 | Score task-HW compatibility |
| RejectHead | 64 → 32 → 1 | Score rejection decision |
| ValueHead | 128 → 64 → 1 | Estimate state value |

---

## 4. State and Action Space

### State Space

The **state** is everything the agent observes before making a decision. The `StateEncoder` (`state_encoder.py`) converts `RLState` into numerical vectors.

#### Task Features (12 dimensions)

| Feature | Normalization | Purpose |
|---------|---------------|---------|
| num_vms | ÷ 16 | Scale of request |
| vcpus_per_vm | ÷ 64 | CPU intensity |
| memory_per_vm | ÷ 512 | Memory intensity |
| storage_per_vm | ÷ 10 | Storage needs |
| network_per_vm | ÷ 1 | Network needs |
| log₁₀(instructions) | ÷ 15 | Computational work |
| requires_accelerator | 0/1 | GPU/accelerator need |
| accelerator_rho | 0-1 | Accelerator utilization |
| num_compatible | ÷ 10 | How many HW types can run this |
| deadline_exists | 0/1 | Time constraint flag |
| deadline | ÷ 3600 | Time constraint value |
| task_intensity | normalized | Combined resource demand |

#### Global Features (5 dimensions)

| Feature | Normalization | Purpose |
|---------|---------------|---------|
| total_power_consumption | ÷ 100000 | Current system power |
| queue_length | ÷ 100 | Pending tasks |
| recent_acceptance_rate | 0-1 | Recent acceptance ratio |
| recent_avg_energy | ÷ 10 | Energy efficiency trend |
| time_of_day | % 86400 / 86400 | Temporal pattern |

#### Per-HW-Type Features (16 dimensions each)

| Feature | Purpose |
|---------|---------|
| utilization_cpu/memory/storage/network/accelerator | Current load |
| capacity ratios (available/total) | Available headroom |
| compute_capability | Processing power |
| accelerator_compute_capability | GPU power |
| power_idle, power_max | Energy model |
| num_running_tasks | Current workload |
| avg_remaining_time | When resources free up |

**All features are normalized to [0, 1]** to help neural network training.

### Action Space

The action space is **discrete** with N+1 options:

```
Actions = {hw_type_1, hw_type_2, ..., hw_type_N, REJECT}
```

In code (`agent.py:282-285`):
```python
if action_idx < num_hw:
    selected_hw_id = hw_type_ids[action_idx]  # Allocate to this HW type
else:
    selected_hw_id = None  # Reject (action = -1)
```

### The Decision Process

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE FLOW                                    │
│                                                                      │
│  1. Receive RLState (task + HW types + global)                       │
│                           │                                          │
│  2. Encode state          │                                          │
│     └── StateEncoder.encode(state)                                   │
│         ├── task_vec: (17,)                                          │
│         └── hw_list: [(hw_id, hw_vec), ...] each hw_vec: (16,)       │
│                           │                                          │
│  3. Determine valid actions                                          │
│     └── Check: compatible? resources available? accelerators?        │
│         └── valid_mask: [True, False, True, True] (example)          │
│                           │                                          │
│  4. Forward pass through PolicyNetwork                               │
│     └── Scores: [0.8, -inf, 0.5, 0.3, 0.2]                           │
│         └── Softmax: [0.45, 0.0, 0.25, 0.18, 0.12]                   │
│                           │                                          │
│  5. Select action                                                    │
│     ├── Deterministic: argmax → hw_type_1                            │
│     └── Stochastic: sample from distribution                         │
│                           │                                          │
│  6. Return RLAction                                                  │
│     └── {action: 1, hw_type_id: 1, confidence: 0.45, ...}            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Encoders

There are two types of encoders in the system serving different purposes.

### Two-Stage Encoding Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ENCODING PIPELINE                                   │
│                                                                              │
│   RLState (Pydantic object)                                                  │
│       │                                                                      │
│       ▼                                                                      │
│   ┌────────────────────────────────────────┐                                 │
│   │  STAGE 1: StateEncoder (preprocessing) │  ← Converts structured data    │
│   │           state_encoder.py             │    to fixed-size vectors       │
│   └────────────────────────────────────────┘                                 │
│       │                                                                      │
│       ├── task_vec: numpy array (17,)                                        │
│       └── hw_vecs: list of numpy arrays, each (16,)                          │
│                                                                              │
│       ▼                                                                      │
│   ┌────────────────────────────────────────┐                                 │
│   │  STAGE 2: Neural Encoders (learned)    │  ← Learns meaningful           │
│   │           agent.py                     │    representations             │
│   │           - TaskEncoder                │                                 │
│   │           - HWEncoder                  │                                 │
│   └────────────────────────────────────────┘                                 │
│       │                                                                      │
│       ├── task_emb: tensor (64,)  ← compressed task representation           │
│       └── hw_embs: tensors (64,)  ← compressed HW representations            │
│                                                                              │
│       ▼                                                                      │
│   Scorer / ValueHead / RejectHead                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Stage 1: StateEncoder (Data Preprocessing)

**File**: `state_encoder.py:17-141`

#### Purpose

Converts the structured `RLState` object (with nested Pydantic models) into **fixed-size numerical vectors** that a neural network can process.

#### Why It's Needed

Neural networks require:
1. **Numerical inputs** - can't process strings like `task_id="task_001"`
2. **Fixed dimensions** - input size must be consistent
3. **Normalized values** - features should be in similar ranges (typically 0-1)

#### What It Does

```python
# state_encoder.py:36-61
def encode(self, state: RLState) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    task_vec = self._encode_task(state.task)          # 12 features
    global_vec = self._encode_global(state.global_state)  # 5 features
    task_global_vec = concatenate([task_vec, global_vec])  # 17 total

    hw_list = []
    for hw in state.hw_types:
        hw_vec = self._encode_single_hw(hw)  # 16 features each
        hw_list.append((hw.hw_type_id, hw_vec))

    return task_global_vec, hw_list
```

#### Feature Engineering Examples

**Task Features** (`state_encoder.py:76-97`):
```python
features = [
    task.num_vms / 16.0,                    # Normalize by max expected
    task.vcpus_per_vm / 64.0,
    task.memory_per_vm / 512.0,
    np.log10(max(task.instructions, 1)) / 15.0,  # Log scale for large range
    float(task.requires_accelerator),        # Boolean → 0/1
    # ... more features
]
```

**Per-HW Features** (`state_encoder.py:113-140`):
```python
features = [
    hw.utilization_cpu,                     # Already 0-1
    hw.utilization_memory,
    capacity_ratio,                         # available / total
    np.log10(max(hw.compute_capability, 1)) / 10.0,  # Log scale
    hw.power_idle / 500.0,                  # Normalize by max watts
    # ... more features
]
```

#### Infrastructure-Agnostic Design

The StateEncoder handles **variable numbers of HW types** by returning a list rather than a fixed-size vector:

```python
# Works with 2 HW types
hw_list = [(1, vec_16), (2, vec_16)]

# Also works with 6 HW types
hw_list = [(1, vec_16), (2, vec_16), (3, vec_16), (4, vec_16), (5, vec_16), (6, vec_16)]
```

### Stage 2: Neural Network Encoders (Learned Representations)

**File**: `agent.py:24-53`

#### Purpose

Transform the preprocessed vectors into **learned embeddings** - compressed representations that capture the most relevant information for making allocation decisions.

#### Why Use Neural Encoders?

The 17-dimensional task vector and 16-dimensional HW vectors contain raw features. Neural encoders learn to:

1. **Extract patterns** - Discover which feature combinations matter
2. **Compress information** - Reduce to essential aspects (64 dims)
3. **Create comparable representations** - Task and HW embeddings live in the same "space"

#### TaskEncoder

**Definition** (`agent.py:24-37`):
```python
class TaskEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64):
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),   # 17 → 64
            nn.ReLU(),
            nn.Linear(64, embed_dim),    # 64 → 64
            nn.ReLU(),
        )
```

**What it learns**: Which task characteristics are most relevant for allocation decisions. For example, it might learn that:
- High `instructions` + `requires_accelerator` → strong signal for GPU allocation
- High `memory_per_vm` + many `num_vms` → important constraint

#### HWEncoder

**Definition** (`agent.py:40-53`):
```python
class HWEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 64):
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),   # 16 → 64
            nn.ReLU(),
            nn.Linear(64, embed_dim),    # 64 → 64
            nn.ReLU(),
        )
```

**Critical design**: The **same HWEncoder is used for all hardware types** (`agent.py:137-141`):

```python
for hw_vec in hw_vecs:
    hw_emb = self.hw_encoder(hw_vec)  # Same weights for HW type 1, 2, 3, etc.
```

**What it learns**: A general understanding of hardware characteristics:
- High `utilization` + low `available_cpus` → congested, less desirable
- High `compute_capability` + low `power_max` → energy efficient
- These patterns apply to **any** hardware type

### Complete Encoding Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE ENCODING FLOW                               │
│                                                                              │
│  INPUT: RLState                                                              │
│  ├── task: TaskState (task_id, num_vms, vcpus, memory, instructions, ...)    │
│  ├── hw_types: [HWTypeState, HWTypeState, HWTypeState]                       │
│  └── global_state: GlobalState (timestamp, power, queue, ...)                │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                      StateEncoder.encode()                            │   │
│  │                                                                       │   │
│  │   TaskState ──────► _encode_task() ────────► [0.25, 0.12, 0.06, ...]  │   │
│  │   GlobalState ────► _encode_global() ──────► [0.5, 0.03, 0.8, ...]    │   │
│  │                                       concat                          │   │
│  │                                         │                             │   │
│  │                                         ▼                             │   │
│  │                              task_vec: (17,) numpy                    │   │
│  │                                                                       │   │
│  │   HWTypeState₁ ──► _encode_single_hw() ──► hw_vec₁: (16,) numpy      │   │
│  │   HWTypeState₂ ──► _encode_single_hw() ──► hw_vec₂: (16,) numpy      │   │
│  │   HWTypeState₃ ──► _encode_single_hw() ──► hw_vec₃: (16,) numpy      │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│                              │                                               │
│                              ▼                                               │
│  ┌───────────────────────────────────────────────────────────────────────┐   │
│  │                     PolicyNetwork.forward()                           │   │
│  │                                                                       │   │
│  │   task_vec (17,) ──► TaskEncoder ──► task_emb (64,)                  │   │
│  │                          │              │                             │   │
│  │          ┌───────────────┴──────────────┼──────────────┐              │   │
│  │          │                              │              │              │   │
│  │          ▼                              ▼              ▼              │   │
│  │   hw_vec₁ (16,)                  hw_vec₂ (16,)  hw_vec₃ (16,)        │   │
│  │       │                              │              │                │   │
│  │       ▼                              ▼              ▼                │   │
│  │   HWEncoder ────────────────────► HWEncoder ──► HWEncoder            │   │
│  │   (shared)                        (shared)      (shared)             │   │
│  │       │                              │              │                │   │
│  │       ▼                              ▼              ▼                │   │
│  │   hw_emb₁ (64,)                hw_emb₂ (64,)  hw_emb₃ (64,)          │   │
│  │       │                              │              │                │   │
│  │       └──────────────┬───────────────┴──────────────┘                │   │
│  │                      │                                               │   │
│  │                      ▼                                               │   │
│  │              [task_emb, hw_emb] pairs                                │   │
│  │                      │                                               │   │
│  │                      ▼                                               │   │
│  │                   Scorer (for each pair)                             │   │
│  │                      │                                               │   │
│  │                      ▼                                               │   │
│  │              scores → softmax → action_probs                         │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  OUTPUT: action_probs [p_hw1, p_hw2, p_hw3, p_reject]                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Encoder Summary

| Encoder | Type | Input | Output | Purpose |
|---------|------|-------|--------|---------|
| **StateEncoder** | Manual (rule-based) | `RLState` object | `(17,)` + list of `(16,)` numpy arrays | Normalize and structure raw data |
| **TaskEncoder** | Neural network (learned) | `(17,)` tensor | `(64,)` tensor | Learn task representation |
| **HWEncoder** | Neural network (learned, shared) | `(16,)` tensor | `(64,)` tensor | Learn HW representation |

**Key insight**: StateEncoder does **feature engineering** (human-designed), while TaskEncoder and HWEncoder do **representation learning** (learned from data through backpropagation during training).

---

## 6. Episodes and Rewards

### What is an Episode?

An **episode** is a sequence of interactions from a starting state until a termination condition. In this system:

- **Start**: `env.reset()` initializes fresh infrastructure state
- **Steps**: Each step processes one task allocation decision
- **End conditions** (`trainer.py:207`, `environment.py:234-240`):
  - `truncated=True`: Max steps reached (2048)
  - `done=True`: All experiences exhausted (offline mode)

### When Does the Reward Function Operate?

The reward is computed **immediately after each action** (`environment.py:228`):

```python
outcome = TaskOutcome(
    task_id=task.task_id,
    action_taken=action,
    accepted=accepted,
    energy_consumed_kwh=energy,
    execution_time_sec=exec_time,
    deadline_met=exec_time <= task.deadline if task.deadline and accepted else None,
    sla_violation=exec_time > task.deadline if task.deadline and accepted else False
)

reward = self.reward_calculator.compute_reward(outcome, current_state)
```

### Reward Function Breakdown

The reward function (`reward.py:44-90`) is **energy-focused**:

```
┌──────────────────────────────────────────────────────────────────────┐
│                    REWARD COMPUTATION                                 │
│                                                                       │
│  IF task rejected:                                                    │
│      return -0.3  (rejection penalty)                                 │
│                                                                       │
│  ELSE (task accepted):                                                │
│      reward = 0                                                       │
│                                                                       │
│      ┌─ Energy Component (weight: 0.8) ─────────────────────────────┐ │
│      │  normalized = energy / baseline (0.05 kWh)                   │ │
│      │                                                               │ │
│      │  if normalized ≤ 0.5:    reward = +1.0  (excellent)          │ │
│      │  if 0.5 < norm ≤ 1.0:    reward = 1.0 - (norm - 0.5)         │ │
│      │  if 1.0 < norm ≤ 2.0:    reward = 0.5 - (norm - 1.0) × 0.75  │ │
│      │  if norm > 2.0:          reward = -0.25 - (norm - 2.0) × 0.5 │ │
│      └───────────────────────────────────────────────────────────────┘ │
│                                                                       │
│      ┌─ Energy Thresholds ──────────────────────────────────────────┐ │
│      │  if energy < 0.03 kWh:  bonus +0.3  (excellent)              │ │
│      │  if energy > 0.08 kWh:  penalty -0.2  (poor)                 │ │
│      └───────────────────────────────────────────────────────────────┘ │
│                                                                       │
│      ┌─ Adaptive Component (after 10 samples) ──────────────────────┐ │
│      │  if energy < running_mean × 0.8:  bonus +0.15                │ │
│      │  if energy > running_mean × 1.2:  penalty -0.1               │ │
│      └───────────────────────────────────────────────────────────────┘ │
│                                                                       │
│      ┌─ SLA Component (weight: 0.15) ───────────────────────────────┐ │
│      │  deadline met:      +1.0 × 0.15 = +0.15                      │ │
│      │  deadline violated: -1.0 × 0.15 = -0.15                      │ │
│      │  no deadline:       +0.3 × 0.15 = +0.045                     │ │
│      └───────────────────────────────────────────────────────────────┘ │
│                                                                       │
│      ┌─ Acceptance Bonus ───────────────────────────────────────────┐ │
│      │  +0.05 for every accepted task                               │ │
│      └───────────────────────────────────────────────────────────────┘ │
│                                                                       │
│      return clamp(total_reward, -2.0, +1.5)                           │
└──────────────────────────────────────────────────────────────────────┘
```

### Reward Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| energy_weight | 0.8 | Primary optimization target |
| sla_weight | 0.15 | Deadline compliance importance |
| rejection_penalty | 0.3 | Cost for rejecting feasible tasks |
| energy_baseline | 0.05 kWh | Reference energy level |
| energy_excellent_threshold | 0.03 kWh | Bonus threshold |
| energy_poor_threshold | 0.08 kWh | Penalty threshold |

### Reward Range

The final reward is clamped to **[-2.0, +1.5]** to prevent extreme values from destabilizing training.

The agent learns to **minimize energy** while **avoiding unnecessary rejections** and **meeting deadlines**.

---

## 7. Summary

| Aspect | Implementation |
|--------|----------------|
| **Data Source** | Simulated environment + optional API submission |
| **Algorithm** | PPO (Proximal Policy Optimization) |
| **Architecture** | Actor-Critic with shared HW encoder |
| **State** | 17 task+global features + 16 per-HW features |
| **Actions** | Discrete: allocate to HW type or reject |
| **Reward** | 80% energy + 15% SLA + penalties/bonuses |
| **Episode** | Up to 2048 allocation decisions |
| **Key Design** | Infrastructure-agnostic via shared weights |

### Key Files

| File | Purpose |
|------|---------|
| `rl/environment.py` | Gymnasium environment, task generation, energy simulation |
| `rl/trainer.py` | PPO training loop, GAE computation, policy updates |
| `rl/agent.py` | PolicyNetwork, TaskEncoder, HWEncoder, Scorer |
| `rl/state_encoder.py` | State preprocessing, feature normalization |
| `rl/reward.py` | Reward calculation with energy focus |
| `rl/schemas.py` | Pydantic data models for API and internal use |
