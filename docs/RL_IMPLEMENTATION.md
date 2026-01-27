# Reinforcement Learning Implementation

## Overview

This document describes the PPO-based reinforcement learning system for energy-efficient cloud resource provisioning. The system learns optimal hardware type selection policies through interaction with the CloudLightning simulator or any external environment.

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a machine learning paradigm where an **agent** learns to make decisions by interacting with an **environment**. Unlike supervised learning (which needs labeled data), RL learns from **rewards** received after taking actions.

```
┌─────────┐    action     ┌─────────────┐
│  Agent  │──────────────▶│ Environment │
│  (RL)   │◀──────────────│  (Cloud)    │
└─────────┘  state,reward └─────────────┘
```

### Key Concepts

| Term | In Our Context |
|------|----------------|
| **State** | Task requirements + HW utilization + global metrics |
| **Action** | Allocate to HW type 1-4 or Reject |
| **Reward** | Energy efficiency + SLA compliance - rejection penalty |
| **Policy** | Neural network that maps state → action probabilities |
| **Episode** | Sequence of allocation decisions (100 tasks) |

## Why RL for Cloud Provisioning?

Traditional approaches have limitations:

| Approach | Limitation |
|----------|------------|
| Heuristics (First-Fit, Best-Fit) | No learning, fixed rules |
| Supervised Learning | Needs optimal labels (unknown) |
| Optimization (ILP, MILP) | Slow, doesn't scale |

**RL advantages:**
- Learns from experience without labeled data
- Adapts to changing workload patterns
- Optimizes long-term objectives (not just immediate)
- Handles complex state spaces

## What is PPO?

**Proximal Policy Optimization (PPO)** is a reinforcement learning algorithm developed by OpenAI (2017). It's the most widely used RL algorithm for decision-making tasks.

### Core Idea

The agent learns a **policy** π(a|s) - a probability distribution over actions given a state. PPO improves this policy iteratively while preventing updates that are "too large" (which can destabilize training).

### How PPO Works

```
1. Collect experiences: (state, action, reward, next_state)
2. Compute advantage: "How much better was this action than average?"
3. Update policy with clipped objective (prevents drastic changes)
4. Repeat
```

### The Clipping Mechanism

The key innovation of PPO is the clipped objective:

```
L_clip = min(
    ratio × advantage,                    # Normal update
    clip(ratio, 1-ε, 1+ε) × advantage    # Clipped update
)

Where:
- ratio = π_new(a|s) / π_old(a|s)  (how much policy changed)
- ε = 0.2 (clip range)
- advantage = Q(s,a) - V(s)  (how good was this action)
```

This keeps the new policy "close" to the old one, ensuring stable learning.

### Why PPO for This Project

| Property | Benefit |
|----------|---------|
| Stable training | Doesn't collapse like older algorithms |
| Sample efficient | Learns from fewer interactions |
| Discrete actions | Works well with our 5-action space |
| Actor-Critic | Learns both policy and value estimation |
| Simple to tune | Fewer hyperparameters than alternatives |

### Comparison with Other RL Algorithms

| Algorithm | Type | Difference |
|-----------|------|------------|
| DQN | Value-based | Learns Q-values, no direct policy |
| A2C/A3C | Policy gradient | No clipping, less stable |
| SAC | Off-policy | For continuous actions, entropy-based |
| TRPO | Policy gradient | PPO predecessor, more complex math |

PPO is often the "default choice" for RL problems due to its balance of simplicity and performance.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     External System                              │
│  (CloudLightning Simulator, Real Cloud Environment, etc.)       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP REST API
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RL Service (FastAPI)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   /rl/      │  │  /rl/       │  │  /rl/       │              │
│  │  predict    │  │ experience  │  │  training   │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────┐            │
│  │                   RLAgent                        │            │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐   │            │
│  │  │  State    │  │  Policy   │  │  Reward   │   │            │
│  │  │  Encoder  │  │  Network  │  │ Calculator│   │            │
│  │  └───────────┘  └───────────┘  └───────────┘   │            │
│  └─────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
rl/
├── __init__.py          # Module exports
├── schemas.py           # Pydantic data models
├── state_encoder.py     # State → vector encoding
├── reward.py            # Reward computation
├── agent.py             # PPO policy network
├── environment.py       # Gymnasium environment
├── trainer.py           # PPO training loop
└── api.py               # FastAPI router
```

## State Space

The state is encoded as a normalized vector with dimensions depending on the number of HW types:

### Task Features (12 dimensions)
| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | num_vms | / 16 |
| 1 | vcpus_per_vm | / 64 |
| 2 | memory_per_vm | / 512 |
| 3 | storage_per_vm | / 10 |
| 4 | network_per_vm | / 1 |
| 5 | log10(instructions) | / 15 |
| 6 | requires_accelerator | boolean |
| 7 | accelerator_rho | [0, 1] |
| 8 | num_compatible_hw_types | / 10 |
| 9 | has_deadline | boolean |
| 10 | deadline (if present) | / 3600 |
| 11 | resource_intensity | normalized composite |

### Hardware Type Features (N × 16 dimensions, where N = number of HW types)
For each HW type:
| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | utilization_cpu | [0, 1] |
| 1 | utilization_memory | [0, 1] |
| 2 | utilization_storage | [0, 1] |
| 3 | utilization_network | [0, 1] |
| 4 | utilization_accelerator | [0, 1] |
| 5 | available_cpus / total_cpus | ratio |
| 6 | available_memory / total_memory | ratio |
| 7 | available_storage / total_storage | ratio |
| 8 | available_network / total_network | ratio |
| 9 | available_accelerators / total | ratio |
| 10 | log10(compute_capability) | / 10 |
| 11 | log10(acc_compute_capability) | / 10 |
| 12 | power_idle | / 500 |
| 13 | power_max | / 500 |
| 14 | num_running_tasks | / 100 |
| 15 | avg_remaining_time | / 3600 |

### Global Features (5 dimensions)
| Index | Feature | Normalization |
|-------|---------|---------------|
| 0 | total_power_consumption | / 100000 |
| 1 | queue_length | / 100 |
| 2 | recent_acceptance_rate | [0, 1] |
| 3 | recent_avg_energy | / 10 |
| 4 | time_of_day | timestamp % 86400 / 86400 |

## Action Space

Infrastructure-agnostic discrete action space:

| Action | Description |
|--------|-------------|
| hw_type_id | Allocate to specified HW type (e.g., 1, 2, 3...) |
| -1 | Reject the task |

The action space size is N+1 where N = number of hardware types in the environment. Actions directly map to hardware type IDs, not fixed indices.

### Action Masking

Invalid actions are masked based on:
- Task compatibility with HW type (`compatible_hw_types` field)
- Resource availability (CPU, memory, accelerators)
- Reject action (-1) is always valid

## Reward Function

```
R = w_energy × R_energy + w_sla × R_sla + R_acceptance

Where:
- R_energy = 1 - (energy_kwh / baseline)  ∈ [0, 1]
- R_sla = +1 (deadline met), -1 (SLA violation), 0 (unknown)
- R_acceptance = +0.1 (accepted), -0.2 (rejected)

Default weights:
- w_energy = 0.5
- w_sla = 0.3
```

## Policy Network Architecture

```
Input (81) → Linear(128) → ReLU → Linear(128) → ReLU
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
            Actor Head                              Critic Head
        Linear(64) → ReLU                       Linear(64) → ReLU
        Linear(5) → Softmax                     Linear(1)
              │                                       │
              ▼                                       ▼
        Action Probs π(a|s)                    State Value V(s)
```

## API Endpoints

### Inference

**POST /rl/predict**

Request:
```json
{
  "state": {
    "task": {
      "task_id": "task_001",
      "num_vms": 4,
      "vcpus_per_vm": 8,
      "memory_per_vm": 32.0,
      "instructions": 1e10,
      "compatible_hw_types": [1, 2],
      "requires_accelerator": false
    },
    "hw_types": [
      {
        "hw_type_id": 1,
        "utilization_cpu": 0.3,
        "available_cpus": 1400,
        "total_cpus": 2000,
        ...
      }
    ],
    "global_state": {
      "timestamp": 100.0,
      "total_power_consumption": 50000
    }
  },
  "deterministic": true
}
```

Response:
```json
{
  "action": {
    "action": 0,
    "action_name": "ALLOCATE_CPU",
    "confidence": 0.85,
    "action_probs": [0.85, 0.10, 0.03, 0.01, 0.01]
  },
  "state_value": 0.42,
  "inference_time_ms": 2.5
}
```

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rl/health` | GET | Service health check |
| `/rl/model` | GET | Get model info |
| `/rl/model/save` | POST | Save model to file |
| `/rl/model/load` | POST | Load model from file |

### Training

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rl/experience` | POST | Submit single experience |
| `/rl/experience/batch` | POST | Submit batch of experiences |
| `/rl/training/status` | GET | Get training status |
| `/rl/training/start` | POST | Start training from buffer |
| `/rl/training/simulate` | POST | Train on simulated environment |

### Reward Configuration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/rl/reward/config` | GET | Get reward weights |
| `/rl/reward/config` | PUT | Update reward weights |
| `/rl/reward/compute` | POST | Compute reward for outcome |

## Training

### Offline Training (Simulated Environment)

```bash
cd scripts
python train_rl_agent.py --timesteps 50000 --save-path models/rl/ppo/model.pth
```

Options:
- `--timesteps`: Total training timesteps (default: 50000)
- `--save-path`: Model save path
- `--lr`: Learning rate (default: 3e-4)
- `--batch-size`: Batch size (default: 64)
- `--epochs`: PPO epochs per update (default: 10)
- `--gamma`: Discount factor (default: 0.99)

### Online Training (With Simulator)

1. Start the RL service: `python main.py`
2. Run simulator with RL broker
3. Submit experiences via `/rl/experience`
4. Trigger training via `/rl/training/start`

## PPO Algorithm Details

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| learning_rate | 3e-4 | Adam optimizer learning rate |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE lambda for advantage estimation |
| clip_range | 0.2 | PPO clipping parameter |
| n_epochs | 10 | Epochs per PPO update |
| batch_size | 64 | Minibatch size |
| buffer_size | 2048 | Rollout buffer size |
| entropy_coef | 0.01 | Entropy bonus coefficient |
| value_coef | 0.5 | Value loss coefficient |
| max_grad_norm | 0.5 | Gradient clipping |

### Update Procedure

1. Collect 2048 timesteps of experience
2. Compute GAE advantages
3. For each of 10 epochs:
   - Shuffle data into minibatches
   - Compute policy ratio: r(θ) = π_new(a|s) / π_old(a|s)
   - Clipped policy loss: L_clip = min(r(θ)A, clip(r(θ), 1±ε)A)
   - Value loss: L_value = (V(s) - R)²
   - Entropy bonus: H = -Σ π(a|s) log π(a|s)
   - Total loss: L = -L_clip + 0.5×L_value - 0.01×H
4. Update policy parameters

## C++ Simulator Integration

### rlBroker

The `rlBroker` class in the C++ simulator:

1. Collects current state from cell resources
2. Serializes state to JSON
3. Sends POST request to `/rl/predict`
4. Parses action from response
5. Executes allocation locally
6. Falls back to `traditionalBroker` on failure

### State Serialization

```cpp
string rlBroker::buildRLStateRequest(const task& _task, double timestamp) {
  // Serialize task, hw_types, global_state to JSON
  // Send to POST /rl/predict
}
```

### Action Parsing

```cpp
int rlBroker::parseRLActionResponse(const string& response) {
  // Parse "action" field from JSON response
  // Returns 0-4 or -1 on error
}
```

## Testing

```bash
cd scripts
python test_rl_module.py
```

Tests:
- StateEncoder: Verifies 81-dim output, value ranges, action masking
- RewardCalculator: Checks reward signs for accept/reject/SLA cases
- RLAgent: Tests prediction, action probabilities, model info

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `schemas.py` | ~225 | Pydantic models for all data structures |
| `state_encoder.py` | ~200 | State encoding and action masking |
| `reward.py` | ~100 | Reward computation logic |
| `agent.py` | ~180 | PPO policy network and inference |
| `environment.py` | ~200 | Gymnasium environment wrapper |
| `trainer.py` | ~320 | PPO training algorithm |
| `api.py` | ~200 | FastAPI REST endpoints |

## Interpreting Training Results

### Understanding the Output

```
2025-12-27 02:20:06 - Training completed. Episodes: 500, Avg reward: 49.564
```

| Metric | Meaning |
|--------|---------|
| Episodes | Number of complete task sequences (100 steps each) |
| Avg Reward | Mean cumulative reward per episode |
| Timesteps | Total individual decisions made |

### Reward Breakdown Per Step

Each allocation decision receives a reward in **[-1, +1]**:

| Outcome | Typical Reward | Explanation |
|---------|----------------|-------------|
| Accept + deadline met | +0.7 to +0.9 | Good allocation, SLA satisfied |
| Accept + no deadline info | +0.4 to +0.6 | Accepted, unknown SLA status |
| Accept + SLA violation | -0.2 to +0.2 | Accepted but deadline missed |
| Reject task | -0.2 | Penalty for rejection |
| Invalid action | Prevented | Action masking blocks this |

### Calculating Per-Step Performance

```
Per-step average = Avg Episode Reward / Episode Length
Example: 49.56 / 100 = 0.496 per step
```

### Performance Benchmarks

| Avg Episode Reward | Quality | Interpretation |
|--------------------|---------|----------------|
| < 0 | Poor | Over-rejecting or wrong HW types |
| 0 - 30 | Learning | Suboptimal, still improving |
| 30 - 50 | Moderate | Reasonable policy |
| 50 - 70 | Good | Balanced acceptance/efficiency |
| > 70 | Excellent | Strong policy (or easy simulation) |

### What to Look For

**Healthy training signs:**
- Reward increases over time (learning)
- Reward stabilizes (convergence)
- Non-uniform action probabilities (learned preferences)

**Warning signs:**
- Reward decreases or oscillates wildly
- All actions have equal probability (~0.2 each)
- 100% rejection rate (action 4 always)

### Validation Steps

1. **Check action distribution:**
   ```python
   agent = RLAgent(model_path="models/rl/ppo/model.pth")
   probs = agent.get_action_probs(test_state)
   # Good: [0.45, 0.30, 0.15, 0.05, 0.05]
   # Bad:  [0.20, 0.20, 0.20, 0.20, 0.20]
   ```

2. **Compare with baseline:**
   - Run simulator with RL broker vs traditional broker
   - Compare total energy consumption
   - Compare acceptance rate

3. **Monitor real performance:**
   - Track energy per accepted task
   - Track SLA violation rate
   - Track rejection rate (should be <20% for feasible tasks)

### Training Duration Guidelines

| Timesteps | Use Case |
|-----------|----------|
| 5,000 | Quick test, verify code works |
| 50,000 | Development, initial training |
| 200,000 | Production-ready baseline |
| 1,000,000+ | Best performance, diminishing returns |

## Distributed Training

The RL module supports multi-GPU distributed training using PyTorch DistributedDataParallel (DDP) with `torchrun` as the recommended launcher.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        torchrun Launcher                         │
│            (sets RANK, LOCAL_RANK, WORLD_SIZE env vars)          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
   │   GPU 0     │      │   GPU 1     │      │   GPU N     │
   │  (rank 0)   │      │  (rank 1)   │      │  (rank N)   │
   │ ┌─────────┐ │      │ ┌─────────┐ │      │ ┌─────────┐ │
   │ │ Policy  │ │      │ │ Policy  │ │      │ │ Policy  │ │
   │ │  (DDP)  │◀┼──────┼▶│  (DDP)  │◀┼──────┼▶│  (DDP)  │ │
   │ └─────────┘ │      │ └─────────┘ │      │ └─────────┘ │
   │ ┌─────────┐ │      │ ┌─────────┐ │      │ ┌─────────┐ │
   │ │VecEnv×8 │ │      │ │VecEnv×8 │ │      │ │VecEnv×8 │ │
   │ └─────────┘ │      │ └─────────┘ │      │ └─────────┘ │
   └─────────────┘      └─────────────┘      └─────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                    NCCL Backend (GPU-to-GPU)
                  Synchronized Gradient Updates
```

### Key Components

| Component | Description |
|-----------|-------------|
| `DistributedPPOTrainer` | Multi-GPU trainer with automatic DDP setup |
| `VectorizedEnv` | Parallel environment wrapper per GPU |
| `RolloutBuffer` | GAE-Lambda advantage computation |

### Training Commands

```bash
# Single GPU (auto-detects, no torchrun needed)
python scripts/train_rl_distributed.py --timesteps 100000 --num-envs 8

# Multi-GPU with torchrun (RECOMMENDED)
torchrun --nproc_per_node=4 scripts/train_rl_distributed.py --timesteps 2000000

# Multi-GPU with custom settings
torchrun --nproc_per_node=4 scripts/train_rl_distributed.py \
    --timesteps 2000000 \
    --num-envs 8 \
    --batch-size 128 \
    --lr 1e-4 \
    --save-path models/rl/ppo/model_v2.pth

# Multi-node training (across machines)
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=MASTER_IP:29500 \
    scripts/train_rl_distributed.py --timesteps 5000000
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--timesteps` | 100000 | Total training timesteps across all GPUs |
| `--save-path` | models/rl/ppo/model_distributed.pth | Model save location |
| `--lr` | 3e-4 | Learning rate |
| `--batch-size` | 64 | PPO mini-batch size |
| `--epochs` | 10 | PPO epochs per update |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda |
| `--clip-range` | 0.2 | PPO clip range |
| `--num-envs` | 8 | Parallel environments per GPU |
| `--rollout-steps` | 256 | Steps per rollout before update |
| `--env-preset` | medium | Environment preset (small/medium/large/enterprise) |
| `--log-interval` | 5000 | Log progress every N timesteps |

### Progress Tracking

Training outputs detailed progress for each GPU and aggregated global stats:

```
======================================================================
DISTRIBUTED PPO TRAINING
======================================================================
  GPUs:                 4
  Envs per GPU:         8
  Total parallel envs:  32
  Timesteps per GPU:    500,000
  Total timesteps:      2,000,000
  Rollout steps:        256
  Batch size:           64
  PPO epochs:           10
  Learning rate:        0.0003
======================================================================
[GPU 0] Initialized - device: cuda:0, envs: 8
[GPU 1] Initialized - device: cuda:1, envs: 8
[GPU 2] Initialized - device: cuda:2, envs: 8
[GPU 3] Initialized - device: cuda:3, envs: 8
[GPU 0] Step 5,000/500,000 (1.0%) | Episodes: 12 | Reward: 0.45 | FPS: 850
[GPU 1] Step 5,000/500,000 (1.0%) | Episodes: 11 | Reward: 0.42 | FPS: 840
[GPU 2] Step 5,000/500,000 (1.0%) | Episodes: 13 | Reward: 0.48 | FPS: 855
[GPU 3] Step 5,000/500,000 (1.0%) | Episodes: 10 | Reward: 0.40 | FPS: 835
----------------------------------------------------------------------
[GLOBAL] Step 20,000/2,000,000 (1.0%)
         Total Episodes: 46 | Avg Reward: 0.44 | Combined FPS: 3,380
         Policy Loss: 0.0234 | Value Loss: 0.0156 | Entropy: 1.2345
         Elapsed: 5.9s | ETA: 586.1s
----------------------------------------------------------------------
...
[GPU 0] Training complete - 500,000 steps, 1,247 episodes
[GPU 1] Training complete - 500,000 steps, 1,238 episodes
[GPU 2] Training complete - 500,000 steps, 1,251 episodes
[GPU 3] Training complete - 500,000 steps, 1,230 episodes
======================================================================
TRAINING COMPLETE
======================================================================
  Total timesteps:  2,000,000
  Total episodes:   4,966
  Total time:       591.2s
  Throughput:       3,383 steps/sec
  Model saved to:   models/rl/ppo/model_distributed.pth
======================================================================
```

### Scaling Guidelines

| GPUs | Envs/GPU | Total Parallel Envs | Expected Speedup |
|------|----------|---------------------|------------------|
| 1    | 8        | 8                   | 1x (baseline)    |
| 2    | 8        | 16                  | ~1.8x            |
| 4    | 8        | 32                  | ~3.5x            |
| 8    | 8        | 64                  | ~6x              |

Note: Speedup is sub-linear due to gradient synchronization overhead via NCCL.

### Best Practices

1. **Use torchrun**: Always prefer `torchrun` over manual `mp.spawn` for multi-GPU training
2. **Balanced workload**: Each GPU processes `total_timesteps / world_size` steps
3. **Seed isolation**: Each GPU uses `rank * 10000 + base_seed` for environment randomization
4. **Single-process saves**: Only rank 0 saves the model to avoid file conflicts
5. **Barrier sync**: `dist.barrier()` ensures all GPUs complete before saving

## Future Improvements

1. **Experience Replay**: Add prioritized experience replay for sample efficiency
2. **Curriculum Learning**: Start with simple scenarios, increase complexity
3. **Multi-Agent**: Extend to multiple cells with coordination
4. **Model Ensemble**: Use multiple policies for robustness
5. **Offline RL**: Train from historical CSV logs without simulator
