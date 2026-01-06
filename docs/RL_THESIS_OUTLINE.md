# Deep Reinforcement Learning for Energy-Efficient Resource Provisioning in Heterogeneous Cloud Environments

## Thesis Outline for Master's Research

---

## 1. Problem Statement

**Research Question:**
*How can deep reinforcement learning be used to optimize resource provisioning decisions in heterogeneous cloud environments, minimizing energy consumption while meeting performance requirements for HPC workloads?*

**Why This Matters:**
- Data centers consume 1-2% of global electricity (~200 TWh/year)
- HPC workloads are growing (AI/ML training, scientific computing)
- Heterogeneous hardware (CPU, GPU, FPGA, specialized accelerators) makes optimal allocation non-trivial
- Static heuristics cannot adapt to dynamic workload patterns

**Gap in Existing Work:**
- Traditional schedulers use static heuristics (First-Fit, Best-Fit, Round-Robin)
- Existing ML approaches predict metrics but don't learn decision policies
- RL approaches exist but rarely address heterogeneous HPC with energy focus

---

## 2. Formulation as Markov Decision Process (MDP)

### 2.1 MDP Definition

An MDP is defined as a tuple: **(S, A, P, R, γ)**

| Component | Definition |
|-----------|------------|
| **S** | State space - representation of cloud system state |
| **A** | Action space - possible allocation decisions |
| **P** | Transition probability P(s'|s,a) - determined by simulator |
| **R** | Reward function R(s,a,s') - energy + SLA signal |
| **γ** | Discount factor (0.95-0.99) - balances immediate vs future rewards |

### 2.2 Episode Definition

```
Episode = Simulation run from t=0 to t=T (e.g., 24 hours simulated time)

Step = One task arrival and allocation decision
   │
   ├── Observe state s_t
   ├── Agent selects action a_t
   ├── Environment transitions to s_{t+1}
   ├── Agent receives reward r_t
   └── Repeat until episode ends
```

---

## 3. State Space Design

### 3.1 State Vector Components

```python
state = {
    # Task features (incoming task to be allocated)
    "task": {
        "num_vms": int,                    # VMs requested
        "vcpus_per_vm": int,               # CPU requirements
        "memory_per_vm": float,            # Memory requirements
        "instructions": float,             # Compute work (FLOPS)
        "implementations": [int],          # Compatible HW types [1,2,3,4]
        "requires_accelerator": bool,
        "accelerator_rho": float,          # Accelerator utilization if applicable
    },

    # Per-HW-type state (for each of 4 HW types)
    "hw_types": [
        {
            "hw_type_id": int,
            "utilization_cpu": float,      # 0.0 - 1.0
            "utilization_memory": float,   # 0.0 - 1.0
            "utilization_storage": float,  # 0.0 - 1.0
            "utilization_network": float,  # 0.0 - 1.0
            "utilization_accelerator": float,
            "available_capacity_ratio": float,  # Available / Total
            "num_running_tasks": int,
            "avg_remaining_time": float,   # Avg time until tasks complete
            "current_power_draw": float,   # Current power consumption
        }
        for hw_type in [1, 2, 3, 4]
    ],

    # Global state
    "global": {
        "total_power_consumption": float,  # Current total power (W)
        "queue_length": int,               # Tasks waiting
        "time_of_day": float,              # 0.0 - 1.0 (normalized)
        "recent_rejection_rate": float,    # Rejections in last N tasks
    }
}
```

### 3.2 State Encoding for Neural Network

```python
def encode_state(state) -> np.ndarray:
    """
    Flatten state dict into fixed-size vector for neural network.

    Task features:        ~10 values
    Per-HW-type features: 4 types × 9 features = 36 values
    Global features:      4 values
    ─────────────────────────────────
    Total:                ~50 values (fixed size)
    """
    task_vec = [
        state["task"]["num_vms"] / 16,  # Normalize
        state["task"]["vcpus_per_vm"] / 64,
        state["task"]["memory_per_vm"] / 512,
        np.log10(state["task"]["instructions"]) / 12,  # Log scale
        float(state["task"]["requires_accelerator"]),
        state["task"]["accelerator_rho"],
        # One-hot for implementations (4 values)
        *[1.0 if i in state["task"]["implementations"] else 0.0 for i in [1,2,3,4]]
    ]

    hw_vec = []
    for hw in state["hw_types"]:
        hw_vec.extend([
            hw["utilization_cpu"],
            hw["utilization_memory"],
            hw["utilization_storage"],
            hw["utilization_network"],
            hw["utilization_accelerator"],
            hw["available_capacity_ratio"],
            hw["num_running_tasks"] / 100,  # Normalize
            hw["avg_remaining_time"] / 3600,  # Normalize to hours
            hw["current_power_draw"] / 10000,  # Normalize
        ])

    global_vec = [
        state["global"]["total_power_consumption"] / 100000,
        state["global"]["queue_length"] / 50,
        state["global"]["time_of_day"],
        state["global"]["recent_rejection_rate"],
    ]

    return np.array(task_vec + hw_vec + global_vec, dtype=np.float32)
```

---

## 4. Action Space Design

### 4.1 Discrete Action Space

```python
# Simple approach: Choose one of 4 HW types + reject option
actions = {
    0: "ALLOCATE_TO_CPU",       # HW Type 1
    1: "ALLOCATE_TO_GPU",       # HW Type 2
    2: "ALLOCATE_TO_DFE",       # HW Type 3
    3: "ALLOCATE_TO_MIC",       # HW Type 4
    4: "REJECT_TASK",           # Cannot allocate
}

# Action space size = 5
```

### 4.2 Action Masking (Important!)

Not all actions are valid for every state:

```python
def get_valid_actions(state) -> List[bool]:
    """
    Returns mask of valid actions.
    Invalid actions should not be selected by agent.
    """
    mask = [False, False, False, False, True]  # Reject always valid

    task = state["task"]
    implementations = task["implementations"]

    for hw_type_id in [1, 2, 3, 4]:
        action_idx = hw_type_id - 1

        # Check implementation compatibility
        if hw_type_id not in implementations:
            continue

        # Check resource sufficiency
        hw = state["hw_types"][action_idx]
        if hw["available_capacity_ratio"] > 0.1:  # Some threshold
            mask[action_idx] = True

    return mask
```

### 4.3 Alternative: Parameterized Actions (Advanced)

```python
# If you want finer control (e.g., which specific servers)
action = {
    "hw_type": int,      # 1-4
    "allocation_strategy": int,  # 0=consolidated, 1=spread
}
# This increases complexity - start with discrete actions
```

---

## 5. Reward Function Design

### 5.1 Multi-Objective Reward

The reward must balance multiple objectives:

```python
def compute_reward(state, action, next_state, task_outcome) -> float:
    """
    Reward function balancing energy, performance, and acceptance.

    Key insight: All components should be normalized to similar scales.
    """

    # Component 1: Energy penalty (negative reward)
    # Lower energy consumption = higher reward
    energy_consumed_kwh = task_outcome["energy_kwh"]
    max_expected_energy = 10.0  # Normalize
    r_energy = -energy_consumed_kwh / max_expected_energy

    # Component 2: SLA compliance (positive reward for meeting deadline)
    if task_outcome["completed"]:
        actual_time = task_outcome["execution_time"]
        deadline = task_outcome["deadline"]
        if actual_time <= deadline:
            r_sla = 1.0  # Met deadline
        else:
            # Penalty proportional to how much deadline was missed
            lateness = (actual_time - deadline) / deadline
            r_sla = -lateness
    else:
        r_sla = 0.0  # Task still running

    # Component 3: Rejection penalty
    if action == 4:  # REJECT_TASK
        r_reject = -2.0  # Strong penalty for rejection
    else:
        r_reject = 0.0

    # Component 4: Resource efficiency bonus
    hw_type_used = action
    utilization = next_state["hw_types"][hw_type_used]["utilization_cpu"]
    # Reward for keeping utilization in efficient range (e.g., 60-80%)
    if 0.6 <= utilization <= 0.8:
        r_efficiency = 0.2
    else:
        r_efficiency = 0.0

    # Weighted combination
    α, β, γ, δ = 0.4, 0.3, 0.2, 0.1  # Tune these weights
    reward = α * r_energy + β * r_sla + γ * r_reject + δ * r_efficiency

    return reward
```

### 5.2 Reward Shaping Considerations

| Issue | Solution |
|-------|----------|
| Sparse rewards | Add intermediate rewards for efficient allocation |
| Delayed rewards | Discount factor γ handles this |
| Scale mismatch | Normalize all components to [-1, 1] range |
| Conflicting objectives | Weight parameters (α, β, γ, δ) control trade-offs |

### 5.3 Alternative: Intrinsic Motivation

```python
# Curiosity-driven exploration bonus (optional, for better exploration)
intrinsic_reward = curiosity_module(state, action, next_state)
total_reward = extrinsic_reward + 0.01 * intrinsic_reward
```

---

## 6. RL Algorithm Selection

### 6.1 Recommended: Proximal Policy Optimization (PPO)

**Why PPO:**
- Stable training (important for thesis timeline)
- Works well with discrete actions
- Handles continuous state spaces
- Good sample efficiency
- Well-documented implementations (Stable-Baselines3)

```python
from stable_baselines3 import PPO

model = PPO(
    policy="MlpPolicy",
    env=cloud_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./logs/"
)
```

### 6.2 Alternative Algorithms

| Algorithm | Pros | Cons | When to Use |
|-----------|------|------|-------------|
| **DQN** | Simple, stable | Only discrete actions | Baseline comparison |
| **A2C** | Fast, parallel | Less stable than PPO | Quick experiments |
| **SAC** | Sample efficient | Continuous actions only | If parameterized actions |
| **TD3** | Robust | Continuous actions only | If parameterized actions |

### 6.3 Neural Network Architecture

```python
# Policy Network (Actor)
policy_net = nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, num_actions),  # 5 actions
    nn.Softmax(dim=-1)
)

# Value Network (Critic)
value_net = nn.Sequential(
    nn.Linear(state_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 1)  # Single value estimate
)
```

---

## 7. Integration with Existing Simulator

### 7.1 Gymnasium Environment Wrapper

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CloudProvisioningEnv(gym.Env):
    """
    Wraps the C++ CloudLightning simulator as a Gymnasium environment.
    """

    def __init__(self, simulator_config: dict):
        super().__init__()

        # Initialize simulator (could be subprocess or Python bindings)
        self.simulator = CloudSimulator(simulator_config)

        # Define spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(50,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)  # 4 HW types + reject

        # Episode tracking
        self.current_step = 0
        self.max_steps = 10000  # Tasks per episode

    def reset(self, seed=None):
        """Reset simulator to initial state."""
        super().reset(seed=seed)
        self.simulator.reset()
        self.current_step = 0

        # Get first task and state
        state = self._get_state()
        info = {"valid_actions": self._get_valid_actions()}

        return state, info

    def step(self, action: int):
        """Execute allocation decision and advance simulation."""

        # Execute action in simulator
        task_outcome = self.simulator.allocate(action)

        # Advance simulation time (process running tasks)
        self.simulator.advance_time()

        # Get new state
        next_state = self._get_state()

        # Compute reward
        reward = self._compute_reward(action, task_outcome)

        # Check termination
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "valid_actions": self._get_valid_actions(),
            "energy_kwh": task_outcome.get("energy_kwh", 0),
            "accepted": action != 4,
        }

        return next_state, reward, terminated, truncated, info

    def _get_state(self) -> np.ndarray:
        """Extract and encode current state from simulator."""
        raw_state = self.simulator.get_state()
        return encode_state(raw_state)

    def _get_valid_actions(self) -> List[bool]:
        """Get mask of valid actions."""
        state = self.simulator.get_state()
        return get_valid_actions(state)

    def _compute_reward(self, action, outcome) -> float:
        """Compute reward for this step."""
        # Implement reward function from Section 5
        return compute_reward(action, outcome)
```

### 7.2 Simulator Communication Options

**Option A: Python Bindings (Recommended if feasible)**
```
C++ Simulator ← pybind11 → Python Environment → RL Agent
```

**Option B: Subprocess + JSON**
```
Python Environment ←→ subprocess ←→ C++ Simulator
                    (JSON over stdin/stdout)
```

**Option C: REST API (Your current approach)**
```
Python RL Agent → HTTP → Python API → Simulator Logic
                         (Rewrite simulator in Python)
```

**Option D: Shared Memory**
```
C++ Simulator ←→ Shared Memory ←→ Python Environment
              (for high-speed communication)
```

### 7.3 Simplified Python Simulator (For Initial Development)

```python
class SimplifiedCloudSimulator:
    """
    Python-native simulator for faster RL training iterations.
    Matches C++ simulator behavior but runs faster.
    """

    def __init__(self, config):
        self.hw_types = self._init_hw_types(config)
        self.running_tasks = []
        self.current_time = 0.0
        self.task_generator = TaskGenerator(config)

    def reset(self):
        self.running_tasks = []
        self.current_time = 0.0
        self.task_generator.reset()
        return self._generate_next_task()

    def allocate(self, action: int) -> dict:
        if action == 4:  # Reject
            return {"accepted": False, "energy_kwh": 0}

        hw_type = self.hw_types[action]
        task = self.current_task

        # Check if allocation is possible
        if not self._can_allocate(task, hw_type):
            return {"accepted": False, "energy_kwh": 0}

        # Allocate resources
        execution_time = self._estimate_execution_time(task, hw_type)
        energy = self._estimate_energy(task, hw_type, execution_time)

        self.running_tasks.append({
            "task": task,
            "hw_type": action,
            "start_time": self.current_time,
            "end_time": self.current_time + execution_time,
            "energy_kwh": energy
        })

        # Update HW state
        hw_type.allocate(task)

        return {
            "accepted": True,
            "energy_kwh": energy,
            "execution_time": execution_time
        }

    def advance_time(self):
        # Move to next task arrival
        self.current_time += self.task_generator.inter_arrival_time()

        # Complete finished tasks
        completed = [t for t in self.running_tasks if t["end_time"] <= self.current_time]
        for task_info in completed:
            self.hw_types[task_info["hw_type"]].deallocate(task_info["task"])
            self.running_tasks.remove(task_info)

        # Generate next task
        self.current_task = self.task_generator.generate()
```

---

## 8. Training Procedure

### 8.1 Training Loop

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Create environments
train_env = CloudProvisioningEnv(train_config)
eval_env = CloudProvisioningEnv(eval_config)

# Callbacks
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best/",
    log_path="./logs/",
    eval_freq=10000,
    n_eval_episodes=10,
    deterministic=True
)

checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./models/checkpoints/",
    name_prefix="ppo_cloud"
)

# Create and train agent
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./tensorboard/"
)

# Train for 1M steps
model.learn(
    total_timesteps=1_000_000,
    callback=[eval_callback, checkpoint_callback]
)

# Save final model
model.save("./models/final/ppo_cloud_final")
```

### 8.2 Hyperparameter Tuning

```python
# Key hyperparameters to tune
hyperparams = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "n_steps": [1024, 2048, 4096],
    "batch_size": [32, 64, 128],
    "gamma": [0.95, 0.99, 0.999],
    "gae_lambda": [0.9, 0.95, 0.99],
    "clip_range": [0.1, 0.2, 0.3],
}

# Use Optuna for hyperparameter search
import optuna

def objective(trial):
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
    # ... more hyperparameters

    model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma, ...)
    model.learn(total_timesteps=100000)

    # Evaluate
    mean_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    return mean_reward

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

### 8.3 Curriculum Learning (Optional, for Stability)

```python
# Start with easy scenarios, gradually increase difficulty
curriculum = [
    {"task_rate": 5, "hw_types": 2, "episodes": 100000},   # Easy
    {"task_rate": 10, "hw_types": 3, "episodes": 200000},  # Medium
    {"task_rate": 15, "hw_types": 4, "episodes": 300000},  # Hard
]

for stage in curriculum:
    env.update_difficulty(stage)
    model.learn(total_timesteps=stage["episodes"])
```

---

## 9. Evaluation Methodology

### 9.1 Baselines for Comparison

| Baseline | Description | Implementation |
|----------|-------------|----------------|
| **Random** | Random valid action | `random.choice(valid_actions)` |
| **Round-Robin** | Cycle through HW types | Rotate through 1,2,3,4 |
| **First-Fit** | First HW with capacity | First valid action |
| **Best-Fit** | HW with least waste | Min remaining capacity |
| **Energy-Greedy** | Lowest energy estimate | Use energy model |
| **Your Scoring Allocator** | Weighted multi-objective | Current implementation |

### 9.2 Evaluation Metrics

```python
class EvaluationMetrics:
    def __init__(self):
        self.total_energy_kwh = 0.0
        self.tasks_completed = 0
        self.tasks_rejected = 0
        self.sla_violations = 0
        self.total_execution_time = 0.0
        self.makespan = 0.0

    def compute_summary(self):
        return {
            "total_energy_kwh": self.total_energy_kwh,
            "acceptance_rate": self.tasks_completed / (self.tasks_completed + self.tasks_rejected),
            "sla_compliance": 1.0 - (self.sla_violations / max(self.tasks_completed, 1)),
            "avg_execution_time": self.total_execution_time / max(self.tasks_completed, 1),
            "makespan": self.makespan,
            "energy_per_task": self.total_energy_kwh / max(self.tasks_completed, 1),
        }
```

### 9.3 Statistical Significance

```python
from scipy import stats

def compare_methods(rl_results, baseline_results, metric="total_energy_kwh"):
    """
    Compare RL agent vs baseline using statistical tests.
    Run multiple independent episodes for each method.
    """
    rl_values = [r[metric] for r in rl_results]
    baseline_values = [r[metric] for r in baseline_results]

    # Welch's t-test (doesn't assume equal variances)
    t_stat, p_value = stats.ttest_ind(rl_values, baseline_values, equal_var=False)

    # Effect size (Cohen's d)
    cohens_d = (np.mean(rl_values) - np.mean(baseline_values)) / np.sqrt(
        (np.std(rl_values)**2 + np.std(baseline_values)**2) / 2
    )

    return {
        "rl_mean": np.mean(rl_values),
        "baseline_mean": np.mean(baseline_values),
        "improvement_pct": (np.mean(baseline_values) - np.mean(rl_values)) / np.mean(baseline_values) * 100,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "cohens_d": cohens_d,
    }
```

### 9.4 Experimental Design

```
Experiment 1: Energy Efficiency
├── Vary: Task arrival rate (5, 10, 15, 20 tasks/sec)
├── Measure: Total energy consumption
└── Compare: RL vs all baselines

Experiment 2: Scalability
├── Vary: Number of HW types (2, 3, 4)
├── Vary: Number of servers (50, 100, 200)
├── Measure: Decision latency, energy, acceptance rate
└── Compare: RL vs baselines

Experiment 3: Workload Adaptation
├── Scenario: Sudden workload change mid-episode
├── Measure: How quickly methods adapt
└── Compare: RL (should adapt) vs heuristics (static)

Experiment 4: Generalization
├── Train on workload A, test on workload B
├── Measure: Performance degradation
└── Shows: RL generalization capability

Experiment 5: Ablation Study
├── Vary: Reward function components
├── Vary: State features included
├── Shows: Which components are important
```

---

## 10. Thesis Chapter Structure

### Chapter 1: Introduction (10 pages)
- Background on cloud computing and HPC
- Energy consumption problem in data centers
- Research questions and objectives
- Thesis contributions
- Thesis outline

### Chapter 2: Literature Review (20 pages)
- Cloud resource management approaches
- Reinforcement learning fundamentals
- RL for resource management (existing work)
- Energy-aware scheduling
- Gap analysis

### Chapter 3: Problem Formulation (15 pages)
- System model (cloud environment description)
- MDP formulation (state, action, reward)
- Multi-objective optimization formulation
- Assumptions and constraints

### Chapter 4: Proposed Approach (25 pages)
- Overall architecture
- State representation design
- Action space and masking
- Reward function design
- PPO algorithm and neural network architecture
- Training procedure

### Chapter 5: Implementation (15 pages)
- Simulator description
- RL environment wrapper
- Training infrastructure
- Implementation challenges and solutions

### Chapter 6: Evaluation (25 pages)
- Experimental setup
- Baselines
- Results and analysis
- Statistical significance
- Discussion

### Chapter 7: Conclusion (10 pages)
- Summary of contributions
- Limitations
- Future work

---

## 11. Timeline (6-Month Thesis)

```
Month 1: Foundation
├── Week 1-2: Literature review on RL for resource management
├── Week 3-4: Set up RL training infrastructure (Stable-Baselines3)
└── Deliverable: Literature review draft, working RL training loop

Month 2: Environment Development
├── Week 1-2: Design state/action/reward (this document)
├── Week 3-4: Implement Gymnasium environment wrapper
└── Deliverable: Working environment with basic simulator

Month 3: Training & Debugging
├── Week 1-2: Train initial RL agent, debug reward function
├── Week 3-4: Hyperparameter tuning
└── Deliverable: Trained agent that beats random baseline

Month 4: Experiments
├── Week 1-2: Implement all baselines
├── Week 3-4: Run main experiments (energy, scalability, adaptation)
└── Deliverable: Experimental results, plots

Month 5: Analysis & Writing
├── Week 1-2: Statistical analysis, ablation studies
├── Week 3-4: Write methodology and results chapters
└── Deliverable: Draft of chapters 3-6

Month 6: Finalization
├── Week 1-2: Write introduction, literature review, conclusion
├── Week 3-4: Revisions, formatting, submission preparation
└── Deliverable: Complete thesis draft
```

---

## 12. Key Differentiators (Novelty for Thesis)

1. **Domain-Specific State Representation**: Novel encoding of heterogeneous HPC workloads and cloud state

2. **Multi-Objective Reward Shaping**: Balanced reward function for energy + SLA + acceptance

3. **Action Masking for Feasibility**: Ensures agent only selects valid allocations

4. **Heterogeneous Hardware Focus**: Specifically addresses CPU/GPU/FPGA/MIC selection

5. **Comprehensive Evaluation**: Comparison with multiple baselines including ML-based approach

---

## 13. Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Simulator too slow for RL | Use simplified Python simulator for training |
| RL doesn't converge | Start with smaller action space, simpler rewards |
| Results not statistically significant | Run many episodes (30+), use proper tests |
| Time constraints | Have fallback to simpler DQN if PPO struggles |
| Generalization fails | Include workload diversity in training |

---

## 14. Tools and Libraries

```python
# Core RL
stable-baselines3>=2.0.0
gymnasium>=0.29.0

# Deep Learning
torch>=2.0.0

# Experiment Tracking
tensorboard>=2.14.0
wandb>=0.15.0  # Optional, for cloud logging

# Hyperparameter Tuning
optuna>=3.3.0

# Analysis
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Development
pytest>=7.4.0
black>=23.0.0
```

---

## Next Steps

1. **Validate this approach with your thesis advisor**
2. **Start with simplified Python simulator** (faster iteration)
3. **Implement basic environment wrapper** (Section 7.1)
4. **Train baseline PPO agent**
5. **Iterate on reward function** (most critical component)
