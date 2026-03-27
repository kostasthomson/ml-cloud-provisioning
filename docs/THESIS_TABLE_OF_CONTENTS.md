# Table of Contents
# Deep Reinforcement Learning for Energy-Efficient Resource Provisioning in Heterogeneous Cloud Environments

---

## 1 Introduction

### 1.1 Problem

#### 1.1.1 Data center energy consumption and environmental impact
- Data center energy consumption (≈1–2% of global electricity use, ~200 TWh/year).

#### 1.1.2 Growth of HPC workloads (AI/ML training, scientific computing)
- Increasing demand from AI/ML training and large-scale scientific computing.

#### 1.1.3 Management complexity of heterogeneous hardware (CPU, GPU, FPGA, MIC)
- Resource provisioning across heterogeneous accelerators (CPU/GPU/FPGA/MIC) increases scheduling complexity.

#### 1.1.4 Limitations of static heuristic methods
- Static, hand-crafted heuristics struggle under dynamic workloads and heterogeneous constraints.

### 1.2 Objectives

#### 1.2.1 Develop a reinforcement learning system for energy-efficient allocation
- Train an RL agent to select the most suitable hardware type per task.

#### 1.2.2 Minimize energy consumption while satisfying SLAs
- Reduce energy per accepted task while meeting deadline constraints.

#### 1.2.3 Create an infrastructure-agnostic model
- Learn policies that generalize across different capacity scales and deployments.

#### 1.2.4 Comparative evaluation with traditional methods (Heuristics, Scoring Allocator)
- Compare PPO against heuristic routing policies and a multi-objective scoring allocator.

### 1.3 Research Questions – Hypotheses

#### 1.3.1 Research Questions (RQ1–RQ3): Optimization, State Features, Generalization
- **RQ1:** Can RL optimize resource provisioning decisions better than static heuristics?
- **RQ2:** How do state representation choices affect performance?
- **RQ3:** Can the learned policy generalize across environments of different capacity/scale?

#### 1.3.2 Research Hypotheses (H1–H2): PPO performance and Domain Randomization
- **H1:** PPO achieves higher acceptance rate (or better energy–SLA trade-off) than heuristics.
- **H2:** Domain randomization improves generalization to unseen capacity regimes.

### 1.4 Contribution

#### 1.4.1 Innovative state representation for heterogeneous clouds (State Encoder v3)
- Capacity-aware encoding for scale normalization and “task fit” modeling.

#### 1.4.2 Multi-objective reward function (Energy, SLA, Acceptance)
- Reward combines energy efficiency, SLA compliance, and acceptance behavior.

#### 1.4.3 Action Masking mechanism for guaranteed valid decisions
- Masks infeasible actions to prevent invalid allocations by construction.

#### 1.4.4 End-to-end system architecture (Simulator ↔ RL Agent ↔ REST API)
- Integrated workflow: simulation → training/evaluation → deployment via API.

### 1.5 Key Terminology

#### 1.5.1 Reinforcement Learning (RL) and Markov Decision Processes (MDP)
- MDP components: state, action, transition dynamics, reward, discount factor.

#### 1.5.2 Proximal Policy Optimization (PPO) algorithm
- On-policy actor-critic method with clipped objective.

#### 1.5.3 Hardware types and Service Level Agreements (SLA)
- Hardware-type specific capabilities/constraints; SLA deadlines and compliance metrics.

### 1.6 Thesis structure

---

## 2 Relevant work and background

### 2.1 Computational clouds and resource provisioning

#### 2.1.1 Heterogeneous cloud environments and accelerators
- Accelerators and heterogeneous compute resources (CPU/GPU/FPGA, etc.).

#### 2.1.2 Traditional routing/scheduling methods (First-Fit, Best-Fit, Round-Robin)
- Baseline heuristics for placing tasks on available resources.

#### 2.1.3 The CloudLightning project and SOSM Broker architecture
- SOSM broker model and its role in heterogeneous resource management.

### 2.2 Energy efficiency and modeling

#### 2.2.1 Energy consumption models and power estimation
- Example linear utilization model:  
  - **E = (P_idle + util × (P_max − P_idle)) × duration**

#### 2.2.2 Green Computing and optimization techniques
- Energy-aware provisioning and data-center efficiency practices.

### 2.3 Reinforcement Learning

#### 2.3.1 Mathematical foundation of MDP (S, A, P, R, γ)
- Formal definition of MDP and objective of expected discounted return.

#### 2.3.2 Policy Gradient algorithms and Actor-Critic architectures
- Policy gradient family; actor-critic separation of policy/value estimation.

#### 2.3.3 Proximal Policy Optimization (PPO) and GAE
- PPO with clipped surrogate objective; advantage estimation via GAE.

### 2.4 Machine learning techniques for resource provisioning

#### 2.4.1 Supervised learning for load prediction vs RL for dynamic allocation
- Forecasting workloads vs learning closed-loop control policies.

#### 2.4.2 RL vs mathematical programming methods (ILP, MILP)
- Trade-offs between learned policies and optimization-based schedulers.

### 2.5 Progress beyond technological evolution
- Motivation for adaptive, learning-based decision-making beyond incremental hardware improvements.

---

## 3 Methodology

### 3.1 Problem formulation as an MDP

#### 3.1.1 Definition of state and action spaces
- State representation (tasks + system + hardware-type features).
- Discrete action selection among hardware types (plus reject).

#### 3.1.2 Transition and reward functions
- Transition dynamics driven by simulator evolution.
- Reward shaped by energy, SLA, and acceptance outcomes.

### 3.2 State space design

#### 3.2.1 Task features (12 dimensions)
- Examples: num_vms, vcpus_per_vm, memory_per_vm, instructions, compatibility flags, deadline-related info.

#### 3.2.2 Hardware type features (N × 16 dimensions)
- Utilization/capacity metrics, power model parameters, compute capabilities per hardware type.

#### 3.2.3 Global system features
- Examples: total power consumption, queue length, acceptance statistics.

#### 3.2.4 Resource scarcity features (Scarcity Features – v2)
- Aggregated scarcity indicators (e.g., utilization, min capacity ratios).

#### 3.2.5 Capacity features (Capacity Features – v3)
- Scale normalization and “task fit” ratios to mitigate scale-related failure modes.

### 3.3 Action space design

#### 3.3.1 Discrete actions and Action Masking
- **N + 1** actions (N hardware types + reject).
- Mask infeasible actions to guarantee validity.

### 3.4 Reward function

#### 3.4.1 Components: Energy (R_energy), SLA (R_sla), Acceptance (R_acceptance)
- **R_energy:** energy efficiency signal.
- **R_sla:** deadline/SLA compliance signal.
- **R_acceptance:** acceptance/rejection shaping.

#### 3.4.2 Weighted combination and weighting coefficients
- Weighted reward composition (with tuned coefficients per component).

### 3.5 The PPO algorithm and the neural network

#### 3.5.1 Actor-Critic network architecture
- Shared backbone with separate actor/critic heads.

#### 3.5.2 Clipped objective function and hyperparameters
- PPO clip range, learning rate, γ, λ (GAE), batch size, epochs, etc.

### 3.6 Training techniques

#### 3.6.1 Domain Randomization (Presets: small, medium, large, stress_test)
- Training across multiple capacity presets to improve robustness.

#### 3.6.2 Curriculum Learning and Distributed Training (Multi-GPU)
- Progressive difficulty scheduling (curriculum).
- Multi-GPU / distributed training for scalable learning.

---

## 4 Implementation

### 4.1 System architecture

#### 4.1.1 Data flow: Simulator ↔ RL Agent ↔ REST API
- End-to-end pipeline connecting simulation, training/inference, and external consumption via API.

### 4.2 Cloud simulation platform

#### 4.2.1 SOSM Broker integration and decision logging (Logging)
- Discrete-event simulation and structured logging of allocation decisions.

### 4.3 RL submodule (Python)

#### 4.3.1 State Encoder (evolution v1, v2, v3)
- Encoder evolution with additional scarcity/capacity representations.

#### 4.3.2 Policy Network and normalization (BatchNorm, Dropout)
- Feed-forward policy/value network with regularization.

#### 4.3.3 Gym environment (CloudProvisioningEnv)
- Environment wrapper exposing simulator interactions in Gym-style API.

### 4.4 REST API interface (FastAPI)

#### 4.4.1 Endpoints for inference, training, and model management
- Predict, training management, and model persistence endpoints.

### 4.5 Baseline algorithms (Baselines)

#### 4.5.1 Scoring Allocator, Random Allocator, and Heuristics
- Multi-objective scoring baseline, random baseline, and classic heuristic methods.

---

## 5 Numerical results

### 5.1 Experimental setup

#### 5.1.1 Environment presets
- Capacity presets used for training/evaluation (e.g., small/medium/large/high-load/stress-test).

#### 5.1.2 Training settings
- Timesteps, hyperparameters, number of environments, GPU configuration.

### 5.2 Performance metrics

#### 5.2.1 Acceptance/Rejection Rates
- Acceptance rate and rejection breakdown (policy vs capacity).

#### 5.2.2 Energy efficiency per task
- Energy per accepted task (e.g., kWh/task) and per-hardware-type analysis.

#### 5.2.3 SLA compliance
- Percentage of tasks meeting deadlines; latency/deadline violation analysis.

### 5.3 Model performance

#### 5.3.1 Comparison of versions (Capacity Features impact)

### 5.4 Comparative analysis

#### 5.4.1 PPO vs Scoring Allocator vs Heuristics
- Comparative results across presets and workload regimes.

### 5.5 Resource utilization analysis

#### 5.5.1 CPU/GPU/Memory utilization patterns and rejection analysis
- Utilization heatmaps and rejection causes per hardware type/preset.

### 5.6 Statistical analysis

#### 5.6.1 Significance tests (Welch’s t-test) and effect size
- Statistical validation of improvements (p-values, confidence intervals, Cohen’s d).

### 5.7 State vector diagnostics

#### 5.7.1 “Scale Blindness” problem analysis and resolution
- Diagnostic experiments showing failure mode and the role of capacity features.

---

## 6 Conclusions

### 6.1 Research outcomes and conclusions

#### 6.1.1 Assessment of PPO and Domain Randomization
- Summary of policy performance and generalization findings.

#### 6.1.2 Importance of Action Masking and Capacity Features
- Contribution of feasibility masking and scale-aware representation.

### 6.2 Research limitations

#### 6.2.1 Limitations of simulation and workload distributions
- Limits stemming from simulation fidelity, synthetic workload assumptions, and hardware abstraction.

### 6.3 Future work

#### 6.3.1 Multi-agent RL, Offline RL, and transition to real environments
- Extension to multi-cell coordination, learning from historical logs, and real deployment validation.

---

## References
### Books
### Articles
### Technical reports
### Electronic references

---

## Appendix A - Technical Details

### A.1 Hyperparameter settings (Hyperparameters)
| Parameter | Value |
|---|---|
| Learning Rate | 1e-4 – 3e-4 |
| Gamma | 0.99 |
| GAE Lambda | 0.95 |
| Clip Range | 0.2 |
| PPO Epochs | 10 |
| Batch Size | 64 |

### A.2 Log file format (CSV format)
```csv
num_vms,cpu_req,mem_req,util_cpu_before,util_mem_before,
avail_cpu_before,avail_mem_before,avail_storage_before,
avail_accelerators_before,total_cpu,total_mem,total_storage,
total_accelerators,avail_network,total_network,util_network,
cpu_idle_power,cpu_max_power,acc_idle_power,acc_max_power,
compute_cap_per_cpu,compute_cap_acc,energy_kwh,chosen_hw_type,accepted
```

### A.3 API endpoints documentation
| Endpoint | Method | Description |
|---|---|---|
| /rl/health | GET | Service health check |
| /rl/predict | POST | Predict action for state |
| /rl/model | GET | Model info |
| /rl/model/save | POST | Save model |
| /rl/model/load | POST | Load model |
| /rl/experience | POST | Submit experience |
| /rl/training/status | GET | Training status |
| /rl/training/start | POST | Start training |

### A.4 Neural network architecture details
```text
Policy Network (Actor-Critic):

Input (state_dim)
    → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(128) → BatchNorm → ReLU → Dropout(0.3)
    → Linear(64) → ReLU
    ├── Actor Head: Linear(num_actions) → Softmax → π(a|s)
    └── Critic Head: Linear(1) → V(s)
```

### A.5 Training commands and scripts
```bash
# Single GPU
python scripts/train_rl_distributed.py --timesteps 100000 --num-envs 8

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 scripts/train_rl_distributed.py --timesteps 2000000

# With domain randomization and curriculum
torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py   --timesteps 500000 --output-dir results/academic_v8   --use-capacity-features --domain-preset mixed_capacity --curriculum --lr 1e-4
```
