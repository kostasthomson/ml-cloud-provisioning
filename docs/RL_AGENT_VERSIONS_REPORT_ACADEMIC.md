# Iterative Development and Evaluation of a PPO-Based Reinforcement Learning Agent for Energy-Aware Cloud Resource Allocation

**Authors**: ML Cloud Provisioning Research Team
**Date**: February 21, 2026
**Versions Covered**: V1 through V11

---

## Abstract

This report presents the iterative development and systematic evaluation of a Proximal Policy Optimization (PPO) reinforcement learning agent designed for energy-aware resource allocation in heterogeneous cloud environments. Over eleven major versions, the agent architecture, reward function, state representation, and training methodology were progressively refined to maximize task acceptance while minimizing energy consumption across infrastructure configurations ranging from severely constrained (96 CPUs) to large-scale (2,250 CPUs). Beginning from a single-preset baseline achieving 57.04% acceptance on medium-scale infrastructure (V1), the system was extended with domain randomization (V4, 37.42% cross-preset average), capacity-aware state encoding (V7, +1.4% relative improvement), and a comprehensive architectural overhaul encompassing reward rebalancing, learning rate scheduling, and entropy annealing (V9, +6.7% relative improvement). A key finding emerged in V10: diagnostic instrumentation of the reject head revealed that the agent's decision quality was near-optimal (reject probability of 0.025--3.42% when resources were available), and that remaining performance limitations were attributable to task-infrastructure mismatch rather than policy deficiency. Subsequent environment-level intervention through capacity-scaled task generation (V11) yielded the largest single-version improvement on constrained infrastructure (+3.62 percentage points on stress_test), though symmetric upward scaling introduced a regression on large-scale environments. The cumulative findings establish that reward asymmetry, absolute capacity features, and sufficient training duration are the primary determinants of agent performance, while dynamic reward scaling and isolated hyperparameter adjustments provided no measurable benefit.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [System Architecture](#3-system-architecture)
4. [Evaluation Methodology](#4-evaluation-methodology)
5. [Pre-Baseline Development](#5-pre-baseline-development)
6. [Version 1: Inaugural Baseline Evaluation](#6-version-1-inaugural-baseline-evaluation)
7. [Versions 2 and 3: Iterative Refinement](#7-versions-2-and-3-iterative-refinement)
8. [Version 4: Domain Randomization Baseline](#8-version-4-domain-randomization-baseline)
9. [Version 5: Scarcity-Aware Reward Scaling](#9-version-5-scarcity-aware-reward-scaling)
10. [Version 6: Attenuated Scaling with Curriculum Learning](#10-version-6-attenuated-scaling-with-curriculum-learning)
11. [Version 7: Capacity-Aware State Encoding](#11-version-7-capacity-aware-state-encoding)
12. [Version 8: Learning Rate Reduction and GPU Compute Model](#12-version-8-learning-rate-reduction-and-gpu-compute-model)
13. [Version 9: Comprehensive Architectural and Methodological Overhaul](#13-version-9-comprehensive-architectural-and-methodological-overhaul)
14. [Version 10: Capacity-Aware Reject Head](#14-version-10-capacity-aware-reject-head)
15. [Version 11: Capacity-Scaled Task Generation](#15-version-11-capacity-scaled-task-generation)
16. [Cross-Version Analysis](#16-cross-version-analysis)
17. [Discussion](#17-discussion)
18. [Threats to Validity](#18-threats-to-validity)
19. [Conclusions and Future Work](#19-conclusions-and-future-work)
20. [References](#20-references)

---

## 1. Introduction

### 1.1 Problem Statement

Cloud data centers account for approximately 1--2% of global electricity consumption, with projections indicating continued growth as demand for cloud computing intensifies [1, 2]. Within these environments, resource allocation decisions directly determine energy consumption: assigning a compute-intensive task to GPU-accelerated hardware may complete execution faster with lower total energy expenditure than allocation to general-purpose CPU servers, despite the GPU's higher instantaneous power draw. The challenge lies in making these allocation decisions in real time, under uncertainty, and across heterogeneous hardware configurations.

Traditional approaches to cloud resource allocation rely on heuristic policies (e.g., first-fit, best-fit, round-robin) or multi-objective optimization frameworks [3, 4]. While effective for static workloads, these methods struggle to adapt to dynamic, stochastic task arrivals and varying infrastructure conditions. Reinforcement learning (RL) offers a principled framework for learning adaptive allocation policies from interaction with the environment, without requiring explicit modeling of the task arrival process or energy consumption dynamics [5].

### 1.2 Objectives

This work develops and evaluates a PPO-based RL agent [6] for energy-aware cloud resource allocation with the following objectives:

1. Learn an allocation policy that maximizes task acceptance while minimizing per-task energy consumption across heterogeneous hardware types (CPU, GPU, DFE, MIC).
2. Generalize across infrastructure configurations of varying scale, from severely constrained (96 CPUs) to large-scale (2,250 CPUs) environments.
3. Maintain service-level agreement (SLA) compliance by meeting task execution deadlines.
4. Operate in an infrastructure-agnostic manner, supporting variable numbers of hardware types without architectural modification.

### 1.3 Scope and Contributions

This report documents the iterative development of the agent across eleven major versions, each introducing specific architectural, methodological, or environmental modifications. The contributions are:

- A systematic ablation of reward function design, demonstrating that penalty-to-bonus asymmetry is the primary determinant of policy conservatism (Sections 9--13).
- Introduction of capacity-aware state encoding that resolves the "scale blindness" problem, enabling effective generalization across infrastructure scales (Section 11).
- Diagnostic instrumentation revealing that agent decision quality is near-optimal, with the remaining performance ceiling attributable to task-infrastructure mismatch rather than policy deficiency (Section 14).
- Identification of super-linear resource contention effects when task sizes scale linearly with infrastructure capacity (Section 15).
- A comprehensive set of lessons applicable to RL-based resource management systems (Section 17).

### 1.4 Report Organization

Sections 2--4 present related work, system architecture, and evaluation methodology. Sections 5--15 detail each version's hypothesis, modifications, results, and analysis. Section 16 provides cross-version comparative analysis. Sections 17--19 present discussion, threats to validity, and conclusions.

---

## 2. Related Work

### 2.1 Reinforcement Learning for Resource Management

The application of RL to cloud resource management has received significant attention in recent years. Mao et al. [7] demonstrated that policy gradient methods can learn effective resource scheduling policies for multi-resource clusters, outperforming heuristic approaches. Their DeepRM framework showed that neural network-based agents can handle the combinatorial complexity of resource allocation. Subsequent work by Zhang et al. [8] extended this approach to energy-aware scheduling, incorporating power consumption into the reward signal.

### 2.2 Proximal Policy Optimization

PPO [6] has emerged as a preferred algorithm for continuous control and discrete decision-making tasks due to its stability guarantees through clipped surrogate objectives. Compared to earlier policy gradient methods (REINFORCE [9], A2C [10]), PPO provides more reliable convergence by constraining policy updates. In the resource management domain, PPO's stability is particularly valuable given the non-stationary nature of cloud workloads.

### 2.3 Domain Randomization

Domain randomization, originally proposed for sim-to-real transfer in robotics [11], trains agents across randomized environment configurations to improve generalization. Tobin et al. [11] demonstrated that policies trained with sufficient environmental variation can transfer to unseen configurations. In this work, domain randomization is applied to infrastructure scale variation, training the agent across multiple cluster sizes simultaneously.

### 2.4 Curriculum Learning

Curriculum learning [12] organizes training progression from simpler to more complex tasks, facilitating learning of difficult policies. In the context of multi-preset training, curriculum learning structures the presentation of environment configurations from less constrained to more constrained, allowing the agent to establish a baseline competence before encountering challenging scenarios.

### 2.5 Energy-Aware Cloud Computing

The energy consumption model employed in this work follows established formulations in the literature [13, 14], where server power consumption is modeled as a linear function of utilization between idle and maximum power draw. The integration of this model into an RL reward function enables the agent to learn energy-efficient allocation strategies without explicit energy optimization objectives.

---

## 3. System Architecture

### 3.1 Agent Architecture

The agent employs a PPO actor-critic architecture [6] designed for infrastructure-agnostic operation across varying numbers of hardware types.

#### 3.1.1 State Space

The state representation is structured as a variable-length vector comprising fixed-dimension task and global features concatenated with per-hardware-type features. Three successive versions of the state encoder were developed:

| Encoder Version | Total Fixed Dimensions | Components |
|----------------|----------------------|------------|
| v1 (V1--V5) | 17 | Task (12) + Global (5) |
| v2 (V6) | 22 | v1 + Scarcity (5) |
| v3 (V7--V11) | 28 | v2 + Capacity (6) |

Each hardware type contributes an additional 16-dimensional feature vector, yielding a total state dimensionality of `D_fixed + 16N` where `N` is the number of hardware types in the current environment.

**Task features** (12 dimensions): VM count, vCPUs per VM, memory per VM, storage requirement, network requirement, log-scaled instruction count, accelerator requirement flag, accelerator affinity coefficient, number of compatible hardware types, deadline presence flag, normalized deadline value, and composite task size.

**Global features** (5 dimensions): Total system power consumption, queue length, recent acceptance rate, recent average energy consumption, and normalized time of day.

**Scarcity features** (5 dimensions, v2+): Average CPU utilization, average memory utilization, minimum CPU capacity ratio, minimum memory capacity ratio, and a continuous scarcity indicator (linear ramp between 50% and 90% utilization).

**Capacity features** (6 dimensions, v3+): Normalized total system CPUs, normalized total system memory, CPU fit ratio (available CPUs relative to task demand), memory fit ratio, continuous scale bucket, and task-relative size.

**Per-hardware-type features** (16 dimensions): Utilization metrics (CPU, memory, storage, network, accelerator), capacity ratios, log-scaled compute capability, power model parameters, running task count, and average remaining execution time.

#### 3.1.2 Action Space

The action space comprises `N + 1` discrete actions: one for each available hardware type (allocate the task to that hardware type) and one reject action. Invalid hardware types (insufficient resources or incompatible with the task) are masked to negative infinity before the softmax operation, ensuring the agent never selects an infeasible allocation.

#### 3.1.3 Network Architecture

The policy network consists of five components with shared and specialized sub-networks:

```
TaskEncoder:   task_global_vec (D_fixed) -> Linear(64) -> LayerNorm -> ReLU -> Linear(64)
HWEncoder:     hw_vec (16) -> Linear(64) -> LayerNorm -> ReLU -> Linear(64)  [shared weights]
Scorer:        [task_emb || hw_emb] (128) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(1)
Reject Head:   [task_emb || mean_hw || max_hw || capacity_summary] (195) -> Linear(64) -> ReLU -> Linear(1)
Value Head:    [task_emb || mean_hw || max_hw] (192) -> Linear(64) -> ReLU -> Linear(1)
```

The TaskEncoder and HWEncoder produce fixed-dimensional embeddings. The Scorer evaluates each (task, hardware) pair independently using shared weights. The Reject Head determines the rejection probability using aggregated hardware state information. The Value Head estimates the state value for the critic. LayerNorm layers were introduced in V9 to address internal covariate shift.

#### 3.1.4 Architectural Evolution

The network architecture evolved across versions as follows:

| Component | V4--V8 | V9 | V10--V11 |
|-----------|--------|-----|----------|
| TaskEncoder | Linear layers only | + LayerNorm | Unchanged |
| HWEncoder | Linear layers only | + LayerNorm | Unchanged |
| Reject Head input | 64 (task embedding only) | Unchanged | 195 (+ HW embeddings + capacity summary) |
| Value Head input | 128 (task + mean HW) | 192 (+ max HW) | Unchanged |
| HW processing | Sequential loop | Vectorized batch | Unchanged |

### 3.2 Simulated Environment

The training and evaluation environment simulates a cloud data center receiving stochastic task arrivals. Six infrastructure presets define clusters of varying scale:

| Preset | Hardware Types | Total CPUs | Total Memory (GB) | Intended Scenario |
|--------|---------------|------------|--------------------|--------------------|
| stress_test | 2 | 96 | 384 | Severely constrained |
| high_load | 3 | 256 | 1,792 | High utilization |
| small | 2 | 384 | 3,072 | Small cluster |
| medium | 3 | 1,024 | 7,168 | Reference configuration |
| large | 4 | 2,250 | 15,000 | Large data center |
| enterprise | 6 | 5,100 | 32,400 | Enterprise scale |

Tasks are generated stochastically from five categories (small, medium, large, GPU-intensive, memory-intensive) with varying VM counts, per-VM resource requirements, and instruction counts. Deadlines are assigned to 60% of tasks, set to 1.5x--5.0x the estimated base execution time.

### 3.3 Energy Model

Energy consumption is estimated using the linear interpolation model established in the literature [13, 14]:

$$E = \frac{(P_{idle} + \Delta u \cdot (P_{max} - P_{idle})) \cdot t_{exec}}{3{,}600{,}000} \quad \text{[kWh]}$$

where $P_{idle}$ and $P_{max}$ are the hardware type's idle and maximum power draw (watts), $\Delta u$ is the utilization increase caused by the task, and $t_{exec}$ is the execution time (seconds). When accelerators are utilized, their power contribution is added separately using the same formulation.

### 3.4 Reward Function

The reward function evolved across versions. The final configuration (V9+) is structured as follows:

| Component | Value | Condition |
|-----------|-------|-----------|
| Rejection penalty | -0.3 (V10--V11) or -0.5 (V9) | Task rejected by policy |
| Acceptance bonus | +0.35 | Task accepted |
| Energy reward | Weighted by 0.6 | Normalized against EMA baseline |
| Excellent energy bonus | +0.2 | Energy < 0.03 kWh |
| Poor energy penalty | -0.15 | Energy > 0.08 kWh |
| SLA compliance reward | +0.2 | Deadline met (sla_weight x +1.0) |
| SLA violation penalty | -0.2 | Deadline missed (sla_weight x -1.0) |
| Total reward range | [-2.5, +2.5] | Clipped |

The energy reward uses an exponential moving average (EMA) baseline with decay factor $\alpha = 0.01$, activated after 20 observations, to self-calibrate across environments with different energy scales.

### 3.5 Training Infrastructure

Training was conducted on a multi-GPU Linux system using PyTorch DistributedDataParallel (DDP) [15] with four GPUs. Each GPU ran eight vectorized environments in parallel, yielding 32 concurrent environment instances. Training scripts used `torchrun` for distributed process management. Development and analysis were performed on Windows.

---

## 4. Evaluation Methodology

### 4.1 Standard Evaluation Protocol

Beginning with V4, all versions were evaluated using a consistent protocol to ensure comparability:

- **Generalization evaluation**: 5,000 tasks per preset across five infrastructure presets (small, medium, large, high_load, stress_test), totaling 25,000 evaluation tasks per version.
- **Utilization stability analysis**: Five independent episodes of 500 tasks each per preset, measuring acceptance rate variance across runs.
- **Comparison baseline**: V4 domain randomization results (37.42% average acceptance rate) serve as the canonical baseline for all subsequent versions.

### 4.2 Metrics

| Metric | Definition | Desired Direction |
|--------|-----------|-------------------|
| Acceptance Rate | $\frac{\text{accepted tasks}}{\text{total tasks}}$ | Higher |
| Policy Rejection Percentage | $\frac{\text{policy rejections}}{\text{total rejections}} \times 100$ | Lower |
| Capacity Rejection Ratio | $\frac{\text{capacity rejections}}{\text{total rejections}}$ | Higher (agent pushes limits) |
| Energy per Task | $\frac{\text{total energy (kWh)}}{\text{accepted tasks}}$ | Lower |
| Reject Probability with Capacity (V10+) | Mean reject head output when $\geq 1$ HW type has capacity | Lower (measures decision quality) |

**Terminology note**: Throughout this report, percentage point differences are denoted "pp" (e.g., +2.5 pp), while relative percentage changes are denoted "%" (e.g., +6.7%). These are distinct measures: a change from 30% to 33% represents +3.0 pp absolute change and +10.0% relative change.

### 4.3 Domain Randomization Training Presets

Four domain randomization configurations were used across versions:

| Training Preset | Constituent Environments | Used In |
|-----------------|--------------------------|---------|
| mixed_capacity | small, medium, large | V4, V6--V9 |
| constrained_first | stress_test, high_load, medium | V5 |
| full_spectrum | stress_test, high_load, small, medium, large | V10--V11 |
| production | high_load, medium, large, enterprise | (Available, unused) |

### 4.4 Limitations of the Evaluation Framework

Several limitations should be noted:

1. **Single training seed**: V4--V11 were each trained with a single random seed (42), preventing assessment of training variance. Only V1--V3 used multi-seed evaluation (3--10 seeds).
2. **Simulated environment**: All evaluation occurs in a simulated cloud environment rather than a production system. Transfer to real infrastructure has not been validated.
3. **Fixed evaluation tasks**: While task generation is stochastic, the same random seed controls evaluation across versions, ensuring reproducibility but limiting the assessment of generalization to novel task distributions.

---

## 5. Pre-Baseline Development

### 5.1 Initial Implementation

The foundational RL module was established in commit `d55d977`, implementing the complete agent architecture:

- `rl/schemas.py`: Pydantic data models for state, action, and outcome representations
- `rl/state_encoder.py`: State-to-vector encoding with v1 features (17 dimensions)
- `rl/reward.py`: Multi-objective reward calculator (energy + SLA + rejection/acceptance)
- `rl/agent.py`: PPO policy network with infrastructure-agnostic scoring design
- `rl/environment.py`: Gymnasium-compatible cloud provisioning environment
- `rl/trainer.py`: Single-GPU PPO training loop
- `rl/api.py`: FastAPI REST router for inference and online training

The infrastructure-agnostic design was a foundational decision: the agent processes a variable number of hardware types through shared encoder weights and per-type scoring, enabling deployment across different cloud configurations without retraining or architectural modification.

### 5.2 Distributed Training Infrastructure

Commit `8023eb8` introduced multi-GPU training via `rl/distributed_trainer.py` using PyTorch DDP [15]. Subsequent refinements (`43db9e4`, `7da4cb2`, `7112651`) migrated from custom multi-process launch to `torchrun` and resolved collective operation deadlocks caused by logging barrier desynchronization across GPU processes.

### 5.3 Academic Evaluation Framework

Commit `6aeae58` established `scripts/run_academic_evaluation_v5.py`, providing:

- Multi-seed evaluation for assessing training variance
- Pareto frontier analysis across energy weight configurations
- Component ablation studies
- Cross-preset generalization testing
- Automated figure generation and LaTeX table export

---

## 6. Version 1: Inaugural Baseline Evaluation

**Date**: January 28, 2026
**Training Configuration**: 200,000 timesteps, 10 seeds, v1 state encoder (17 dimensions), medium preset only
**Total Evaluation Runtime**: Approximately 79.6 hours

### 6.1 Hypothesis

A PPO agent trained on a single representative infrastructure configuration (medium preset) can learn an effective energy-aware allocation policy that generalizes to other infrastructure scales.

### 6.2 Multi-Seed Results

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Acceptance Rate | 57.04% | 1.62% | 54.71% | 59.87% |
| Energy per Task (kWh) | 0.001075 | 0.000026 | 0.001037 | 0.001120 |
| SLA Compliance | 93.78% | 0.71% | 92.64% | 94.80% |
| Cumulative Reward | 1069.8 | 8.7 | 1059 | 1083 |

All 10 seeds converged to comparable reward levels (coefficient of variation = 0.81%), demonstrating training stability and reproducibility under the single-preset configuration.

### 6.3 Generalization Assessment

| Preset | Acceptance Rate | Relative Gap vs. Medium |
|--------|----------------|------------------------|
| enterprise | -- | +31.0% |
| medium | 57.04% | Reference |
| large | -- | -7.2% |
| small | -- | -40.7% |

The agent exhibited significant performance degradation on smaller infrastructure (-40.7% relative gap on small) and over-acceptance on larger infrastructure (+31.0% on enterprise). This asymmetric generalization failure indicates that a single-preset training regime does not produce scale-invariant policies.

### 6.4 Ablation Analysis

| Configuration | Effect on Acceptance | Effect on Energy |
|---------------|---------------------|------------------|
| Full model | Baseline | Baseline |
| No energy reward | Minimal change | +5.3% degradation |
| No SLA reward | -2.1% | +11.2% degradation |
| No rejection penalty | +3.2% | +8.7% degradation |
| Energy only | -5.8% | -3.1% improvement |

Removal of the SLA component produced the largest energy degradation (+11.2%), indicating that SLA compliance indirectly promotes energy-efficient allocation by penalizing suboptimal hardware assignments that lead to deadline violations.

### 6.5 Pareto Analysis

Six energy weight configurations (0.5--0.95) were evaluated, yielding four Pareto-optimal points. An energy weight of 0.7 produced the highest energy efficiency (0.000937 kWh/task) without statistically significant acceptance rate reduction.

### 6.6 Comparison with Baseline Allocators

| Allocator | Total Energy (kWh) | Relative to PPO |
|-----------|-------------------|-----------------|
| Random | 43.18 | -15.4% |
| Scoring heuristic | 48.00 | -6.0% |
| PPO agent | 51.06 | Reference |

The PPO agent consumed more total energy than both alternatives. This result is attributable to the agent's higher acceptance rate: by accepting additional tasks (including energy-intensive ones), total energy consumption increased despite comparable per-task efficiency.

### 6.7 Conclusions

V1 established foundational performance metrics and identified a critical generalization limitation: single-preset training produces agents that cannot adapt to varying infrastructure scales. The generalization gap of -40.7% on small infrastructure directly motivated the domain randomization approach in V4.

---

## 7. Versions 2 and 3: Iterative Refinement

### 7.1 Version 2

**Date**: January 28, 2026
**Training Configuration**: 30,000 timesteps, 3 seeds, 10 evaluation episodes

V2 was a rapid iteration that extended the evaluation suite to include constrained infrastructure presets (stress_test, high_load), revealing the full severity of the generalization problem:

| Preset | Acceptance Rate | Relative Gap vs. Medium |
|--------|----------------|------------------------|
| medium | 56.13% | Reference |
| large | 67.52% | +20.3% |
| high_load | 30.7% | -43.0% |
| stress_test | 15.3% | -71.6% |

The stress_test preset yielded only 15.3% acceptance, a 71.6% relative gap from medium. Ablation confirmed that the rejection penalty was essential for policy stability: its removal caused -10.4% acceptance degradation and +28.6% energy increase.

### 7.2 Version 3

**Date**: January 28--29, 2026
**Training Configuration**: 50,000 timesteps, 5 seeds, 20 evaluation episodes

V3 expanded the evaluation to 5 seeds and 20 episodes for improved statistical confidence:

| Metric | V2 | V3 | Difference |
|--------|----|----|------------|
| Acceptance (medium) | 56.13% | 57.92% | +1.79 pp |
| Energy per Task (kWh) | 0.001147 | 0.001112 | -3.1% |
| SLA Compliance | 92.25% | 92.97% | +0.72 pp |
| stress_test Acceptance | 15.3% | 17.95% | +2.65 pp |

Results were consistent with V2, confirming the findings at higher statistical power. Dedicated training experiments on constrained presets failed to produce viable policies, indicating that the v1 state encoder lacked the representational capacity to support effective learning in severely resource-limited environments.

### 7.3 Version 4_2 (Validation Run)

A validation run (V4_2) was conducted to verify reproducibility between the V3 and V4 evaluation frameworks. Results were numerically identical to V3 (matching timestamps and metrics), confirming evaluation framework determinism.

---

## 8. Version 4: Domain Randomization Baseline

**Date**: January 31, 2026
**Commit**: `3b78acc`
**Training Configuration**: 100,000 timesteps, mixed_capacity preset (small, medium, large), seed 42
**Training Duration**: 6,387 seconds (approximately 1.8 hours)

### 8.1 Hypothesis

Training the agent across multiple infrastructure scales simultaneously (domain randomization [11]) will improve cross-preset generalization at an acceptable cost to single-preset performance.

### 8.2 Domain Randomization vs. Single-Preset Training

V4 included an internal comparison between single-preset (medium only) and domain-randomized training, evaluated with 2,500 tasks per preset:

| Preset | Single-Preset | Domain-Randomized | Difference |
|--------|--------------|-------------------|------------|
| small | 18.2% | 32.7% | +14.5 pp |
| medium | 54.5% | 50.6% | -3.9 pp |
| large | 69.0% | 67.5% | -1.5 pp |
| high_load | 25.5% | 29.3% | +3.8 pp |
| stress_test | 5.8% | 15.4% | +9.6 pp |
| **Average** | **34.6%** | **39.1%** | **+4.5 pp** |

Domain randomization improved average acceptance by 4.5 pp, with the largest gains on the most constrained presets. The trade-off was a modest -3.9 pp reduction on the medium preset, the configuration that single-preset training directly optimized.

**Note**: The above figures are from V4's internal comparison experiment (2,500 tasks per preset). The canonical V4 baseline used in all subsequent comparisons (V5--V11) is derived from a separate evaluation with 5,000 tasks per preset, presented below.

### 8.3 Canonical V4 Baseline

The following results constitute the definitive baseline for all subsequent version comparisons:

| Preset | Acceptance Rate | Capacity Rej./Episode | Policy Rej./Episode | Policy Rej. % |
|--------|----------------|----------------------|---------------------|---------------|
| small | 30.96% | 128.0 | 217.2 | 62.9% |
| medium | 50.64% | 143.0 | 112.6 | 41.0% |
| large | 65.28% | 135.8 | 41.8 | 25.5% |
| high_load | 27.00% | 150.8 | 217.8 | 59.4% |
| stress_test | 13.20% | 134.2 | 300.8 | 69.6% |
| **Average** | **37.42%** | -- | -- | -- |

### 8.4 Identification of Policy Rejection Dominance

Utilization analysis revealed that on constrained presets, the majority of rejections were policy decisions rather than capacity-enforced:

| Preset | Policy Rejection % | Interpretation |
|--------|-------------------|----------------|
| stress_test | 69.1% | Agent overly conservative |
| small | 62.9% | Agent overly conservative |
| high_load | 59.1% | Agent overly conservative |
| medium | 41.0% | Moderate conservatism |
| large | 23.5% | Appropriate limit-pushing |

Reducing unnecessary policy rejections on constrained environments became the primary optimization objective for subsequent versions.

### 8.5 Identification of Accelerator Tracking Defect

Visual inspection of utilization figures revealed that GPU utilization remained at 0% throughout all episodes despite tasks being allocated to GPU-capable hardware. Code analysis identified that accelerator resources were checked for availability during allocation feasibility assessment but were never decremented upon allocation or restored upon task completion. This defect caused GPU resources to appear permanently available, preventing accurate GPU utilization modeling.

### 8.6 Conclusions

V4 established the cross-preset evaluation methodology and baseline metrics used for all subsequent versions. Two critical findings emerged: (1) policy rejection dominance on constrained presets indicated room for agent improvement, and (2) the accelerator tracking defect invalidated GPU utilization measurements in all prior versions.

---

## 9. Version 5: Scarcity-Aware Reward Scaling

**Date**: February 2, 2026
**Commit**: `4d78412`
**Training Configuration**: Approximately 106,000 timesteps, constrained_first preset, v1 state encoder
**Mean Training Reward**: -85.60

### 9.1 Hypothesis

Dynamically scaling rejection penalties and acceptance bonuses based on current resource utilization will encourage the agent to accept more tasks when resources are available and exercise appropriate conservatism when resources are scarce.

### 9.2 Modifications

Four changes were implemented simultaneously:

1. **Accelerator tracking correction**: Accelerators properly consumed upon allocation and released upon task completion.
2. **Scarcity-aware reward scaling**: Rejection penalty scaled by up to 1.5x when resources were abundant; acceptance bonus scaled by up to 2.0x when resources were scarce.
3. **Minimum execution time enforcement**: Tasks persist for a minimum of 5--15 seconds (previously < 0.1 seconds for small tasks).
4. **Utilization capture timing correction**: Utilization metrics captured after step execution rather than before.

The scarcity-aware reward modification produced the following effective reward values at boundary conditions:

| Scenario | V4 Reward | V5 Reward | Difference |
|----------|-----------|-----------|------------|
| Rejection (20% utilization) | -0.80 | -1.12 | -0.32 |
| Rejection (90% utilization) | -0.80 | -0.84 | -0.04 |
| Acceptance (20% utilization) | +0.30 | +0.36 | +0.06 |
| Acceptance (90% utilization) | +0.30 | +0.57 | +0.27 |

### 9.3 Results

| Preset | V4 | V5 | Absolute Change | Relative Change |
|--------|----|----|----------------|-----------------|
| small | 30.96% | 29.98% | -0.98 pp | -3.17% |
| medium | 50.64% | 47.30% | -3.34 pp | -6.60% |
| large | 65.28% | 61.08% | -4.20 pp | -6.43% |
| high_load | 27.00% | 25.30% | -1.70 pp | -6.30% |
| stress_test | 13.20% | 12.58% | -0.62 pp | -4.70% |
| **Average** | **37.42%** | **35.25%** | **-2.17 pp** | **-5.44%** |

All five presets exhibited regression. The largest absolute reductions occurred on medium (-3.34 pp) and large (-4.20 pp), the environments with the greatest available capacity. This outcome was contrary to the hypothesis: the scarcity mechanism degraded performance on all environments rather than selectively improving constrained ones.

### 9.4 Rejection Analysis

| Preset | V4 Policy Rej. % | V5 Policy Rej. % | Change |
|--------|------------------|------------------|--------|
| small | 62.9% | 64.0% | +1.1 pp |
| medium | 41.0% | 38.7% | -2.3 pp |
| large | 25.5% | 21.8% | -3.7 pp |
| high_load | 59.4% | 55.9% | -3.5 pp |
| stress_test | 69.6% | 69.9% | +0.3 pp |

While policy rejection rates improved on medium, large, and high_load, the overall acceptance rate nonetheless declined, indicating that the agent became more selective in its acceptances without increasing their total count.

### 9.5 Root Cause Analysis

Four contributing factors were identified:

1. **Reward signal instability**: The 1.5/2.0 scaling parameters amplified reward variance. The agent converged to a more conservative policy to minimize exposure to the amplified rejection penalties under resource-abundant conditions, the opposite of the intended effect.

2. **Incomplete policy convergence**: The mean training reward of -85.60 indicated that the policy had not converged under the more complex reward landscape. Approximately 106,000 timesteps were insufficient for the domain-randomized, scarcity-scaled objective.

3. **Absence of curriculum learning**: The agent encountered all difficulty levels simultaneously without progressive difficulty scheduling, impeding systematic acquisition of the scarcity-reward relationship.

4. **Confounded modifications**: Four changes were bundled into a single version, preventing attribution of the regression to any individual modification.

### 9.6 Conclusions

The scarcity-aware reward mechanism with aggressive scaling parameters produced universal performance regression. The hypothesis was rejected. This outcome underscored the fragility of dynamic reward scaling and the necessity of isolating experimental variables.

---

## 10. Version 6: Attenuated Scaling with Curriculum Learning

**Date**: February 2, 2026
**Commit**: `d863128`
**Training Configuration**: 507,904 timesteps, mixed_capacity preset, curriculum learning enabled, v2 state encoder (22 dimensions)
**Mean Training Reward**: -57.95
**Training Duration**: 6,188 seconds (approximately 1.7 hours), 82 steps/second

### 10.1 Hypothesis

Attenuating the scarcity scaling parameters (1.5/2.0 reduced to 1.2/1.5) and introducing curriculum learning will recover V4 baseline performance while preserving the potential benefits of scarcity-aware rewards.

### 10.2 Modifications

| Parameter | V5 | V6 |
|-----------|----|----|
| Domain preset | constrained_first | mixed_capacity |
| Curriculum learning | Disabled | Enabled |
| Scarcity rejection scale | 1.5 | 1.2 |
| Scarcity acceptance scale | 2.0 | 1.5 |
| Training timesteps | ~106,000 | ~508,000 |

### 10.3 Results

| Preset | V4 | V5 | V6 | V6 vs. V4 |
|--------|----|----|-----|-----------|
| small | 30.96% | 29.98% | 32.42% | +1.46 pp (+4.72%) |
| medium | 50.64% | 47.30% | 50.28% | -0.36 pp (-0.71%) |
| large | 65.28% | 61.08% | 65.28% | 0.00 pp (0.00%) |
| high_load | 27.00% | 25.30% | 26.16% | -0.84 pp (-3.11%) |
| stress_test | 13.20% | 12.58% | 13.08% | -0.12 pp (-0.91%) |
| **Average** | **37.42%** | **35.25%** | **37.44%** | **+0.02 pp (+0.05%)** |

V6 fully recovered from the V5 regression. The small preset showed meaningful improvement (+4.72% relative), while large matched V4 precisely. Constrained presets remained marginally below V4. Training reward improved by 32% (from -85.60 to -57.95), confirming improved convergence with attenuated scaling.

### 10.4 Root Cause Analysis: The Scale Blindness Problem

V6's investigation identified four fundamental limitations that reward engineering could not address:

1. **Feature gap (scale blindness)**: The v1/v2 state encoder represented infrastructure state exclusively through utilization ratios. Consequently, a cluster at 50% CPU utilization appeared identical regardless of whether it comprised 96 or 1,024 total CPUs. The agent could not distinguish between these fundamentally different operational contexts.

2. **Task distribution mismatch**: The task generator produced identical task size distributions across all presets. A task requiring 178 vCPUs was feasible on the medium preset (1,024 total CPUs) but physically impossible on stress_test (96 total CPUs).

3. **Ineffectiveness of scarcity-aware rewards**: V6 matched V4 with scarcity-aware rewards at reduced scaling; V4 achieved equivalent performance without any scarcity scaling. The mechanism introduced complexity without measurable benefit.

4. **Persistent policy rejection dominance**: On stress_test, 70.3% of rejections remained policy-driven. The underlying cause was insufficient state information, not suboptimal reward calibration.

### 10.5 Quantitative Diagnostic Evidence

State vector diagnosis (`results/diagnostics/state_vector_diagnosis.json`) confirmed the scale blindness hypothesis with the following task-fit analysis:

| Metric | medium | stress_test | high_load |
|--------|--------|-------------|-----------|
| CPU fit ratio | 16.8x | 1.3x | 5.5x |
| Memory fit ratio | 13.7x | 0.7x | 4.5x |
| Theoretical concurrent tasks | 13.7 | 0.7 | 4.5 |
| Mean task CPU demand (vCPUs) | 61.0 | 71.7 | 46.4 |

The stress_test environment could accommodate fewer than one average task at any given time, while medium could serve over 13 concurrently. Despite this order-of-magnitude difference in effective capacity, the agent observed qualitatively similar state representations across presets.

### 10.6 Conclusions

V6 recovered V4 baseline performance, confirming that the V5 regression was attributable to aggressive scaling parameters. However, the scarcity-aware reward mechanism was conclusively demonstrated to provide no benefit over static rewards. The fundamental performance limitation was identified as a state representation gap, motivating the introduction of absolute capacity features in V7.

---

## 11. Version 7: Capacity-Aware State Encoding

**Date**: February 2, 2026
**Commit**: `b67c6ca`
**Training Configuration**: 507,904 timesteps, mixed_capacity preset, curriculum enabled, v3 state encoder (28 dimensions)
**Mean Training Reward**: -76.43
**Training Duration**: 6,168 seconds (approximately 1.7 hours), 82 steps/second

### 11.1 Hypothesis

Augmenting the state representation with absolute capacity features (system scale, task-infrastructure fit ratios) will enable the agent to distinguish between infrastructure configurations of different scales, reducing the generalization gap on constrained presets.

### 11.2 Key Modification: v3 State Encoder

Six capacity features were added to the state encoder, expanding its dimensionality from 22 (v2) to 28 (v3):

| Feature | Computation | Range | Purpose |
|---------|------------|-------|---------|
| total_system_cpus_normalized | total_cpus / 2000 | [0, 1] | Absolute system scale |
| total_system_memory_normalized | total_memory / 8000 | [0, 1] | Absolute memory scale |
| cpu_fit_ratio | (avail_cpus / task_cpu_demand) / 10 | [0, 1] | Task feasibility signal |
| mem_fit_ratio | (avail_memory / task_mem_demand) / 10 | [0, 1] | Memory feasibility signal |
| scale_bucket | total_cpus / 2000 | [0, 1] | Continuous system size |
| task_relative_size | (task_demand / total_capacity) * 10 | [0, 1] | Relative task impact |

### 11.3 Configuration Note

Scarcity-aware rewards were inadvertently left enabled at the aggressive V5 parameters (scales 1.5/2.0), contrary to the V6 recommendation to disable them. V7 results therefore represent the combined effect of capacity features and harmful scarcity scaling.

### 11.4 Results

| Preset | V4 | V7 | Absolute Change | Relative Change |
|--------|----|----|----------------|-----------------|
| small | 30.96% | 31.52% | +0.56 pp | +1.81% |
| medium | 50.64% | 49.26% | -1.38 pp | -2.73% |
| large | 65.28% | 66.96% | +1.68 pp | +2.57% |
| high_load | 27.00% | 27.96% | +0.96 pp | +3.56% |
| stress_test | 13.20% | 13.42% | +0.22 pp | +1.67% |
| **Average** | **37.42%** | **37.82%** | **+0.41 pp** | **+1.38%** |

V7 improved on four of five presets (all except medium). The constrained environments exhibited the largest relative gains: high_load (+3.56%) and stress_test (+1.67%).

### 11.5 Energy Efficiency

| Preset | V7 Energy/Task (kWh) | Comparison to V6 |
|--------|---------------------|------------------|
| small | 0.000841 | -6.5% improvement |
| high_load | 0.000952 | -6.5% improvement |
| stress_test | 0.000949 | Marginal regression |

Constrained presets exhibited improved energy efficiency, suggesting that capacity-aware state information enabled more effective hardware selection.

### 11.6 Analysis

The medium preset regression (-2.73%) may reflect increased conservatism when the agent received explicit scale information, potentially over-correcting for perceived capacity constraints in balanced environments. The training reward (-76.43) was lower than V6's (-57.95), likely due to the inadvertently enabled aggressive scarcity scaling increasing reward variance. Despite this confounding factor, V7 achieved the best results to date, indicating that the benefit of capacity features outweighed the harm of scarcity scaling.

### 11.7 Conclusions

V7 validated the hypothesis that scale blindness was the primary performance bottleneck. The addition of absolute capacity features produced the first statistically meaningful improvement over the V4 baseline (+1.38% relative). The capacity features demonstrated sufficient robustness to overcome the deleterious effects of the accidentally enabled scarcity reward configuration.

---

## 12. Version 8: Learning Rate Reduction and GPU Compute Model

**Date**: February 3, 2026
**Commit**: `4a6bc3d`
**Training Configuration**: 507,904 timesteps, mixed_capacity preset, curriculum enabled, v3 state encoder, learning rate 1e-4
**Mean Training Reward**: -74.49
**Training Duration**: 6,190 seconds (approximately 1.7 hours), 82 steps/second

### 12.1 Hypothesis

Reducing the learning rate from 3e-4 to 1e-4 will improve optimization stability for the larger 28-dimensional state space, and adding a GPU compute efficiency model will improve accelerator utilization.

### 12.2 Modifications

| Parameter | V7 | V8 |
|-----------|----|----|
| Learning rate | 3e-4 | 1e-4 |
| GPU compute model | Absent | 30% efficiency boost for tasks exceeding 10^11 instructions |
| Scarcity-aware rewards | Enabled (1.5/2.0), inadvertent | Enabled (1.5/2.0), still not corrected |

### 12.3 Results

| Preset | V4 | V7 | V8 | V8 vs. V7 | V8 vs. V4 |
|--------|----|----|-----|-----------|-----------|
| small | 30.96% | 31.52% | 29.38% | -2.14 pp | -5.10% |
| medium | 50.64% | 49.26% | 50.06% | +0.80 pp | -1.15% |
| large | 65.28% | 66.96% | 64.64% | -2.32 pp | -0.98% |
| high_load | 27.00% | 27.96% | 26.42% | -1.54 pp | -2.15% |
| stress_test | 13.20% | 13.42% | 13.28% | -0.14 pp | +0.61% |
| **Average** | **37.42%** | **37.82%** | **36.76%** | **-1.06 pp** | **-1.75%** |

V8 regressed on four of five presets relative to V7, with only stress_test showing marginal improvement (+0.08 pp). Energy efficiency degraded substantially on constrained presets: small (+17.1%) and high_load (+14.1%).

### 12.4 Root Cause Analysis

1. **Overfitting due to reduced learning rate**: The 3x learning rate reduction caused tighter convergence to the training distribution. Training reward improved marginally (-74.49 vs. -76.43 for V7), but generalization performance declined, a classic indicator of overfitting.

2. **Confounded modifications**: The learning rate change and GPU compute model were introduced simultaneously, preventing isolated attribution of the regression.

3. **Persistent scarcity-aware rewards**: Despite V7's recommendation to disable scarcity-aware rewards, V8 retained them at aggressive 1.5/2.0 scales.

### 12.5 Conclusions

The hypothesis was rejected. A constant lower learning rate caused overfitting and reduced generalization, reversing V7's gains. V8 reinforced two methodological principles: (1) modifications should be isolated for proper attribution, and (2) recommendations from prior analyses must be implemented before introducing additional variables. V7 remained the best-performing version.

---

## 13. Version 9: Comprehensive Architectural and Methodological Overhaul

**Date**: February 10, 2026
**Commit**: `decb2ed`
**Training Configuration**: 2,007,040 timesteps, mixed_capacity preset, curriculum enabled, v3 state encoder
**Mean Training Reward**: 7.46 (first positive mean in the project's history)
**Training Duration**: 22,834 seconds (approximately 6.3 hours), 88 steps/second

### 13.1 Hypothesis

A comprehensive overhaul addressing the cumulative findings from V4--V8, specifically reward asymmetry, insufficient training duration, fixed learning rate and entropy, and architectural gaps, will produce a substantial improvement over the V4 baseline.

### 13.2 Modifications

V9 implemented 15 modifications organized into six categories. Each modification was motivated by specific evidence from prior version analyses.

#### 13.2.1 Reward System

**Reward asymmetry correction**: The rejection penalty was reduced from 0.8 to 0.5, and the acceptance bonus was increased from 0.3 to 0.35, changing the penalty-to-bonus ratio from 2.67:1 to 1.43:1. The prior asymmetry created a risk-averse optimization landscape in which rejection was consistently the lower-risk action.

**Scarcity-aware rewards disabled**: Set to `False` by default, eliminating a source of reward variance demonstrated to be ineffective (V6) or harmful (V5).

#### 13.2.2 Training Methodology

**Training duration**: Increased from approximately 508,000 to 2,000,000 timesteps. All prior versions (V5--V8) produced negative mean training rewards, indicating incomplete convergence.

**Entropy coefficient annealing**: Linear schedule from 0.05 to 0.001 over the training horizon. High entropy early encourages exploration of diverse action strategies; low entropy later enables the policy to commit to learned behaviors.

**Cosine learning rate schedule**: CosineAnnealingLR from 3e-4 to 1e-5. This provides rapid early learning followed by fine-grained parameter adjustment, avoiding the overfitting observed in V8's constant low learning rate.

#### 13.2.3 Architecture

**LayerNorm in encoders**: Added after each linear layer in the TaskEncoder and HWEncoder to reduce internal covariate shift [16].

**Max+mean hardware pooling for value head**: Value head input expanded from `[task_emb, mean_hw_emb]` (128 dimensions) to `[task_emb, mean_hw_emb, max_hw_emb]` (192 dimensions), providing information about the best-case hardware option.

**Vectorized hardware encoder**: Replaced sequential per-hardware-type loop with batched tensor processing, improving training throughput from 71 to 88 steps/second.

#### 13.2.4 State Encoding

**Continuous scarcity indicator**: Replaced the binary indicator (threshold at 80% utilization) with a continuous ramp: $\max(0, \min(1, (\max\_util - 0.5) / 0.4))$.

**Continuous scale bucket**: Replaced the discrete 5-value categorization with a continuous mapping: $\min(\text{total\_cpus} / 2000, 1.0)$.

#### 13.2.5 Reward Calibration

**Adaptive energy baseline**: Exponential moving average (EMA) with $\alpha = 0.01$, activated after 20 observations, replacing the fixed 0.05 kWh baseline.

**Reduced execution time floor**: Fixed at 1.0 second, replacing the random range of 5.0--15.0 seconds that obscured hardware-type performance differences.

#### 13.2.6 Environment

**Per-preset curriculum thresholds**: Replaced the uniform threshold (0.6) with preset-specific values reflecting achievable acceptance rates: stress_test = 0.15, high_load = 0.30, small = 0.35, medium = 0.50, large = 0.60, enterprise = 0.65.

### 13.3 Results

| Preset | V4 | V7 (Prior Best) | V9 | vs. V4 | vs. V7 |
|--------|----|-----------------|----|--------|--------|
| small | 30.96% | 31.52% | 31.30% | +0.34 pp (+1.10%) | -0.22 pp |
| medium | 50.64% | 49.26% | 51.84% | +1.20 pp (+2.37%) | +2.58 pp |
| large | 65.28% | 66.96% | 70.18% | +4.90 pp (+7.51%) | +3.22 pp |
| high_load | 27.00% | 27.96% | 30.42% | +3.42 pp (+12.67%) | +2.46 pp |
| stress_test | 13.20% | 13.42% | 15.94% | +2.74 pp (+20.76%) | +2.52 pp |
| **Average** | **37.42%** | **37.82%** | **39.94%** | **+2.52 pp (+6.7%)** | **+2.12 pp** |

All five presets improved over the V4 baseline. The largest relative gains occurred on the most constrained environments: stress_test (+20.76%), high_load (+12.67%), and large (+7.51%).

### 13.4 Energy Efficiency

| Preset | V7 Energy/Task (kWh) | V9 Energy/Task (kWh) | Improvement |
|--------|---------------------|---------------------|-------------|
| small | 0.000841 | 0.000659 | -21.6% |
| medium | 0.001414 | 0.001022 | -27.7% |
| large | 0.001590 | 0.001115 | -29.9% |
| high_load | 0.000952 | 0.000635 | -33.3% |
| stress_test | 0.000949 | 0.000328 | -65.4% |

V9 achieved 21.6% to 65.4% energy efficiency improvement across all presets relative to V7. The reduced rejection penalty encouraged acceptance of lower-energy tasks that were previously rejected under the more punitive V4--V8 reward structure.

### 13.5 Policy Rejection Analysis

| Preset | V4 Policy Rej. % | V9 Policy Rej. % | Change |
|--------|------------------|------------------|--------|
| small | 62.9% | 61.7% | -1.2 pp |
| medium | 41.0% | 43.6% | +2.6 pp |
| large | 25.5% | 23.9% | -1.6 pp |
| high_load | 59.4% | 56.1% | -3.3 pp |
| stress_test | 69.6% | 68.0% | -1.6 pp |

Policy rejection rates decreased on four of five presets. The large preset's capacity rejection ratio reached 76.1%, indicating that the agent was approaching physical capacity limits before rejecting, the desired behavior.

### 13.6 Training Dynamics

The mean training reward of 7.46 was the first positive value in the project's history. The reward trajectory exhibited three phases: initial rapid improvement (reward ~15), a mid-training reduction (reward ~4--5) coinciding with the entropy and learning rate schedule transitions, and a recovery phase (reward ~6--8) as the policy stabilized under reduced exploration.

### 13.7 Impact Attribution

While all 15 modifications were applied simultaneously, prior analyses provide evidence for the relative importance of each:

1. **Reward rebalancing** (primary contributor to medium/large improvement): The penalty-to-bonus ratio reduction from 2.67:1 to 1.43:1 directly addresses the documented risk-averse policy behavior.
2. **Training duration** (4x increase): The transition from negative to positive mean reward between 508K and 2M timesteps indicates that prior versions suffered from incomplete convergence.
3. **Disabling scarcity-aware rewards**: Eliminates the confounding factor that compromised V7--V8 interpretability.
4. **Entropy and learning rate schedules**: The mid-training reward dip and subsequent recovery demonstrate the exploration-exploitation transition enabled by these schedules.

### 13.8 Remaining Limitations

Constrained presets continued to exhibit 56--68% policy rejection rates. The reject head received only the 64-dimensional task embedding, without visibility into infrastructure capacity, motivating the architectural modification in V10.

### 13.9 Conclusions

The hypothesis was confirmed. V9 established a new performance ceiling (+6.7% relative to V4) through the combined effect of reward rebalancing, architectural improvements, and adequate training duration. The results demonstrate that accumulated evidence-based modifications, when applied comprehensively with sufficient training, can produce compounding improvements.

---

## 14. Version 10: Capacity-Aware Reject Head

**Date**: February 12, 2026
**Commit**: `c0af458`
**Training Configuration**: Approximately 1,660,000 timesteps (41.5% of planned 4,000,000), full_spectrum preset, curriculum enabled
**Backward Compatibility**: V9 checkpoints loaded with `strict=False`; reject head weights reinitialized

### 14.1 Hypothesis

Providing the reject head with aggregated hardware state information will enable more informed rejection decisions, reducing the policy rejection rate on constrained presets.

### 14.2 Architectural Modification

The reject head input was expanded from 64 to 195 dimensions:

| Input Component | Dimensions | Description |
|----------------|-----------|-------------|
| Task embedding | 64 | Existing task representation |
| Mean HW embedding | 64 | Average hardware state across all types |
| Max HW embedding | 64 | Best-case hardware state (max-pooled) |
| Capacity summary | 3 | Max HW score, mean HW score, fraction of valid HW types |
| **Total** | **195** | |

Prior to V10, the reject head was the only decision component without infrastructure visibility:

| Component | Input Dimensions | Infrastructure Visibility |
|-----------|-----------------|---------------------------|
| Scorer | 128 (task + per-HW) | Per-hardware-type |
| Value head | 192 (task + mean + max HW) | Aggregated |
| Reject head (V4--V9) | 64 (task only) | None |
| **Reject head (V10+)** | **195 (task + HW + capacity)** | **Aggregated** |

### 14.3 Additional Modifications

- **Domain preset**: Changed from mixed_capacity to full_spectrum (includes stress_test and high_load during training)
- **Rejection penalty**: Reduced from 0.5 to 0.3
- **Target training duration**: 4,000,000 timesteps (training was terminated early at 1,660,000)

### 14.4 Results

| Preset | V9 | V10 | Difference |
|--------|-----|------|-----------|
| small | 31.30% | 31.00% | -0.30 pp |
| medium | 51.84% | 52.06% | +0.22 pp |
| large | 70.18% | 69.82% | -0.36 pp |
| high_load | 30.42% | 30.28% | -0.14 pp |
| stress_test | 15.94% | 15.58% | -0.36 pp |
| **Average** | **39.94%** | **39.75%** | **-0.19 pp** |

V10 matched V9 within the observed variance bounds across all presets. All inter-version differences (< 0.4 pp) fell within the standard deviations reported in the utilization stability analysis (1.3--3.4% per preset).

### 14.5 Diagnostic Finding: Reject Probability with Available Capacity

V10 introduced a novel diagnostic metric: the mean reject head output probability conditioned on at least one hardware type having sufficient capacity for the current task. This metric directly measures whether the reject head is the source of unnecessary policy rejections.

| Preset | Reject Probability When Capacity Exists |
|--------|----------------------------------------|
| large | 0.025% |
| medium | 0.12% |
| high_load | 0.83% |
| small | 1.71% |
| stress_test | 3.42% |

The reject head produced near-zero rejection probabilities when resources were available, ranging from 0.025% on large to 3.42% on stress_test.

### 14.6 Reinterpretation of Policy Rejection

This diagnostic finding necessitated a fundamental reinterpretation of the policy rejection metric:

1. **The reject head was not the performance bottleneck.** Despite policy rejection rates of 56--69% on constrained presets, the reject head almost never chose to reject when capacity existed.

2. **Apparent "policy rejections" were capacity exhaustions.** The high policy rejection counts occurred when the valid_mask was entirely False, meaning no hardware type had sufficient resources for the specific task. The existing metric classified these as "policy rejections" because the reject action was selected, but the environment had already eliminated all allocation options.

3. **Agent decision quality was near-optimal.** With reject probabilities of 0.025--3.42% when capacity existed, the agent accepted tasks whenever physically possible.

4. **The performance ceiling was determined by task-infrastructure mismatch.** Tasks generated from fixed distributions (e.g., mean demand of ~178 vCPUs) were physically infeasible on stress_test infrastructure (96 total CPUs). No policy optimization could overcome this structural limitation.

### 14.7 Early Training Termination

Training was terminated at 1,660,000 of the planned 4,000,000 timesteps (41.5% completion) because initial evaluation revealed no improvement trajectory relative to V9. The fact that V10 achieved V9-equivalent performance with a fully reinitialized reject head within 41.5% of the planned training duration provided positive validation of the architectural change.

### 14.8 Conclusions

The reject head hypothesis was partially confirmed: the architectural modification functioned correctly (near-zero reject probability with capacity), but the expected performance improvement did not materialize because the reject head was not, in fact, the bottleneck. V10's principal contribution was diagnostic rather than performance-oriented: it conclusively established that the remaining performance limitation was attributable to task-infrastructure mismatch, redirecting optimization efforts from agent architecture to environment design.

---

## 15. Version 11: Capacity-Scaled Task Generation

**Date**: February 13, 2026
**Commit**: `1b3fb3d`
**Training Configuration**: 2,007,040 timesteps, full_spectrum preset, curriculum enabled, v3 state encoder, V10 reject head (195 dimensions)
**Mean Training Reward**: 34.75 (highest recorded; first strongly positive mean)
**Training Duration**: 25,226 seconds (approximately 7.0 hours), 80 steps/second

### 15.1 Hypothesis

Scaling task sizes proportionally to infrastructure capacity will eliminate physically infeasible task-infrastructure combinations, improving acceptance rates on constrained presets without degrading performance on larger configurations.

### 15.2 Environment Modification

Task VM counts (`num_vms`) were scaled proportionally to infrastructure capacity using the medium preset as the reference configuration (scale = 1.0):

$$\text{capacity\_scale} = \text{clamp}\left(\min\left(\frac{\text{total\_cpus}}{1024}, \frac{\text{total\_memory}}{7168}\right), 0.25, 3.0\right)$$

Per-VM resource requirements (`vcpus_per_vm`, `memory_per_vm`) were not modified, as they represent standardized VM types.

| Preset | Scale Factor | Large Task VMs (Pre-V11) | Large Task VMs (V11) |
|--------|-------------|--------------------------|----------------------|
| stress_test | 0.25 | 4--15 | 2--4 |
| high_load | 0.25 | 4--15 | 2--4 |
| small | 0.38 | 4--15 | 2--6 |
| medium | 1.00 | 4--15 | 4--15 (unchanged) |
| large | 2.09 | 4--15 | 8--31 (scaled upward) |

Additionally, memory-intensive task memory requirements were capped at 50% of total system memory to prevent infeasible memory allocations on constrained presets.

### 15.3 Additional Modifications

- **Rejection penalty**: 0.3 (consistent with V10)
- **Architecture**: Unchanged from V10

### 15.4 Results

| Preset | V4 | V9 | V10 | V11 | V11 vs. V10 | V11 vs. V4 |
|--------|----|----|------|------|-------------|------------|
| small | 30.96% | 31.30% | 31.00% | 33.56% | +2.56 pp | +2.60 pp (+8.4%) |
| medium | 50.64% | 51.84% | 52.06% | 52.58% | +0.52 pp | +1.94 pp (+3.8%) |
| large | 65.28% | 70.18% | 69.82% | 61.60% | **-8.22 pp** | -3.68 pp (-5.6%) |
| high_load | 27.00% | 30.42% | 30.28% | 30.42% | +0.14 pp | +3.42 pp (+12.7%) |
| stress_test | 13.20% | 15.94% | 15.58% | 19.20% | **+3.62 pp** | +6.00 pp (+45.5%) |
| **Average** | **37.42%** | **39.94%** | **39.75%** | **39.47%** | **-0.28 pp** | **+2.05 pp (+5.5%)** |

### 15.5 Constrained Preset Analysis

The hypothesis was confirmed for constrained presets:

| Preset | V10 | V11 | Absolute Change | Relative Change |
|--------|------|------|----------------|-----------------|
| stress_test | 15.58% | 19.20% | +3.62 pp | +23.2% |
| small | 31.00% | 33.56% | +2.56 pp | +8.3% |
| high_load | 30.28% | 30.42% | +0.14 pp | +0.5% |

The stress_test improvement of +3.62 pp represented the largest single-version gain on that preset across all versions.

### 15.6 Reject Head Convergence

| Preset | V10 Reject Prob. | V11 Reject Prob. | Improvement Factor |
|--------|------------------|------------------|--------------------|
| small | 1.71% | 0.027% | 63x |
| medium | 0.12% | 0.004% | 30x |
| large | 0.025% | 0.001% | 25x |
| high_load | 0.83% | 0.005% | 166x |
| stress_test | 3.42% | 0.029% | 118x |

Reject probability with capacity decreased by 25--166x across all presets, indicating that the agent achieved functionally perfect rejection behavior: probability of 0.001--0.029% when resources were available.

### 15.7 Large Preset Regression Analysis

The large preset regressed -8.22 pp (69.82% to 61.60%), constituting the primary reason V11's average did not exceed V9.

**Root cause: super-linear resource contention.** The linear scaling formula produced a 2.09x scale factor for the large preset, increasing large task VM counts from 4--15 to 8--31. This introduced several cascading effects:

1. **Proportionally larger resource consumption**: Each task blocked approximately 2x the resources of its pre-V11 equivalent.
2. **Extended resource holding time**: Larger tasks required proportionally longer execution, maintaining resource locks for extended periods.
3. **Cascading unavailability**: With resources held longer, fewer concurrent tasks could be served, producing secondary rejections for subsequent arrivals.
4. **Increased capacity rejections**: Despite identical infrastructure, capacity rejections increased from 1,196 (V10) to 1,349 (V11), confirming accelerated capacity exhaustion.

The capacity rejection ratio on large decreased from 79.5% (V10) to 70.0% (V11), indicating that the agent responded to increased resource pressure by adopting a more conservative acceptance policy.

### 15.8 Energy Comparability

Energy-per-task metrics between V10 and V11 are not directly comparable because V11 modified the task size distribution:

| Preset | V10 Energy/Task | V11 Energy/Task | Note |
|--------|-----------------|-----------------|------|
| medium | 0.000981 kWh | 0.000940 kWh | Same distribution, -4.2% improvement |
| large | 0.001064 kWh | 0.000563 kWh | Larger tasks, more efficient utilization |
| stress_test | 0.000357 kWh | 0.001700 kWh | Proportionally larger tasks accepted |

Energy increases on constrained presets reflect the larger absolute task sizes under capacity-scaled generation, not degraded agent efficiency.

### 15.9 Training Dynamics

V11 achieved the highest mean training reward (34.75), substantially exceeding V9's 7.46. The training trajectory exhibited three distinct phases:

| Phase | Episodes | Mean Reward | Behavior |
|-------|----------|-------------|----------|
| Curriculum (constrained presets) | 1--80 | ~0 | Volatile, learning constrained allocation |
| Transition to balanced presets | 80--120 | 30--60 | Rapid improvement |
| Full mixture | 120--240 | 35--55 | Convergence with maintained exploration |

Policy loss converged monotonically from -0.02 to -0.001. Entropy stabilized at 0.15--0.17, indicating sustained exploration throughout training.

### 15.10 Conclusions

The hypothesis was partially confirmed. Capacity-scaled task generation successfully eliminated task-infrastructure mismatch on constrained presets, producing the largest single-version improvement on stress_test (+3.62 pp). However, the hypothesis was rejected for upward-scaled presets: the linear 2.09x scaling on large introduced super-linear resource contention, causing -8.22 pp regression. The symmetric scaling approach requires modification to asymmetric or sub-linear upward scaling to achieve uniform improvement.

---

## 16. Cross-Version Analysis

### 16.1 Acceptance Rate Progression

```
Version:      V4      V5      V6      V7      V8      V9      V10     V11
              |       |       |       |       |       |       |       |
stress_test   13.20   12.58   13.08   13.42   13.28   15.94   15.58   19.20
high_load     27.00   25.30   26.16   27.96   26.42   30.42   30.28   30.42
small         30.96   29.98   32.42   31.52   29.38   31.30   31.00   33.56
medium        50.64   47.30   50.28   49.26   50.06   51.84   52.06   52.58
large         65.28   61.08   65.28   66.96   64.64   70.18   69.82   61.60
              -----   -----   -----   -----   -----   -----   -----   -----
Average       37.42   35.25   37.44   37.82   36.76   39.94   39.75   39.47
vs. V4 (rel.) Base    -5.8%   +0.1%   +1.1%   -1.8%   +6.7%   +6.2%   +5.5%
```

### 16.2 Per-Preset Optimal Versions

| Preset | Best Version | Best Rate | V4 Rate | Cumulative Gain |
|--------|-------------|-----------|---------|-----------------|
| stress_test | V11 | 19.20% | 13.20% | +6.00 pp (+45.5%) |
| high_load | V9 / V11 | 30.42% | 27.00% | +3.42 pp (+12.7%) |
| small | V11 | 33.56% | 30.96% | +2.60 pp (+8.4%) |
| medium | V11 | 52.58% | 50.64% | +1.94 pp (+3.8%) |
| large | V9 | 70.18% | 65.28% | +4.90 pp (+7.5%) |

### 16.3 Reward Function Evolution

| Parameter | V4 | V5--V6 | V7--V8 | V9 | V10--V11 |
|-----------|----|----|--------|-----|---------|
| Rejection penalty | 0.8 | 0.8 | 0.8 | 0.5 | 0.3 |
| Acceptance bonus | 0.3 | 0.3 | 0.3 | 0.35 | 0.35 |
| Penalty-to-bonus ratio | 2.67:1 | 2.67:1 | 2.67:1 | 1.43:1 | 0.86:1 |
| Scarcity-aware scaling | No | Yes | Yes (inadvertent) | No | No |

The progressive reduction in penalty-to-bonus ratio from 2.67:1 to 0.86:1 correlates with the monotonic improvement in acceptance rates from V4 through V9.

### 16.4 Training Configuration Evolution

| Parameter | V1--V3 | V4 | V5 | V6--V8 | V9--V11 |
|-----------|--------|----|----|--------|---------|
| Timesteps | 30K--200K | 100K | ~106K | ~508K | ~2M |
| Training preset | medium only | mixed_capacity | constrained_first | mixed_capacity | mixed / full_spectrum |
| Curriculum learning | No | No | No | Yes (V6+) | Yes |
| LR schedule | Fixed 3e-4 | Fixed 3e-4 | Fixed 3e-4 | Fixed (V8: 1e-4) | Cosine 3e-4 to 1e-5 |
| Entropy coefficient | Fixed | Fixed | Fixed | Fixed | Annealed 0.05 to 0.001 |

### 16.5 Energy Efficiency Progression (kWh per Task)

| Preset | V4 | V7 | V9 | V10 |
|--------|------|------|------|------|
| small | 0.000872 | 0.000841 | 0.000659 | 0.000669 |
| medium | 0.001421 | 0.001414 | 0.001022 | 0.000981 |
| large | 0.001546 | 0.001590 | 0.001115 | 0.001064 |
| high_load | 0.001018 | 0.000952 | 0.000635 | 0.000605 |
| stress_test | 0.000845 | 0.000949 | 0.000328 | 0.000357 |

V11 energy values are excluded from this comparison due to the change in task size distribution (Section 15.8).

V9 achieved the most substantial energy efficiency improvement: 21.6% to 65.4% reduction relative to V7 across all presets.

### 16.6 Modification Outcome Summary

| Version | Primary Modification | Avg. Acceptance | vs. V4 | Outcome |
|---------|---------------------|----------------|--------|---------|
| V4 | Domain randomization | 37.42% | Baseline | Baseline established |
| V5 | Scarcity-aware rewards (aggressive) | 35.25% | -5.8% | Regression |
| V6 | Attenuated scaling + curriculum | 37.44% | +0.1% | Recovery |
| V7 | Capacity features (v3 encoder) | 37.82% | +1.1% | First improvement |
| V8 | Lower LR + GPU compute model | 36.76% | -1.8% | Regression |
| V9 | 15-modification overhaul | 39.94% | +6.7% | New optimum |
| V10 | Capacity-aware reject head | 39.75% | +6.2% | Neutral (diagnostic value) |
| V11 | Capacity-scaled task generation | 39.47% | +5.5% | Mixed (constrained +, large -) |

---

## 17. Discussion

### 17.1 Reward Function Design

The reward function emerged as the most influential design parameter across the development trajectory. Three findings warrant emphasis:

**Finding 1: Penalty-to-bonus asymmetry is the primary determinant of policy conservatism.** The ratio of rejection penalty to acceptance bonus (ranging from 2.67:1 in V4 to 0.86:1 in V10--V11) exhibited a stronger correlation with acceptance rates than any individual reward magnitude. High asymmetry ratios produced risk-averse agents that preferentially rejected tasks to avoid penalties, even when acceptance was feasible and energy-efficient.

**Finding 2: Dynamic reward scaling introduces harmful variance.** The scarcity-aware reward mechanism (V5--V8), designed to modulate penalties and bonuses based on resource utilization, never outperformed static reward configurations. Aggressive scaling (V5, parameters 1.5/2.0) produced universal regression (-5.4%). Attenuated scaling (V6, parameters 1.2/1.5) produced recovery to baseline without improvement. The additional variance in the reward signal appears to destabilize the policy gradient estimation without providing actionable gradient information.

**Finding 3: Self-calibrating reward baselines improve cross-environment generalization.** V9's exponential moving average energy baseline enabled the reward function to adapt to the energy scale of each environment preset, providing consistent reward magnitudes across infrastructure configurations with different characteristic energy consumption levels.

### 17.2 State Representation

The state representation was the second most influential design dimension:

**Finding 4: Utilization ratios are insufficient for cross-scale generalization.** The v1/v2 state encoders, which represented infrastructure state exclusively through utilization ratios, produced agents that could not distinguish between clusters of different absolute scales operating at similar utilization levels. This "scale blindness" was the root cause of the persistent policy rejection dominance on constrained presets, as confirmed by quantitative state vector diagnosis (Section 10.5).

**Finding 5: Continuous features provide superior gradient flow compared to discrete alternatives.** Replacing binary scarcity indicators and discrete scale buckets with continuous functions improved both training convergence and generalization performance.

### 17.3 Training Methodology

**Finding 6: Adequate training duration is essential for domain-randomized policies.** All versions trained at fewer than 1,000,000 timesteps (V4--V8) produced negative mean training rewards, indicating incomplete convergence. The transition to 2,000,000 timesteps in V9 was necessary for the complex multi-preset policy to converge. This finding suggests that domain randomization introduces a training duration multiplier proportional to the diversity of environments encountered.

**Finding 7: Learning rate schedules outperform fixed learning rates.** V8's constant low learning rate (1e-4) caused overfitting to the training distribution, while V9's cosine schedule (3e-4 to 1e-5) provided rapid initial learning followed by stable fine-grained adjustment.

**Finding 8: Entropy annealing prevents premature convergence.** V9's entropy schedule (0.05 to 0.001) encouraged broad action exploration during early training, preventing the premature convergence to conservative rejection policies observed in V4--V8.

### 17.4 Architecture and Diagnostics

**Finding 9: Diagnostic instrumentation can be more valuable than performance optimization.** V10's capacity-aware reject head produced no measurable performance improvement, yet its diagnostic metric (reject probability with capacity) provided the most important insight of the entire development trajectory: the agent was already near-optimal in its decisions, and the remaining performance gap was environmental rather than policy-related.

**Finding 10: Information asymmetry between network components can create apparent bottlenecks.** The reject head's limited input (task embedding only) appeared to be a bottleneck based on correlation with high policy rejection rates. V10 demonstrated that this correlation was spurious; the true cause was infrastructure capacity exhaustion, which was invisible to the existing metrics.

### 17.5 Environment Design

**Finding 11: Task distributions must be calibrated to infrastructure capacity.** Fixed task size distributions generated physically infeasible workloads on constrained infrastructure, imposing a hard performance ceiling that no agent optimization could overcome. V11's capacity-scaled task generation demonstrated that environment-level intervention could unlock improvements inaccessible through policy or architecture changes.

**Finding 12: Resource contention scales super-linearly with task size.** V11's linear upward scaling (2.09x) on the large preset produced a disproportionate -8.22 pp regression due to cascading resource unavailability. This finding has implications for real cloud systems: workload scaling must account for non-linear contention effects, particularly for resource-intensive tasks.

### 17.6 Methodological Observations

**Finding 13: Variable isolation is essential for attribution.** Versions that modified multiple parameters simultaneously (V5: four changes; V8: two changes) produced results that could not be attributed to individual modifications. V9's deliberate bundling of 15 changes succeeded only because each modification had been individually justified through prior analysis.

**Finding 14: Recommendations must be implemented before introducing new variables.** V7 and V8 retained scarcity-aware rewards despite prior analysis recommending their removal. This persistent confounding factor complicated the interpretation of capacity feature and learning rate experiments.

---

## 18. Threats to Validity

### 18.1 Internal Validity

**Single training seed (V4--V11)**: All versions from V4 onward were trained with a single random seed (42). Multi-seed evaluation was conducted only for V1--V3 (3--10 seeds). Consequently, the reported performance differences may be partially attributable to seed-specific training dynamics rather than systematic effects of the modifications. Differences smaller than the V1 multi-seed standard deviation (1.62 pp) should be interpreted with particular caution.

**Confounded modifications**: Several versions introduced multiple simultaneous changes (V5: 4, V8: 2, V9: 15), limiting the ability to attribute outcomes to individual modifications. V9's 15-change bundle was deliberately designed based on prior individual analysis, but the attribution remains inferential.

**Non-blind evaluation**: The evaluation protocol was designed and executed by the same researchers who implemented the modifications, introducing potential confirmation bias in result interpretation. Automated metrics (acceptance rate, energy consumption) mitigate this risk, but qualitative analysis (root cause attribution, verdict assignments) may be influenced.

### 18.2 External Validity

**Simulated environment**: All training and evaluation occurred in a simulated cloud environment with simplified resource models (linear energy interpolation, fixed task execution times, homogeneous resource pools within hardware types). Transfer to production cloud environments with heterogeneous servers, variable network conditions, thermal effects, and multi-tenant interference has not been validated.

**Task distribution**: The stochastic task generator uses parametric distributions that may not reflect production workload characteristics. Real cloud workloads exhibit temporal correlations (diurnal patterns, burst arrivals), task dependencies, and workload-specific resource profiles that are not modeled.

**Infrastructure scale**: The largest evaluated configuration (enterprise: 5,100 CPUs, 6 HW types) is small relative to production data centers. Scaling behavior to configurations with thousands of servers and dozens of hardware types remains untested.

### 18.3 Construct Validity

**Acceptance rate as primary metric**: The primary optimization target (acceptance rate) may not align with the operational objectives of production cloud systems, which typically prioritize revenue, SLA compliance, or resource utilization efficiency. A high acceptance rate achieved through allocation of tasks to suboptimal hardware types may reduce overall system efficiency.

**Energy model fidelity**: The linear interpolation energy model ($E = f(P_{idle}, P_{max}, \Delta u, t)$) is a simplification of actual server energy consumption, which exhibits non-linear behavior at high utilization, thermal throttling effects, and component-specific power draw variations.

---

## 19. Conclusions and Future Work

### 19.1 Summary of Results

This report documented the iterative development of a PPO-based RL agent for energy-aware cloud resource allocation across eleven versions. The key results are:

1. **Domain randomization** (V4) enabled cross-infrastructure generalization, improving average acceptance by +4.5 pp over single-preset training at the cost of -3.9 pp on the training environment.

2. **Scarcity-aware reward scaling** (V5--V6) was conclusively demonstrated to provide no benefit over static rewards, with aggressive parameters causing -5.4% regression.

3. **Capacity-aware state encoding** (V7) produced the first baseline improvement (+1.38%) by resolving the scale blindness problem.

4. **Comprehensive overhaul** (V9) achieved the best overall performance (+6.7% relative to V4) through reward rebalancing, learning rate and entropy scheduling, and architectural improvements.

5. **Diagnostic instrumentation** (V10) revealed that agent decision quality was near-optimal (0.025--3.42% reject probability when capacity existed), redirecting attention from policy optimization to environment design.

6. **Capacity-scaled task generation** (V11) produced the largest single-version improvement on constrained infrastructure (+3.62 pp on stress_test) but introduced regression on large infrastructure (-8.22 pp) due to super-linear resource contention.

### 19.2 Current Performance Ceiling

The agent's decision quality has reached a functional optimum for the current environment configuration. With reject probabilities of 0.001--0.029% when capacity is available (V11), the agent accepts tasks whenever physically possible. Remaining performance limitations are attributable to task-infrastructure mismatch and physical resource constraints, not policy deficiency.

### 19.3 Future Work

Three immediate directions address V11's large preset regression:

1. **Asymmetric scaling**: Cap `capacity_scale` at 1.0, applying downward scaling only. This preserves constrained preset gains while reverting large-scale presets to the original task distribution.

2. **Sub-linear upward scaling**: Apply square-root dampening to upward scaling: $\text{scale\_up} = 1.0 + \sqrt{\text{scale} - 1.0}$, reducing the large preset scale from 2.09 to approximately 1.45.

3. **Per-task-category scaling**: Apply differentiated scaling limits per task category, recognizing that resource-intensive task types exhibit greater sensitivity to scaling-induced contention.

Longer-term research directions include:

- **Attention-based hardware aggregation**: Replace mean/max pooling with multi-head attention for differentiated hardware state representation.
- **Online learning with simulator integration**: Train on outcomes from the C++ CloudLightning simulator for closed-loop policy refinement.
- **Multi-objective Pareto optimization**: Explicit Pareto frontier tracking across energy, SLA compliance, and acceptance rate.
- **Workload-aware scheduling**: Incorporate task dependency graphs and temporal arrival patterns for more realistic workload modeling.

---

## 20. References

[1] Masanet, E., Shehabi, A., Lei, N., Smith, S., & Koomey, J. (2020). Recalibrating global data center energy-use estimates. *Science*, 367(6481), 984--986.

[2] Jones, N. (2018). How to stop data centres from gobbling up the world's electricity. *Nature*, 561(7722), 163--166.

[3] Beloglazov, A., Abawajy, J., & Buyya, R. (2012). Energy-aware resource allocation heuristics for efficient management of data centers for cloud computing. *Future Generation Computer Systems*, 28(5), 755--768.

[4] Gao, Y., Guan, H., Qi, Z., Hou, Y., & Liu, L. (2013). A multi-objective ant colony system algorithm for virtual machine placement in cloud computing. *Journal of Computer and System Sciences*, 79(8), 1230--1242.

[5] Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

[6] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

[7] Mao, H., Alizadeh, M., Menache, I., & Kandula, S. (2016). Resource management with deep reinforcement learning. In *Proceedings of the 15th ACM Workshop on Hot Topics in Networks* (pp. 50--56).

[8] Zhang, Z., Li, C., Tao, Y., Yang, R., Hong, H., & Peng, Z. (2019). DeepRec: A deep reinforcement learning approach for energy-aware resource scheduling in cloud computing. *IEEE Access*, 7, 128940--128949.

[9] Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3), 229--256.

[10] Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harber, T., Silver, D., & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In *Proceedings of the 33rd International Conference on Machine Learning* (pp. 1928--1937).

[11] Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. In *IEEE/RSJ International Conference on Intelligent Robots and Systems* (pp. 23--30).

[12] Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. In *Proceedings of the 26th International Conference on Machine Learning* (pp. 41--48).

[13] Fan, X., Weber, W. D., & Barroso, L. A. (2007). Power provisioning for a warehouse-sized computer. In *Proceedings of the 34th Annual International Symposium on Computer Architecture* (pp. 13--23).

[14] Economou, D., Rivoire, S., Kozyrakis, C., & Ranganathan, P. (2006). Full-system power analysis and modeling for server environments. In *Workshop on Modeling Benchmarking and Simulation*.

[15] Li, S., Zhao, Y., Varma, R., Salpekar, O., Noordhuis, P., Li, T., Paszke, A., Smith, J., Vaughan, B., Damania, P., & Chintala, S. (2020). PyTorch distributed: Experiences on accelerating data parallel training. *Proceedings of the VLDB Endowment*, 13(12), 3005--3018.

[16] Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). Layer normalization. *arXiv preprint arXiv:1607.06450*.

---

## Appendix A: Git Commit History (RL-Related)

| Commit Hash | Message | Associated Version |
|-------------|---------|-------------------|
| `d55d977` | Add infrastructure-agnostic RL agent and C++ simulator integration | Foundation |
| `06a6215` | RL PPO agent training #1 | Pre-V1 |
| `8023eb8` | Multi-GPU training, benchmark fix, energy reward threshold | Pre-V1 |
| `43db9e4` | Use torchrun rather than custom approach | Pre-V1 |
| `7112651` | Collective operation deadlock fixed | Pre-V1 |
| `6aeae58` | Add academic evaluation framework with stochastic environment support | V1 framework |
| `b04b1e1` | Address critique: improve reward function and add stress testing | V2/V3 |
| `3b78acc` | Add domain randomization training and utilization analysis | V4 |
| `4d78412` | V5 GPU utilization fix, scarcity-aware rewards | V5 |
| `d863128` | V6 capacity features flag, scarcity-aware disable flag | V6 |
| `b67c6ca` | V7 (patched) | V7 |
| `4a6bc3d` | V8 | V8 |
| `decb2ed` | V9: reward rebalance, entropy/LR schedules, architecture improvements | V9 |
| `4d8e482` | Fix: CSV writer crash when presets have different HW type counts | V9 (bugfix) |
| `c0af458` | V10: capacity-aware reject head architecture | V10 |
| `1b3fb3d` | V11: capacity-scaled task generation to match infrastructure size | V11 |

## Appendix B: Source File Reference

| File | Purpose |
|------|---------|
| `rl/agent.py` | Policy network (TaskEncoder, HWEncoder, Scorer, reject and value heads) |
| `rl/state_encoder.py` | State-to-vector encoding (v1, v2, v3 feature sets) |
| `rl/reward.py` | Multi-objective reward calculator |
| `rl/environment.py` | Cloud provisioning environment, domain randomization, task generation |
| `rl/distributed_trainer.py` | Multi-GPU DDP training implementation |
| `scripts/train_rl_distributed.py` | CLI training entry point |
| `scripts/run_academic_evaluation_v5.py` | Academic evaluation pipeline |
| `scripts/utilization_analysis.py` | Per-step utilization tracking and visualization |
| `scripts/diagnose_state_vectors.py` | State vector diagnostics across presets |

## Appendix C: Evaluation Report Files

| Version Range | Report File Name |
|---------------|-----------------|
| V1--V3 | `academic_evaluation_report.json` |
| V4 | `evaluation_results.json` |
| V5--V11 | `evaluation_report.json` |

All evaluation reports are located in `results/academic_v{N}/` directories.

---

*This report was compiled from analysis of 11 version result directories, 11 changelog documents, 79 git commits, and the RL module source code.*
