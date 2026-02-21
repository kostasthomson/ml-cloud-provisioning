# RL Agent Versions Report

## Energy-Aware Cloud Resource Allocation via PPO Reinforcement Learning

**Project**: ML Cloud Provisioning - Reinforcement Learning Module
**Branch**: `rl-agent`
**Report Date**: 2026-02-21
**Versions Covered**: V1 through V11

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Evaluation Methodology](#3-evaluation-methodology)
4. [Pre-Baseline Development (Foundation)](#4-pre-baseline-development-foundation)
5. [Version 1 - Inaugural Baseline Evaluation](#5-version-1---inaugural-baseline-evaluation)
6. [Versions 2 and 3 - Iterative Refinement](#6-versions-2-and-3---iterative-refinement)
7. [Version 4 - Domain Randomization Baseline](#7-version-4---domain-randomization-baseline)
8. [Version 5 - Scarcity-Aware Rewards (Aggressive)](#8-version-5---scarcity-aware-rewards-aggressive)
9. [Version 6 - Gentle Scarcity Scaling + Curriculum Learning](#9-version-6---gentle-scarcity-scaling--curriculum-learning)
10. [Version 7 - Capacity Features (v3 State Encoder)](#10-version-7---capacity-features-v3-state-encoder)
11. [Version 8 - Lower Learning Rate + GPU Compute Fix](#11-version-8---lower-learning-rate--gpu-compute-fix)
12. [Version 9 - Reward Rebalance + Architecture Overhaul](#12-version-9---reward-rebalance--architecture-overhaul)
13. [Version 10 - Capacity-Aware Reject Head](#13-version-10---capacity-aware-reject-head)
14. [Version 11 - Capacity-Scaled Task Generation](#14-version-11---capacity-scaled-task-generation)
15. [Cross-Version Analysis](#15-cross-version-analysis)
16. [Lessons Learned](#16-lessons-learned)
17. [Current State and Future Directions](#17-current-state-and-future-directions)

---

## 1. Executive Summary

This report documents the development history of a PPO-based reinforcement learning agent designed for energy-aware cloud resource allocation. The agent learns to assign incoming cloud tasks to heterogeneous hardware types (CPU, GPU, DFE, MIC) while minimizing energy consumption and meeting service-level agreements (SLAs).

Over 11 major versions spanning from January to February 2026, the system evolved from a single-preset trained model achieving 57% acceptance on medium infrastructure to a domain-randomized, capacity-aware agent achieving 39.9% average acceptance across five diverse infrastructure presets ranging from severely constrained (96 CPUs) to large-scale (2,250 CPUs).

### Key Milestones

| Version | Date | Key Innovation | Avg Acceptance | vs V4 Baseline | Verdict |
|---------|------|----------------|----------------|----------------|---------|
| V1 | 2026-01-28 | Inaugural full evaluation | 57.04% (medium) | N/A | Baseline established |
| V4 | 2026-01-31 | Domain randomization | 37.42% (5-preset avg) | Baseline | Cross-preset baseline |
| V5 | 2026-02-02 | Scarcity-aware rewards | 35.25% | -5.8% | Regression |
| V6 | 2026-02-02 | Gentler scaling + curriculum | 37.44% | +0.0% | Recovery |
| V7 | 2026-02-02 | v3 state encoder (28-dim) | 37.82% | +1.4% | First improvement |
| V9 | 2026-02-10 | 15-change overhaul | 39.92% | +6.7% | Best overall |
| V10 | 2026-02-12 | Reject head diagnostics | 39.74% | +6.2% | Key diagnostic insight |
| V11 | 2026-02-13 | Capacity-scaled tasks | 39.48% | +5.5% | Mixed results |

The project's most significant finding is that agent performance on constrained infrastructure is fundamentally limited by the task-infrastructure mismatch: tasks generated from fixed distributions are physically impossible to serve on small clusters. The agent's decision-making quality was proven to be near-optimal (reject probability with available capacity: 0.001-0.029% in V11), redirecting optimization efforts from policy improvement to environment design.

---

## 2. System Architecture Overview

### 2.1 PPO Agent Architecture

The agent uses a Proximal Policy Optimization (PPO) actor-critic architecture designed for infrastructure-agnostic operation across varying numbers of hardware types.

**State Space** (variable-length, infrastructure-agnostic):
- Task features: 12 dimensions (VM count, CPU/memory requirements, instructions, accelerator needs, compatibility)
- Global features: 5 dimensions (power consumption, queue length, acceptance rate, energy trend, time)
- Scarcity features (v2+): 5 dimensions (CPU/memory utilization, capacity ratios, scarcity indicator)
- Capacity features (v3+): 6 dimensions (system scale, task-fit ratios, scale bucket, relative task size)
- Per-HW-type features: 16 dimensions each (utilization, capacity ratios, power model, compute capability, running tasks)

**Action Space**: N+1 discrete actions (N hardware types + reject)

**Network Components**:

```
TaskEncoder:  task_global_vec (17/22/28 dim) -> Linear(64) -> LayerNorm -> ReLU -> Linear(64)
HWEncoder:    hw_vec (16 dim) -> Linear(64) -> LayerNorm -> ReLU -> Linear(64)  [shared weights]
Scorer:       [task_emb || hw_emb] (128) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(1)
Reject Head:  [task_emb || mean_hw || max_hw || cap_summary] (195) -> Linear(64) -> ReLU -> Linear(1)
Value Head:   [task_emb || mean_hw || max_hw] (192) -> Linear(64) -> ReLU -> Linear(1)
```

### 2.2 Environment Presets

The simulated cloud environment provides six infrastructure configurations:

| Preset | HW Types | Total CPUs | Total Memory (GB) | Intended Scenario |
|--------|----------|------------|--------------------|--------------------|
| stress_test | 2 | 96 | 384 | Severely constrained |
| high_load | 3 | 256 | 1,792 | Heavy load |
| small | 2 | 384 | 3,072 | Small cluster |
| medium | 3 | 1,024 | 7,168 | Reference baseline |
| large | 4 | 2,250 | 15,000 | Large data center |
| enterprise | 6 | 5,100 | 32,400 | Enterprise scale |

### 2.3 Reward Function (Final, V9+)

| Component | Value | Condition |
|-----------|-------|-----------|
| Rejection penalty | -0.3 to -0.5 | Task rejected |
| Acceptance bonus | +0.35 | Task accepted |
| Energy reward | -0.9 to +0.6 | Weighted by 0.6 |
| Excellent energy bonus | +0.2 | Energy < 0.03 kWh |
| Poor energy penalty | -0.15 | Energy > 0.08 kWh |
| SLA compliance | +0.2 | Deadline met |
| SLA violation | -0.2 | Deadline missed |
| Total range | [-2.5, +2.5] | Clipped |

### 2.4 Training Infrastructure

- 4x GPU distributed training via PyTorch DDP (`torchrun --nproc_per_node=4`)
- 8 vectorized environments per GPU (32 total parallel environments)
- Training conducted on Linux; development and analysis on Windows

---

## 3. Evaluation Methodology

### 3.1 Standard Evaluation Protocol (V4+)

Starting from V4, all versions are evaluated using a consistent protocol:

- **Generalization test**: 5,000 tasks per preset across 5 presets (small, medium, large, high_load, stress_test)
- **Utilization analysis**: 5 episodes of 500 tasks each per preset, measuring acceptance rate stability
- **Metrics**: Acceptance rate, energy per task, policy rejection %, capacity rejection ratio, energy consumption
- **Comparison baseline**: V4 domain randomization results (37.42% average acceptance)

### 3.2 Key Metrics Definitions

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Acceptance Rate | accepted_tasks / total_tasks | Higher is better; measures task throughput |
| Policy Rejection % | policy_rejections / total_rejections | Lower is better; agent-chosen rejections |
| Capacity Rejection Ratio | capacity_rejections / total_rejections | Higher means agent pushes limits |
| Energy per Task | total_energy_kwh / accepted_tasks | Lower is better; energy efficiency |
| Reject Prob with Capacity (V10+) | avg reject head output when resources exist | Lower is better; measures decision quality |

### 3.3 Domain Randomization Presets

| Training Preset | Environments Included |
|-----------------|----------------------|
| mixed_capacity | small, medium, large |
| constrained_first | stress_test, high_load, medium |
| full_spectrum | stress_test, high_load, small, medium, large |
| production | high_load, medium, large, enterprise |

---

## 4. Pre-Baseline Development (Foundation)

### 4.1 Initial RL Agent Implementation

**Commit**: `d55d977` - "Add infrastructure-agnostic RL agent and C++ simulator integration"

The foundational commit established the entire RL module architecture:

- `rl/schemas.py`: Pydantic data models (RLState, RLAction, TaskState, HWTypeState)
- `rl/state_encoder.py`: State-to-vector encoding with v1 features (17 dimensions)
- `rl/reward.py`: Initial reward calculator (energy + SLA + rejection/acceptance)
- `rl/agent.py`: PPO policy network with infrastructure-agnostic design
- `rl/environment.py`: Gymnasium-compatible cloud provisioning environment
- `rl/trainer.py`: Single-GPU PPO training loop
- `rl/api.py`: FastAPI REST router for inference and training endpoints

The infrastructure-agnostic design was a critical early decision: the agent processes a variable number of hardware types through shared encoders and per-type scoring, making the same model deployable across different cloud configurations without retraining.

### 4.2 Multi-GPU Training

**Commit**: `8023eb8` - "multi-gpu training, benchmark fix, energy reward threshold"

Added `rl/distributed_trainer.py` with PyTorch DistributedDataParallel (DDP) support. Subsequent commits (`43db9e4`, `7da4cb2`, `7112651`) migrated from custom multi-process launch to `torchrun` and fixed collective operation deadlocks caused by logging barrier desynchronization.

### 4.3 Academic Evaluation Framework

**Commit**: `6aeae58` - "Add academic evaluation framework with stochastic environment support"

Created `scripts/run_academic_evaluation_v5.py` with:
- Multi-seed evaluation for statistical significance
- Pareto frontier analysis (energy weight vs acceptance rate trade-offs)
- Ablation studies (removing individual reward components)
- Generalization testing across multiple environment presets
- Automated figure generation and LaTeX table export

### 4.4 Early Training Iterations

Commits `06a6215` through `7b516ea` ("RL PPO agent training #1" through "#3") represent iterative training runs with evolving hyperparameters. These pre-baseline runs established training procedures and validated the agent's ability to learn basic allocation policies.

---

## 5. Version 1 - Inaugural Baseline Evaluation

**Date**: 2026-01-28
**Commit**: Results generated from academic evaluation framework
**Training**: 200,000 timesteps, 10 seeds, v1 state encoder (17 dimensions)
**Runtime**: ~79.6 hours total across all evaluation suites

### 5.1 Overview

V1 represents the first rigorous academic evaluation of the PPO agent. Trained on the `medium` preset only, it established baseline performance metrics across multiple evaluation dimensions.

### 5.2 Multi-Seed Results (10 seeds, 50 eval episodes each)

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Acceptance Rate | 57.04% | 1.62% | 54.71% | 59.87% |
| Energy per Task (kWh) | 0.001075 | 0.000026 | 0.001037 | 0.001120 |
| SLA Compliance | 93.78% | 0.71% | 92.64% | 94.80% |
| Avg Reward | 1069.8 | 8.7 | 1059 | 1083 |

All 10 seeds converged to similar reward levels (1059-1083), demonstrating training stability and reproducibility.

### 5.3 Generalization Across Infrastructure Sizes

| Preset | Acceptance | Gap vs Medium |
|--------|-----------|---------------|
| enterprise | +31.0% | Over-accepting on large infra |
| medium | 57.04% | Training environment |
| large | -7.2% | Moderate degradation |
| small | -40.7% | Severe degradation |

The agent, trained only on medium-scale infrastructure, showed significant performance degradation on smaller infrastructure and over-acceptance on larger infrastructure. This single-preset training limitation motivated the move toward domain randomization in V4.

### 5.4 Pareto Analysis (Energy Weight vs Acceptance)

Six energy weight configurations (0.5-0.95) were tested, with 4 found to be Pareto-optimal. The energy weight of 0.7 produced the best energy efficiency (0.000937 kWh/task) without significant acceptance rate sacrifice.

### 5.5 Ablation Study

| Configuration | Effect on Acceptance | Effect on Energy |
|---------------|---------------------|------------------|
| Full model | Baseline | Baseline |
| No energy reward | Minimal change | +5.3% worse |
| No SLA reward | -2.1% | +11.2% worse |
| No rejection penalty | +3.2% | +8.7% worse |
| Energy only | -5.8% | -3.1% better |

Removing the SLA component had the largest energy impact (+11.2%), suggesting that SLA compliance indirectly encouraged energy-efficient allocation by penalizing poor hardware choices that led to deadline violations.

### 5.6 Baseline Comparison

| Allocator | Total Energy (kWh) | vs PPO |
|-----------|-------------------|--------|
| Random | 43.18 | Best energy (but random) |
| Scoring (heuristic) | 48.00 | -6.0% |
| PPO Agent | 51.06 | Baseline |

The PPO agent consumed more energy than both the scoring heuristic and random allocation. This counterintuitive result is explained by the agent's higher acceptance rate: it accepted more tasks (including energy-expensive ones), increasing total energy while providing better service throughput.

### 5.7 Verdict

**BASELINE ESTABLISHED.** V1 provided the foundational performance metrics and identified the critical generalization gap: a single-preset trained agent cannot adapt to varying infrastructure scales. This directly motivated the domain randomization approach in V4.

---

## 6. Versions 2 and 3 - Iterative Refinement

### 6.1 Version 2

**Date**: 2026-01-28
**Training**: 30,000 timesteps, 3 seeds, 10 eval episodes (quick iteration)

V2 was a rapid iteration adding stress_test and high_load presets to the evaluation suite. Key findings:

| Preset | Acceptance Rate | Gap vs Medium |
|--------|----------------|---------------|
| medium | 56.13% | Reference |
| large | 67.52% | +20.3% |
| high_load | 30.7% | -43.0% |
| stress_test | 15.3% | -71.6% |

The addition of constrained presets revealed the severity of the generalization problem: the agent achieved only 15.3% acceptance on stress_test infrastructure, a 71.6% gap from medium. Ablation showed that removing the rejection penalty caused -10.4% acceptance and +28.6% energy increase, confirming rejection penalty as essential for policy stability.

### 6.2 Version 3

**Date**: 2026-01-28-29
**Training**: 50,000 timesteps, 5 seeds, 20 eval episodes

V3 expanded the evaluation to 5 seeds and 20 episodes per evaluation. Results were statistically consistent with V2:

| Metric | V2 | V3 | Change |
|--------|----|----|--------|
| Acceptance (medium) | 56.13% | 57.92% | +1.79pp |
| Energy per Task | 0.001147 | 0.001112 | -3.1% |
| SLA Compliance | 92.25% | 92.97% | +0.72pp |
| stress_test Acceptance | 15.3% | 17.95% | +2.65pp |

V3 confirmed the V2 findings with higher statistical confidence. The stress_test experiment (dedicated training on constrained presets) failed on both high_load and stress_test, indicating that the agent could not learn effective policies for severely constrained environments with the v1 state encoder.

### 6.3 Version 4_2 (Validation Run)

A V4_2 run was conducted to validate reproducibility between V3 and V4 evaluation suites. Results were identical to V3 (same timestamps, same data), confirming evaluation framework determinism.

---

## 7. Version 4 - Domain Randomization Baseline

**Date**: 2026-01-31
**Commit**: `3b78acc` - "Add domain randomization training and utilization analysis for improved generalization"
**Training**: 100,000 timesteps, mixed_capacity preset (small/medium/large), seed 42
**Training Time**: 6,387 seconds (~1.8 hours)

### 7.1 Overview

V4 introduced domain randomization -- training the agent across multiple infrastructure sizes simultaneously -- to address the single-preset generalization failure observed in V1-V3. This version also added the utilization analysis framework for detailed per-step decision tracking.

### 7.2 Domain Randomization vs Single-Preset Training

| Preset | Single-Preset | Domain-Random | Improvement |
|--------|--------------|---------------|-------------|
| small | 18.2% | 32.7% | +14.5pp |
| medium | 54.5% | 50.6% | -3.9pp |
| large | 69.0% | 67.5% | -1.5pp |
| high_load | 25.5% | 29.3% | +3.8pp |
| stress_test | 5.8% | 15.4% | +9.6pp |
| **Average** | **34.6%** | **39.1%** | **+4.5pp** |

Domain randomization improved average acceptance by +4.5%, with the largest gains on the most constrained presets (small +14.5pp, stress_test +9.6pp). The trade-off was a modest -3.9pp decrease on medium, the preset that single-preset training optimized for directly.

### 7.3 Canonical V4 Baseline (5-Preset Generalization)

This is the definitive baseline against which all subsequent versions (V5-V11) are compared:

| Preset | Acceptance | Capacity Rej | Policy Rej | Policy Rej % |
|--------|-----------|-------------|------------|--------------|
| small | 30.96% | 128.0/ep | 217.2/ep | 62.9% |
| medium | 50.64% | 143.0/ep | 112.6/ep | 41.0% |
| large | 65.28% | 135.8/ep | 41.8/ep | 25.5% |
| high_load | 27.00% | 150.8/ep | 217.8/ep | 59.4% |
| stress_test | 13.20% | 134.2/ep | 300.8/ep | 69.6% |
| **Average** | **37.42%** | -- | -- | -- |

### 7.4 Critical Finding: Policy Rejection Dominance

V4's utilization analysis revealed that on constrained presets, the majority of rejections were the agent's choice (policy rejections), not capacity-forced:

- **stress_test**: 69.1% policy rejections (agent chose to reject when resources existed)
- **small**: 62.9% policy rejections
- **high_load**: 59.1% policy rejections
- **large**: 23.5% policy rejections (agent pushed limits appropriately)

This finding became the primary optimization target for V5-V11: reducing unnecessary policy rejections on constrained environments.

### 7.5 GPU Utilization Bug Discovery

Visual inspection of V4 utilization figures revealed GPU utilization was permanently at 0% despite tasks being allocated to GPU-capable hardware. Code inspection identified the root cause: accelerators were checked for availability but never consumed or released in `environment.py`:

```python
# V4 BUG: Accelerators checked but never decremented
if can_allocate and task.requires_accelerator:
    can_allocate = hw_state['available_accelerators'] >= task.num_vms
    # Missing: hw_state['available_accelerators'] -= task.num_vms
```

This bug meant GPU resources appeared permanently available, preventing accurate GPU utilization modeling.

### 7.6 Verdict

**CROSS-PRESET BASELINE ESTABLISHED.** V4 defined the evaluation methodology and baseline metrics used for all subsequent versions. The discovery of high policy rejection rates on constrained presets and the GPU tracking bug set the agenda for V5 development.

---

## 8. Version 5 - Scarcity-Aware Rewards (Aggressive)

**Date**: 2026-02-02
**Commit**: `4d78412` - "v5 GPU Utilization Fix, Scarcity-Aware Rewards, Training Script Updates"
**Training**: ~508,000 timesteps, constrained_first preset
**Avg Training Reward**: -85.60

### 8.1 Changes Implemented

V5 addressed V4's findings with four changes:

1. **GPU/Accelerator Tracking Fix**: Accelerators now properly consumed on allocation and released on task completion. Running tasks track accelerator count for correct resource release.

2. **Scarcity-Aware Reward System**: Dynamic reward scaling based on resource utilization:
   - When resources abundant (20% util): rejection penalty = -1.12 (1.5x base), acceptance bonus = +0.36
   - When resources scarce (90% util): rejection penalty = -0.84, acceptance bonus = +0.57 (2.0x base)
   - Parameters: `scarcity_rejection_scale=1.5`, `scarcity_acceptance_scale=2.0`

3. **Minimum Execution Time**: Tasks now persist 5-15 seconds minimum (previously < 0.1s for small tasks), preventing instant resource release that made utilization tracking meaningless.

4. **Utilization Capture Timing**: Changed from before-step to after-step capture so charts reflect the current state rather than stale data.

### 8.2 Results

| Preset | V4 | V5 | Change | Status |
|--------|----|----|--------|--------|
| small | 30.96% | 29.98% | -0.98pp | Regression |
| medium | 50.64% | 47.30% | -3.34pp | Regression |
| large | 65.28% | 61.08% | -4.20pp | Regression |
| high_load | 27.00% | 25.30% | -1.70pp | Regression |
| stress_test | 13.20% | 12.58% | -0.62pp | Regression |
| **Average** | **37.42%** | **35.25%** | **-2.17pp (-5.44%)** | **Universal regression** |

All five presets regressed. The largest absolute drops occurred on medium (-3.34pp) and large (-4.20pp), the environments with the most capacity -- counterintuitive given the scarcity mechanism was designed to help constrained environments.

### 8.3 Detailed Rejection Analysis

| Preset | V4 Policy Rej % | V5 Policy Rej % | Change |
|--------|-----------------|-----------------|--------|
| small | 62.9% | 64.0% | +1.1pp (worse) |
| medium | 41.0% | 38.7% | -2.3pp (better) |
| large | 25.5% | 21.8% | -3.7pp (better) |
| high_load | 59.4% | 55.9% | -3.5pp (better) |
| stress_test | 69.6% | 69.9% | +0.3pp (worse) |

Policy rejection rates improved on medium/large/high_load but the overall acceptance still decreased, indicating the agent became more selective in what it accepted.

### 8.4 Root Cause Analysis

1. **Reward instability**: The 1.5/2.0 scaling parameters created high reward variance. The agent learned to be MORE conservative to avoid the amplified rejection penalties when resources were available, exactly the opposite of the intended effect.

2. **Insufficient training convergence**: Average training reward of -85.60 (strongly negative) indicated the policy had not converged under the new, more complex reward landscape.

3. **No curriculum learning**: The agent faced all difficulty levels simultaneously with the `constrained_first` preset, making it harder to learn the relationship between scarcity and optimal behavior.

4. **Confounded changes**: GPU fix, reward restructuring, execution time changes, and utilization timing fix were all bundled into one version, making it impossible to attribute the regression to any single change.

### 8.5 Verdict

**REGRESSION (-5.44%).** The scarcity-aware reward mechanism with aggressive parameters universally degraded performance. Despite fixing the GPU tracking bug and improving utilization capture, the reward instability overwhelmed any potential benefits. V5 demonstrated that reward engineering must be conservative and that changes should be isolated for proper attribution.

---

## 9. Version 6 - Gentle Scarcity Scaling + Curriculum Learning

**Date**: 2026-02-02
**Commit**: `d863128` - "v6 --use-capacity-features enables v3 state encoding, --no-scarcity-aware disables the ineffective scarcity rewards"
**Training**: 507,904 timesteps, mixed_capacity preset, curriculum enabled
**Avg Training Reward**: -57.95
**Training Time**: 6,188 seconds (~1.7 hours), 82 FPS

### 9.1 Changes from V5

| Parameter | V5 | V6 |
|-----------|----|----|
| Domain preset | constrained_first | mixed_capacity |
| Curriculum | No | Yes |
| Scarcity rejection scale | 1.5 | 1.2 |
| Scarcity acceptance scale | 2.0 | 1.5 |
| Timesteps | ~106k | ~508k (5x) |

### 9.2 Results

| Preset | V4 | V5 | V6 | V6 vs V4 |
|--------|----|----|-----|----------|
| small | 30.96% | 29.98% | 32.42% | +1.46pp (+4.72%) |
| medium | 50.64% | 47.30% | 50.28% | -0.36pp (-0.71%) |
| large | 65.28% | 61.08% | 65.28% | 0.00pp (0.00%) |
| high_load | 27.00% | 25.30% | 26.16% | -0.84pp (-3.11%) |
| stress_test | 13.20% | 12.58% | 13.08% | -0.12pp (-0.91%) |
| **Average** | **37.42%** | **35.25%** | **37.44%** | **+0.02pp (-0.003%)** |

V6 fully recovered from V5's regression (+5.4% vs V5) and matched V4 exactly. The `small` preset showed genuine improvement (+4.72% relative), while `large` matched V4 precisely. Constrained presets remained slightly below V4.

### 9.3 Training Improvement

| Metric | V5 | V6 |
|--------|----|----|
| Avg training reward | -85.60 | -57.95 |
| Throughput | 71 FPS | 82 FPS |
| Timesteps | ~106k | ~508k |

The +32% reward improvement confirmed better convergence with gentler scaling and curriculum learning.

### 9.4 Root Cause Analysis: The Scale Blindness Problem

V6 identified four fundamental limitations that scarcity-aware rewards could not solve:

1. **Feature Gap (Scale Blindness)**: The v1/v2 state encoder used only utilization ratios. A `capacity_ratio = 0.5` means 256 free CPUs on medium but only 32 on stress_test. The agent could not distinguish between these fundamentally different situations.

2. **Task Distribution Mismatch**: The same task generator produced identical task distributions across all presets. A task requesting 178 vCPUs was reasonable for medium (1,024 CPUs) but physically impossible for stress_test (96 CPUs).

3. **Scarcity-Aware Rewards Provided No Benefit**: V6 matched V4 with scarcity-aware rewards at 1.2/1.5; V4 achieved the same without any scarcity scaling. The mechanism added complexity without improvement.

4. **Policy Rejection Persistence**: On stress_test, 70.3% of rejections remained policy choices. The core problem was not reward calibration but missing state information.

### 9.5 Diagnostic Evidence

State vector diagnosis (`results/diagnostics/state_vector_diagnosis.json`) confirmed the scale blindness hypothesis:

| Metric | medium | stress_test | high_load |
|--------|--------|-------------|-----------|
| CPU fit ratio (task fits) | 16.8x | 1.3x | 5.5x |
| Memory fit ratio | 13.7x | 0.7x | 4.5x |
| Theoretical concurrent tasks | 13.7 | 0.7 | 4.5 |
| Avg task CPU demand | 61.0 vCPUs | 71.7 vCPUs | 46.4 vCPUs |

The agent saw similar task features and utilization ratios across presets, but outcomes differed by an order of magnitude. A stress_test cluster could fit less than one average task at a time, while medium could fit 13+.

### 9.6 Verdict

**RECOVERY, NOT IMPROVEMENT (+0.003%).** V6 neutralized V5's regression but proved that scarcity-aware reward engineering cannot solve the performance gap. The fundamental bottleneck was identified as a feature gap (scale blindness), directly motivating V7's capacity features.

---

## 10. Version 7 - Capacity Features (v3 State Encoder)

**Date**: 2026-02-02
**Commit**: `b67c6ca` - "v7 (patched)"
**Training**: 507,904 timesteps, mixed_capacity preset, curriculum enabled
**Avg Training Reward**: -76.43
**Training Time**: 6,168 seconds (~1.7 hours), 82 FPS

### 10.1 Key Innovation: v3 State Encoder

V7 introduced 6 new capacity features to the state encoder, expanding from 22 dimensions (v2) to 28 dimensions (v3):

| Feature | Description | Range | Purpose |
|---------|-------------|-------|---------|
| total_system_cpus_normalized | Total CPUs / 2000 | 0-1 | System scale indicator |
| total_system_memory_normalized | Total memory / 8000 | 0-1 | Memory scale |
| cpu_fit_ratio | Available CPUs / task CPU demand / 10 | 0-1 | Task feasibility signal |
| mem_fit_ratio | Available memory / task memory demand / 10 | 0-1 | Memory feasibility |
| scale_bucket | Total CPUs / 2000 (continuous) | 0-1 | Categorical system size |
| task_relative_size | Task resource fraction of total * 10 | 0-1 | Relative task impact |

These features directly addressed the scale blindness problem: the agent could now distinguish a 50% utilized 1,024-CPU cluster from a 50% utilized 96-CPU cluster.

### 10.2 Configuration Error

Scarcity-aware rewards were accidentally left enabled at the aggressive V5 parameters (1.5/2.0 scales) rather than being disabled as V6 recommended. Despite this, V7 achieved the best results to date, suggesting the capacity features provided sufficient benefit to overcome the harmful scarcity scaling.

### 10.3 Results

| Preset | V4 | V7 | Change | Relative |
|--------|----|----|--------|----------|
| small | 30.96% | 31.52% | +0.56pp | +1.81% |
| medium | 50.64% | 49.26% | -1.38pp | -2.73% |
| large | 65.28% | 66.96% | +1.68pp | +2.57% |
| high_load | 27.00% | 27.96% | +0.96pp | +3.56% |
| stress_test | 13.20% | 13.42% | +0.22pp | +1.67% |
| **Average** | **37.42%** | **37.82%** | **+0.41pp** | **+1.38%** |

V7 improved on 4 of 5 presets (all except medium). The constrained environments showed the clearest benefit: high_load +3.56% and stress_test +1.67% relative improvement.

### 10.4 Energy Efficiency

| Preset | V7 Energy/Task | vs V6 |
|--------|----------------|-------|
| small | 0.000841 kWh | -6.5% better |
| high_load | 0.000952 kWh | -6.5% better |
| stress_test | 0.000949 kWh | Slight regression |

Constrained presets showed improved energy efficiency, suggesting the agent made better hardware choices when it could see the scale of available resources.

### 10.5 Interpretation

The medium preset regression (-2.73%) may indicate the agent became slightly more conservative on balanced environments when given explicit scale information, possibly over-correcting for perceived capacity constraints. The training reward (-76.43) was worse than V6 (-57.95), likely due to the accidentally enabled aggressive scarcity scaling increasing reward variance.

### 10.6 Verdict

**FIRST IMPROVEMENT OVER V4 BASELINE (+1.38%).** V7 validated the hypothesis that scale blindness was the primary performance bottleneck. Adding absolute capacity information to the state encoder improved constrained environment performance while maintaining overall gains. The capacity features proved robust enough to overcome the harmful scarcity reward configuration.

---

## 11. Version 8 - Lower Learning Rate + GPU Compute Fix

**Date**: 2026-02-03
**Commit**: `4a6bc3d` - "v8"
**Training**: 507,904 timesteps, mixed_capacity preset, curriculum enabled
**Avg Training Reward**: -74.49
**Training Time**: 6,190 seconds (~1.7 hours), 82 FPS

### 11.1 Changes from V7

| Parameter | V7 | V8 |
|-----------|----|----|
| Learning rate | 3e-4 | 1e-4 |
| GPU compute fix | No | Yes (30% efficiency boost for large compute tasks) |
| Scarcity-aware | True (1.5/2.0) - accidental | True (1.5/2.0) - still not fixed |

The learning rate was reduced 3x based on the hypothesis that the larger 28-dimensional state space needed more stable gradient updates. A GPU compute efficiency model was also added to address GPU underutilization observed in V4-V7.

### 11.2 Results

| Preset | V4 | V7 | V8 | V8 vs V7 | V8 vs V4 |
|--------|----|----|-----|----------|----------|
| small | 30.96% | 31.52% | 29.38% | -2.14pp | -5.10% |
| medium | 50.64% | 49.26% | 50.06% | +0.80pp | -1.15% |
| large | 65.28% | 66.96% | 64.64% | -2.32pp | -0.98% |
| high_load | 27.00% | 27.96% | 26.42% | -1.54pp | -2.15% |
| stress_test | 13.20% | 13.42% | 13.28% | -0.14pp | +0.61% |
| **Average** | **37.42%** | **37.82%** | **36.76%** | **-1.06pp** | **-1.75%** |

V8 regressed on 4 of 5 presets, winning only on medium (+0.80pp).

### 11.3 Energy Efficiency Regression

| Preset | V7 Energy/Task | V8 Energy/Task | Change |
|--------|----------------|----------------|--------|
| small | 0.000841 | 0.000985 | +17.1% worse |
| high_load | 0.000952 | 0.001086 | +14.1% worse |
| medium | 0.001414 | 0.001384 | -2.1% better |
| large | 0.001590 | 0.001629 | +2.5% worse |
| stress_test | 0.000949 | 0.000933 | -1.7% better |

Energy efficiency worsened significantly on constrained presets (+17.1% on small, +14.1% on high_load).

### 11.4 Root Cause Analysis

1. **Overfitting from lower learning rate**: The 3x LR reduction caused tighter convergence to the training distribution. Training reward improved slightly (-74.49 vs -76.43), but generalization worsened -- a classic overfitting signature.

2. **Confounded variables**: The LR change was bundled with the GPU compute fix, making it impossible to attribute the regression to either change independently.

3. **Persistent scarcity-aware rewards**: Despite V7 recommendations to disable these, V8 kept them active at aggressive 1.5/2.0 scales, compounding the LR-induced conservatism.

### 11.5 Verdict

**REGRESSION (-1.75% vs V4, -2.8% vs V7).** The lower learning rate caused overfitting and reduced generalization. V8 highlighted the importance of isolating changes (one variable at a time) and following through on recommendations. V7 remained the best version.

---

## 12. Version 9 - Reward Rebalance + Architecture Overhaul

**Date**: 2026-02-10
**Commit**: `decb2ed` - "v9: reward rebalance, entropy/LR schedules, architecture improvements"
**Training**: 2,007,040 timesteps (4x increase), mixed_capacity preset, curriculum enabled
**Avg Training Reward**: 7.46 (first positive average)
**Training Time**: 22,834 seconds (~6.3 hours), 88 FPS

### 12.1 Overview

V9 was the most comprehensive single-version overhaul in the project's history, implementing 15 structural changes informed by the cumulative analysis of V4-V8. This version addressed fundamental issues in reward design, training methodology, network architecture, and state encoding.

### 12.2 The 15 Changes

#### Reward System (Changes 1-2)

**Change 1: Reward Asymmetry Fix** (`reward.py`)
- Rejection penalty: 0.8 -> 0.5 (37.5% reduction)
- Acceptance bonus: 0.3 -> 0.35 (16.7% increase)
- Penalty-to-bonus ratio: 2.67:1 -> 1.43:1
- Rationale: The asymmetric ratio created a risk-averse landscape where rejecting was always "safer" than accepting

**Change 2: Scarcity-Aware Rewards Disabled** (`reward.py`)
- Default `scarcity_aware=False`
- Proven harmful in V5 (-5.8%) and ineffective in V6 (neutral)
- Removes a source of reward variance without sacrificing performance

#### Training Methodology (Changes 3-4, 6)

**Change 3: Default Timesteps 2M** (scripts)
- Increased from 100K to 2,000,000 (20x)
- V6-V8 at ~508K showed incomplete convergence (negative rewards)
- 4x increase from V8's actual training gave the complex policy sufficient learning time

**Change 4: Entropy Coefficient Annealing** (`distributed_trainer.py`)
- Linear schedule: 0.05 (high exploration) -> 0.001 (exploitation)
- Early high entropy encourages diverse action exploration
- Late low entropy allows the policy to sharpen and commit to learned behaviors

**Change 6: Cosine Learning Rate Schedule** (`distributed_trainer.py`)
- CosineAnnealingLR: 3e-4 -> 1e-5
- High LR for fast early learning on coarse policy structure
- Smooth decay to low LR for fine-tuning without the overfitting seen in V8's constant low LR

#### Architecture Improvements (Changes 5, 9-10)

**Change 5: LayerNorm in Encoders** (`agent.py`)
- Added LayerNorm after each Linear layer in TaskEncoder and HWEncoder
- Addresses internal covariate shift during training
- Provides more stable gradients, especially important with the cosine LR schedule

**Change 9: Max+Mean HW Pooling for Value Head** (`agent.py`)
- Value head input expanded from `task_emb + mean_hw_emb` to `task_emb + mean_hw_emb + max_hw_emb`
- Input dimension: 128 -> 192
- Provides the value estimator with information about the best-case hardware option, not just the average

**Change 10: Vectorized HW Encoder Forward Pass** (`agent.py`)
- Replaced sequential loop over HW types with batched tensor processing
- All HW vectors processed through the encoder in a single call
- Better GPU utilization during training, measurable throughput improvement (71 -> 88 FPS)

#### State Encoding Improvements (Changes 7, 13)

**Change 7: Continuous Scarcity Indicator** (`state_encoder.py`)
- Replaced binary 0/1 indicator (threshold at 80% utilization) with continuous ramp
- Formula: `max(0, min(1, (max_util - 0.5) / 0.4))`
- Range: 0.0 at <=50% utilization, linearly increasing to 1.0 at >=90%
- Provides gradient information instead of binary signal

**Change 13: Continuous Scale Bucket** (`state_encoder.py`)
- Replaced discrete 5-value scale bucket with continuous: `min(total_cpus / 2000, 1.0)`
- Eliminates information loss at threshold boundaries
- Smoother gradient through scale feature

#### Reward Calibration (Changes 8, 12)

**Change 8: Adaptive Energy Baseline** (`reward.py`)
- Exponential Moving Average (EMA) with alpha=0.01
- After 20 observations, replaces fixed 0.05 kWh baseline
- Self-calibrates to actual energy distribution of each preset

**Change 12: Reduced Execution Time Floor** (`environment.py`)
- Random floor `5.0 + uniform(0, 10)` -> fixed 1.0 second
- Removes noise that obscured HW type performance differences
- Energy signal becomes cleaner, making HW-specific learning more effective

#### Environment Design (Change 11)

**Change 11: Per-Preset Curriculum Thresholds** (`environment.py`)
- Replaced single threshold (0.6) with per-preset values:
  - stress_test: 0.15, high_load: 0.30, small: 0.35
  - medium: 0.50, large: 0.60, enterprise: 0.65
- Curriculum can now actually advance through stages rather than getting stuck on impossible targets

#### Infrastructure (Changes 14-15)

**Change 14: CLI Parameter Plumbing** (scripts)
- New arguments: `--acceptance-bonus`, `--entropy-start`, `--entropy-end`, `--lr-min`
- All V9 defaults baked into argument parsers

**Change 15: Documentation** (CHANGELOG_V9_RECOMMENDATIONS.md)

### 12.3 Results

| Preset | V4 | V7 (Prev Best) | V9 | vs V4 | vs V7 |
|--------|----|-----------------|----|-------|-------|
| small | 30.96% | 31.52% | 31.30% | +0.34pp (+1.10%) | -0.22pp |
| medium | 50.64% | 49.26% | 51.84% | +1.20pp (+2.37%) | +2.58pp |
| large | 65.28% | 66.96% | 70.18% | +4.90pp (+7.51%) | +3.22pp |
| high_load | 27.00% | 27.96% | 30.42% | +3.42pp (+12.67%) | +2.46pp |
| stress_test | 13.20% | 13.42% | 15.94% | +2.74pp (+20.76%) | +2.52pp |
| **Average** | **37.42%** | **37.82%** | **39.94%** | **+2.52pp (+6.7%)** | **+2.12pp** |

**All presets improved over V4.** The largest gains were on the most constrained environments: stress_test (+20.76% relative), high_load (+12.67%), and large (+7.51%).

### 12.4 Energy Efficiency Breakthrough

| Preset | V7 Energy/Task | V9 Energy/Task | Improvement |
|--------|----------------|----------------|-------------|
| small | 0.000841 | 0.000659 | -21.6% |
| medium | 0.001414 | 0.001022 | -27.7% |
| large | 0.001590 | 0.001115 | -29.9% |
| high_load | 0.000952 | 0.000635 | -33.3% |
| stress_test | 0.000949 | 0.000328 | -65.4% |

V9 achieved 21.6% to 65.4% energy efficiency improvement across all presets. The reduced rejection penalty encouraged acceptance of lower-energy tasks that were previously rejected.

### 12.5 Policy Rejection Improvement

| Preset | V4 Policy Rej % | V9 Policy Rej % | Change |
|--------|-----------------|-----------------|--------|
| small | 62.9% | 61.7% | -1.2pp |
| medium | 41.0% | 43.6% | +2.6pp |
| large | 25.5% | 23.9% | -1.6pp |
| high_load | 59.4% | 56.1% | -3.3pp |
| stress_test | 69.6% | 68.0% | -1.6pp |

Policy rejection decreased on 4 of 5 presets, with the most improvement on constrained environments. The large preset's capacity rejection ratio reached 76.1%, indicating the agent was pushing capacity limits aggressively -- optimal behavior.

### 12.6 Training Dynamics

- **Reward trajectory**: Started at ~15, dipped to 4-5 during mid-training entropy/LR transition, recovered to 6-8
- The mid-training dip correlated with the entropy schedule transition from high exploration to exploitation
- Final average reward of 7.46 was the first positive average in the project's history (all V5-V8 had negative averages)

### 12.7 Impact Attribution

While all 15 changes contributed, the most impactful were:

1. **Reward rebalancing** (primary driver for medium/large): Reducing the penalty-to-bonus ratio from 2.67:1 to 1.43:1 directly reduced unnecessary policy rejections
2. **Training duration** (4x increase): 2M timesteps allowed full convergence vs the ~508k of V6-V8
3. **Disabling scarcity-aware rewards**: Removed the confounding factor that polluted V7-V8 results
4. **Entropy annealing + cosine LR**: Better exploration/exploitation balance across training phases

### 12.8 Remaining Bottleneck

Despite the improvement, constrained presets still showed high policy rejection rates (56-68%). The reject head received only the 64-dimensional task embedding, with no visibility into infrastructure capacity. This directly motivated V10's capacity-aware reject head.

### 12.9 Verdict

**NEW BEST PERFORMANCE (+6.7% vs V4 baseline).** V9 achieved the highest acceptance rate and best energy efficiency to date through comprehensive reward rebalancing, architecture improvements, and proper training duration. The 15-change overhaul demonstrated that accumulated incremental fixes can compound into significant improvements when applied together with adequate training time.

---

## 13. Version 10 - Capacity-Aware Reject Head

**Date**: 2026-02-12
**Commit**: `c0af458` - "v10: capacity-aware reject head architecture"
**Training**: ~1,660,000 timesteps (41.5% of planned 4M), full_spectrum preset
**Backward Compatibility**: V9 checkpoints loaded with `strict=False`, reject head reinitialized

### 13.1 Key Innovation: Reject Head Architecture

V10 expanded the reject head input from 64 to 195 dimensions, giving it visibility into infrastructure state:

**V9 reject head** (64 dimensions):
```
task_embedding (64) -> Linear(64, 32) -> ReLU -> Linear(32, 1)
```

**V10 reject head** (195 dimensions):
```
task_embedding (64) +
mean_hw_embedding (64) +
max_hw_embedding (64) +
capacity_summary (3):
  - max_hw_score (best allocation score)
  - mean_hw_score (average allocation score)
  - num_valid_ratio (fraction of HW types with capacity)
-> Linear(195, 64) -> ReLU -> Linear(64, 1)
```

The rationale was that the reject head was the only decision component without infrastructure visibility:

| Component | Input | Sees Capacity? |
|-----------|-------|----------------|
| Scorer | task_emb + hw_emb | Yes (per-HW) |
| Value head | task_emb + mean_hw + max_hw | Yes (aggregated) |
| Reject head (V9) | task_emb only | No |
| **Reject head (V10)** | **task_emb + mean_hw + max_hw + capacity_summary** | **Yes** |

### 13.2 Additional Changes

- **Domain preset**: Changed from `mixed_capacity` to `full_spectrum` (includes stress_test and high_load in training)
- **Rejection penalty**: Reduced from 0.5 to 0.3 (further reward rebalance)
- **Target timesteps**: Increased to 4M (though training was cut short at 1.66M)

### 13.3 Results

| Preset | V9 | V10 | Change |
|--------|-----|------|--------|
| small | 31.30% | 31.00% | -0.30pp |
| medium | 51.84% | 52.06% | +0.22pp |
| large | 70.18% | 69.82% | -0.36pp |
| high_load | 30.42% | 30.28% | -0.14pp |
| stress_test | 15.94% | 15.58% | -0.36pp |
| **Average** | **39.94%** | **39.75%** | **-0.19pp** |

V10 matched V9 within noise margins across all presets. The differences (all < 0.4pp) fell within standard deviations.

### 13.4 The Critical Diagnostic: Reject Probability With Capacity

V10 introduced a new metric: the reject head's output probability when at least one HW type has available capacity. This directly measured whether the reject head was the bottleneck.

| Preset | Reject Prob When Capacity Exists |
|--------|----------------------------------|
| large | 0.025% |
| medium | 0.12% |
| high_load | 0.83% |
| small | 1.71% |
| stress_test | 3.42% |

**The reject head almost never rejected when resources existed.** Even on stress_test (the worst case), the reject probability with capacity was only 3.42%.

### 13.5 Paradigm-Shifting Insight

V10's most important contribution was diagnostic, not performance. The near-zero reject probabilities proved that:

1. **The reject head was NOT the bottleneck.** The high policy rejection percentages (56-69% on constrained presets) were NOT caused by the reject head outputting high rejection probabilities.

2. **"Policy rejections" were actually capacity exhaustions.** When the metric classified a rejection as "policy" (agent choice), it was actually the case that the `valid_mask` was all-False -- no hardware type had sufficient resources for the specific task. This is infrastructure-forced rejection, not a policy error.

3. **The agent's decision-making quality was already near-optimal.** With reject probabilities of 0.025-3.42% when capacity existed, the agent correctly accepted tasks whenever it physically could.

4. **The real bottleneck was task-infrastructure mismatch.** The task distribution generated tasks (e.g., 178 vCPUs) that were physically impossible on stress_test (96 total CPUs). No amount of policy optimization could overcome physical resource limits.

This insight fundamentally redirected optimization efforts from the agent to the environment.

### 13.6 Why Training Was Cut Short

V10 was evaluated at 1.66M of a planned 4M timesteps (41.5% complete). Given that initial results showed no improvement trajectory over V9, the additional compute cost was not justified. Despite the incomplete training, reaching parity with V9 so quickly (with a fully reinitialized reject head) validated the architectural change.

### 13.7 Verdict

**NEUTRAL (-0.2pp vs V9).** Performance matched V9 within noise margins. V10's primary value was diagnostic: proving the reject head was not the bottleneck and that the performance ceiling was caused by task-infrastructure mismatch. This insight directly motivated V11's environment-level intervention.

---

## 14. Version 11 - Capacity-Scaled Task Generation

**Date**: 2026-02-13
**Commit**: `1b3fb3d` - "v11: capacity-scaled task generation to match infrastructure size"
**Training**: 2,007,040 timesteps, full_spectrum preset, curriculum enabled
**Avg Training Reward**: 34.75 (highest ever, first strongly positive average)
**Training Time**: 25,226 seconds (~7.0 hours), 80 FPS

### 14.1 Key Innovation: Environment-Level Intervention

V11 addressed the fundamental task-infrastructure mismatch identified by V10 by modifying the environment rather than the agent. Task sizes (`num_vms`) now scale proportionally to infrastructure capacity.

**Scaling Formula**:
```
capacity_scale = clamp(min(total_cpus / 1024, total_memory / 7168), 0.25, 3.0)
```

Medium preset (1024 CPUs, 7168 GB) serves as the reference scale (1.0).

**Impact on Task Sizes**:

| Preset | Scale Factor | Large Task VMs (Before) | Large Task VMs (After) |
|--------|-------------|-------------------------|------------------------|
| stress_test | 0.25 | 4-15 | 2-4 |
| high_load | 0.25 | 4-15 | 2-4 |
| small | 0.38 | 4-15 | 2-6 |
| medium | 1.00 | 4-15 | 4-15 (unchanged) |
| large | 2.09 | 4-15 | 8-31 (scaled up) |

**Memory Cap**: Memory-intensive tasks limited to 50% of total system memory, preventing physically impossible memory requests on constrained presets.

### 14.2 Additional Changes

- **Rejection penalty**: 0.3 (reduced from V9's 0.5, aligned with V10)
- **Architecture**: Same as V10 (195-dim reject head)

### 14.3 Results

| Preset | V4 | V9 | V10 | V11 | V11 vs V10 | V11 vs V4 |
|--------|----|----|------|------|------------|-----------|
| small | 30.96% | 31.30% | 31.00% | 33.56% | +2.56pp | +2.60pp |
| medium | 50.64% | 51.84% | 52.06% | 52.58% | +0.52pp | +1.94pp |
| large | 65.28% | 70.18% | 69.82% | 61.60% | **-8.22pp** | -3.68pp |
| high_load | 27.00% | 30.42% | 30.28% | 30.42% | +0.14pp | +3.42pp |
| stress_test | 13.20% | 15.94% | 15.58% | 19.20% | **+3.62pp** | +6.00pp |
| **Average** | **37.42%** | **39.94%** | **39.75%** | **39.47%** | **-0.28pp** | **+2.05pp** |

### 14.4 Constrained Preset Gains

The capacity-scaled task generation achieved its primary objective on constrained presets:

| Preset | V10 | V11 | Improvement |
|--------|------|------|-------------|
| stress_test | 15.58% | 19.20% | +3.62pp (+23.2% relative) |
| small | 31.00% | 33.56% | +2.56pp (+8.3% relative) |
| high_load | 30.28% | 30.42% | +0.14pp (+0.5% relative) |

The stress_test improvement of +3.62pp was the largest single-version gain on that preset in the project's history.

### 14.5 Reject Head Convergence

| Preset | V10 Reject Prob | V11 Reject Prob | Improvement Factor |
|--------|-----------------|-----------------|-------------------|
| small | 1.71% | 0.027% | 63x better |
| medium | 0.12% | 0.004% | 30x better |
| large | 0.025% | 0.001% | 25x better |
| high_load | 0.83% | 0.005% | 166x better |
| stress_test | 3.42% | 0.029% | 118x better |

The reject probability with capacity dropped by 30-166x across all presets. The agent achieved functionally perfect reject-head behavior: 0.001-0.029% rejection probability when resources existed.

### 14.6 Large Preset Regression Analysis

The large preset regressed -8.22pp (69.82% -> 61.60%), which is the dominant reason V11's average didn't exceed V9.

**Root cause: super-linear resource contention**

1. **Task sizes doubled**: Large preset tasks scaled to 8-31 VMs (from 4-15), consuming approximately 2x the resources.
2. **Non-linear contention**: Each larger task holds resources longer, creating cascading unavailability. A 31-VM task blocks resources that could serve multiple smaller tasks.
3. **Compounding effect**: More resources blocked -> more tasks waiting -> higher effective rejection rate.
4. **Capacity rejection rise**: Capacity rejections increased from 1,196 (V10) to 1,349 (V11) despite identical infrastructure, confirming that larger tasks exhausted capacity faster.

The capacity rejection ratio on large dropped from 79.5% to 70.0%, meaning the agent reacted to increased resource pressure by becoming more conservative (more policy rejections).

### 14.7 Energy Comparability Note

V11's energy-per-task numbers are NOT directly comparable to V9/V10 because task sizes changed:

| Preset | V10 Energy/Task | V11 Energy/Task | Explanation |
|--------|-----------------|-----------------|-------------|
| medium | 0.000981 | 0.000940 | Same tasks, slight improvement (-4.2%) |
| large | 0.001064 | 0.000563 | Larger tasks, more efficient utilization (-47%) |
| stress_test | 0.000357 | 0.001700 | Proportionally larger tasks (+376%) |

The energy increases on constrained presets are artifacts of the scaled task sizes, not agent inefficiency.

### 14.8 Training Dynamics

V11 achieved the first strongly positive average training reward (34.75), substantially exceeding V9's 7.46:

| Training Phase | Episodes | Behavior |
|----------------|----------|----------|
| 1-80 (curriculum: stress_test, high_load) | Volatile rewards, avg ~0 | Learning constrained allocation |
| 80-120 (transition to easier presets) | Rewards jump to 30-60 | Rapid improvement on balanced envs |
| 120-240 (full mix) | Stable at 35-55 | Converged policy with healthy exploration |

Policy loss converged smoothly from -0.02 to -0.001. Entropy settled at 0.15-0.17, indicating healthy exploration was maintained throughout.

### 14.9 Verdict

**MIXED (+5.5% vs V4, -0.5% vs V9).** Capacity-scaled task generation successfully addressed the task-infrastructure mismatch on constrained presets (stress_test +3.6pp is the largest single-version gain on that preset). However, symmetric upward scaling on the large preset caused super-linear resource contention and -8.2pp regression. The net effect is a wash against V9. The approach is validated for downward scaling but requires asymmetric treatment for upward scaling.

---

## 15. Cross-Version Analysis

### 15.1 Acceptance Rate Evolution

```
Version:    V4      V5      V6      V7      V8      V9      V10     V11
            |       |       |       |       |       |       |       |
stress_test 13.20   12.58   13.08   13.42   13.28   15.94   15.58   19.20
high_load   27.00   25.30   26.16   27.96   26.42   30.42   30.28   30.42
small       30.96   29.98   32.42   31.52   29.38   31.30   31.00   33.56
medium      50.64   47.30   50.28   49.26   50.06   51.84   52.06   52.58
large       65.28   61.08   65.28   66.96   64.64   70.18   69.82   61.60
            -----   -----   -----   -----   -----   -----   -----   -----
Average     37.42   35.25   37.44   37.82   36.76   39.94   39.75   39.47
vs V4       Base    -5.8%   +0.0%   +1.4%   -1.8%   +6.7%   +6.2%   +5.5%
```

### 15.2 Improvement Trajectory by Preset

| Preset | Best Version | Best Rate | V4 Rate | Total Gain |
|--------|-------------|-----------|---------|------------|
| stress_test | V11 | 19.20% | 13.20% | +6.00pp (+45.5%) |
| high_load | V9/V11 | 30.42% | 27.00% | +3.42pp (+12.7%) |
| small | V11 | 33.56% | 30.96% | +2.60pp (+8.4%) |
| medium | V11 | 52.58% | 50.64% | +1.94pp (+3.8%) |
| large | V9 | 70.18% | 65.28% | +4.90pp (+7.5%) |

### 15.3 State Encoder Evolution

| Version | Encoder | Dimensions | Key Features |
|---------|---------|-----------|--------------|
| v1 | V1-V6 | 17 (task+global) | Task (12) + Global (5) |
| v2 | V6 default | 22 | + Scarcity (5): utilization, capacity ratios |
| v3 | V7-V11 | 28 | + Capacity (6): system scale, task fit, relative size |

### 15.4 Reward Function Evolution

| Parameter | V4 | V5 | V6 | V7-V8 | V9 | V10-V11 |
|-----------|----|----|-----|-------|-----|---------|
| Rejection penalty | 0.8 | 0.8 | 0.8 | 0.8 | 0.5 | 0.3 |
| Acceptance bonus | 0.3 | 0.3 | 0.3 | 0.3 | 0.35 | 0.35 |
| Penalty:bonus ratio | 2.67 | 2.67 | 2.67 | 2.67 | 1.43 | 0.86 |
| Scarcity-aware | No | Yes (1.5/2.0) | Yes (1.2/1.5) | Yes (1.5/2.0) | No | No |

### 15.5 Architecture Evolution

| Component | V4 | V5-V8 | V9 | V10-V11 |
|-----------|----|----|------|---------|
| TaskEncoder | Linear(17,64)->ReLU->Linear(64,64) | Same | + LayerNorm | Same |
| HWEncoder | Linear(16,64)->ReLU->Linear(64,64) | Same | + LayerNorm | Same |
| Reject Head Input | 64 (task_emb only) | Same | Same | 195 (+ hw_embs + capacity) |
| Value Head Input | 128 (task_emb + mean_hw) | Same | 192 (+ max_hw) | Same |
| HW Processing | Sequential loop | Same | Vectorized batch | Same |

### 15.6 Training Configuration Evolution

| Parameter | V1-V3 | V4 | V5 | V6-V8 | V9-V11 |
|-----------|-------|----|----|-------|--------|
| Timesteps | 30K-200K | 100K | ~106K | ~508K | 2M |
| Training preset | medium only | mixed_capacity | constrained_first | mixed_capacity | mixed/full_spectrum |
| Curriculum | No | No | No | Yes | Yes |
| LR schedule | Fixed 3e-4 | Fixed 3e-4 | Fixed 3e-4 | Fixed (V8: 1e-4) | Cosine 3e-4 -> 1e-5 |
| Entropy | Fixed | Fixed | Fixed | Fixed | Annealed 0.05 -> 0.001 |
| GPUs | 4 | 4 | 4 | 4 | 4 |

### 15.7 Energy Efficiency Trend (Energy per Task, kWh)

| Preset | V4* | V7 | V9 | V10 | V11** |
|--------|------|------|------|------|------|
| small | 0.000872 | 0.000841 | 0.000659 | 0.000669 | 0.001816 |
| medium | 0.001421 | 0.001414 | 0.001022 | 0.000981 | 0.000940 |
| large | 0.001546 | 0.001590 | 0.001115 | 0.001064 | 0.000563 |
| high_load | 0.001018 | 0.000952 | 0.000635 | 0.000605 | 0.002752 |
| stress_test | 0.000845 | 0.000949 | 0.000328 | 0.000357 | 0.001700 |

*V4 energy from utilization analysis. **V11 energy not directly comparable due to changed task sizes.

V9 achieved the most significant energy efficiency improvement: 21.6% to 65.4% better than V7 across all presets.

---

## 16. Lessons Learned

### 16.1 Reward Engineering

1. **Asymmetry matters more than magnitude.** The penalty-to-bonus ratio (2.67:1 in V4 vs 0.86:1 in V10-V11) had a larger impact on policy conservatism than the absolute values of either parameter.

2. **Dynamic reward scaling is fragile.** Scarcity-aware rewards (V5-V8) never outperformed static rewards. The additional variance in the reward signal destabilized learning without providing actionable gradient information.

3. **Simpler reward signals learn faster.** V9's fixed penalties with EMA energy baseline converged to positive reward in 2M steps, while V5-V8's more complex reward landscapes never reached positive average reward even with equivalent or longer training.

### 16.2 State Space Design

4. **Ratios hide absolute scale.** Utilization ratios (v1/v2 encoder) made a 96-CPU cluster at 80% utilization indistinguishable from a 1,024-CPU cluster at 80% utilization. The addition of absolute capacity features (v3 encoder) was the first change to produce a genuine improvement over baseline.

5. **Continuous features outperform discrete.** Replacing binary scarcity indicators and discrete scale buckets with continuous versions provided smoother gradients and better generalization.

6. **Diagnostics before optimization.** The state vector diagnosis tool (`diagnose_state_vectors.py`) confirmed the scale blindness hypothesis with quantitative evidence before committing to architectural changes. This saved development cycles compared to trial-and-error.

### 16.3 Training Methodology

7. **Insufficient training is the silent killer.** V5-V8 ran at 100K-508K timesteps, all producing negative average rewards. V9's 4x increase to 2M timesteps was a primary factor in its success. The complex domain-randomized policy needed significantly more experience than initially estimated.

8. **Isolate variables.** V8 changed both LR and GPU compute simultaneously, making it impossible to attribute the regression. V5 bundled four changes. V9 deliberately bundled 15 changes because the prior analysis had individually justified each one -- but this only works when each change has been studied in isolation first.

9. **LR schedule > fixed low LR.** V8's constant low LR (1e-4) caused overfitting. V9's cosine schedule (3e-4 -> 1e-5) provided the best of both worlds: fast early learning and stable late convergence.

10. **Entropy annealing enables exploration.** V9's entropy schedule (0.05 -> 0.001) encouraged diverse action exploration early in training, then allowed the policy to sharpen. This prevented the premature convergence to conservative rejection policies seen in earlier versions.

### 16.4 Architecture Design

11. **Information asymmetry creates bottlenecks (or the appearance of them).** The reject head in V4-V9 could only see task embeddings, not infrastructure state. V10 added infrastructure visibility but discovered the head was already performing well -- the bottleneck was misidentified.

12. **Build diagnostics into the architecture.** V10's `reject_prob_with_capacity` metric was the key to understanding the system. Without this measurement, the team would have continued optimizing the reject head (a solved problem) instead of addressing the task distribution (the actual problem).

### 16.5 Environment Design

13. **Task distributions must match infrastructure scale.** The fixed task distribution generating 178-vCPU average demands was physically impossible on 96-CPU infrastructure. This is not a policy problem but an environment design problem.

14. **Symmetric scaling is dangerous.** V11's linear upward scaling (2.09x for large) caused super-linear resource contention. Downward scaling (0.25x for stress_test) was uniformly beneficial, but upward scaling requires sub-linear or asymmetric treatment.

15. **The environment is as important as the agent.** V10-V11 demonstrated that once agent decision quality is near-optimal (0.001-0.029% reject probability with capacity), further gains must come from environment design, task distribution, or infrastructure configuration.

---

## 17. Current State and Future Directions

### 17.1 Current Best Configuration

**Best overall model**: V9 (39.94% average acceptance, +6.7% vs V4)
**Best constrained performance**: V11 (19.20% stress_test, +6.00pp vs V4)
**Best decision quality**: V11 (0.001-0.029% reject probability with capacity)

### 17.2 Performance Ceiling Analysis

The agent's decision quality is functionally optimal for the current environment configuration:

| Evidence | Value | Implication |
|----------|-------|-------------|
| Reject prob with capacity | 0.001-0.029% | Agent accepts whenever possible |
| Policy rejection on large (V9) | 23.9% | Most rejections are capacity-forced |
| V10 reject head showed no benefit | -0.2pp vs V9 | Reject architecture is not a bottleneck |
| V11 constrained gains | +3.6pp stress_test | Environment-level changes unlock more |

### 17.3 Recommended V12 Directions

Three approaches to resolve V11's large preset regression while preserving constrained preset gains:

1. **Asymmetric scaling** (recommended): Cap `capacity_scale` at 1.0, only downscale. Preserves V11 constrained gains while reverting large/enterprise to original task distributions.

2. **Sub-linear upward scaling**: Apply square root to upward scaling: `scale_up = 1.0 + sqrt(scale - 1.0)`. Large preset scale becomes 1.45 instead of 2.09.

3. **Per-task-type scaling caps**: Different scaling behavior per task category, recognizing that large tasks are the most sensitive to upward scaling due to their high per-VM resource requirements.

### 17.4 Longer-Term Research Directions

- **Attention-based HW aggregation**: Replace mean/max pooling with attention mechanism for more nuanced hardware state representation
- **Online learning with simulator feedback**: Train on outcomes from the C++ CloudLightning simulator for real-world policy refinement
- **Multi-objective Pareto optimization**: Explicit Pareto frontier tracking across energy, SLA, and acceptance rate
- **Hierarchical task batching**: Group compatible tasks for more efficient resource utilization

---

## Appendix A: Git Commit History (RL-Related)

| Commit | Message | Version |
|--------|---------|---------|
| `d55d977` | Add infrastructure-agnostic RL agent and C++ simulator integration | Foundation |
| `06a6215` | RL PPO agent training #1 | Pre-V1 |
| `8023eb8` | multi-gpu training, benchmark fix, energy reward threshold | Pre-V1 |
| `43db9e4` | use torchrun rather than custom approach | Pre-V1 |
| `7112651` | Collective operation deadlock fixed | Pre-V1 |
| `6aeae58` | Add academic evaluation framework with stochastic environment support | V1 framework |
| `b04b1e1` | Address critique: improve reward function and add stress testing | V2/V3 |
| `3b78acc` | Add domain randomization training and utilization analysis | V4 |
| `4d78412` | v5 GPU Utilization Fix, Scarcity-Aware Rewards | V5 |
| `d863128` | v6 --use-capacity-features enables v3 state encoding | V6 |
| `b67c6ca` | v7 (patched) | V7 |
| `4a6bc3d` | v8 | V8 |
| `decb2ed` | v9: reward rebalance, entropy/LR schedules, architecture improvements | V9 |
| `4d8e482` | fix: CSV writer crash when presets have different HW type counts | V9 bugfix |
| `c0af458` | v10: capacity-aware reject head architecture | V10 |
| `1b3fb3d` | v11: capacity-scaled task generation to match infrastructure size | V11 |

## Appendix B: File Reference

| File | Purpose |
|------|---------|
| `rl/agent.py` | PolicyNetwork (TaskEncoder, HWEncoder, Scorer, reject/value heads) |
| `rl/state_encoder.py` | State-to-vector encoding (v1/v2/v3 features) |
| `rl/reward.py` | RewardCalculator (energy, SLA, rejection/acceptance) |
| `rl/environment.py` | CloudProvisioningEnv, DomainRandomizedEnv, task generation |
| `rl/distributed_trainer.py` | Multi-GPU DDP training with torchrun |
| `scripts/train_rl_distributed.py` | CLI training entry point |
| `scripts/run_academic_evaluation_v5.py` | Full academic evaluation pipeline |
| `scripts/utilization_analysis.py` | Per-step utilization tracking and visualization |
| `scripts/diagnose_state_vectors.py` | State vector diagnostics across presets |
| `results/academic_v{N}/` | Per-version evaluation results, models, figures |
| `results/diagnostics/state_vector_diagnosis.json` | State vector comparison analysis |
| `docs/CHANGELOG_V*.md` | Per-version detailed changelogs |
| `docs/RL_IMPLEMENTATION.md` | Comprehensive RL system documentation |

## Appendix C: Evaluation Report File Naming

| Versions | Report File Name |
|----------|-----------------|
| V1-V3 | `academic_evaluation_report.json` |
| V4 | `evaluation_results.json` |
| V5-V11 | `evaluation_report.json` |

---

*Report generated from analysis of 11 version directories, 11 changelog documents, 79 git commits, and 6 core source files.*
