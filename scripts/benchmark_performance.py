"""
Benchmark script for comparing PPO Agent vs Scoring Allocator vs Random strategies.
"""

import os
import sys
import random
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.environment import CloudProvisioningEnv
from rl.agent import RLAgent
from rl.schemas import RLState, HWTypeState, TaskState
from entities.schemas import (
    ScoringAllocationRequest, ScoringTaskImplementation, HardwareTypeStatus,
    OngoingTask, ResourceUsage
)
from entities.allocator.scoring_allocator import ScoringAllocator

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    total_energy_kwh: float = 0.0
    accepted_tasks: int = 0
    total_tasks: int = 0
    sla_violations: int = 0
    episode_rewards: List[float] = field(default_factory=list)
    episode_energies: List[float] = field(default_factory=list)
    episode_sla_rates: List[float] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tasks / max(self.total_tasks, 1)

    @property
    def sla_compliance_rate(self) -> float:
        if self.accepted_tasks == 0:
            return 1.0
        return 1.0 - (self.sla_violations / self.accepted_tasks)

    @property
    def efficiency_index(self) -> float:
        if self.accepted_tasks == 0:
            return float('inf')
        return self.total_energy_kwh / self.accepted_tasks


def adapt_rl_state_to_scoring_request(state: RLState) -> ScoringAllocationRequest:
    task = state.task
    implementations = []
    for hw_type_id in task.compatible_hw_types:
        requires_acc = task.requires_accelerator and hw_type_id != 1
        implementations.append(ScoringTaskImplementation(
            impl_id=hw_type_id,
            instructions=task.instructions,
            vcpus_per_vm=task.vcpus_per_vm,
            memory_per_vm=task.memory_per_vm,
            storage_per_vm=task.storage_per_vm,
            network_per_vm=task.network_per_vm,
            requires_accelerator=requires_acc,
            accelerator_rho=task.accelerator_rho if requires_acc else 0.0
        ))
    if not implementations:
        implementations.append(ScoringTaskImplementation(
            impl_id=1,
            instructions=task.instructions,
            vcpus_per_vm=task.vcpus_per_vm,
            memory_per_vm=task.memory_per_vm,
            storage_per_vm=task.storage_per_vm,
            network_per_vm=task.network_per_vm,
            requires_accelerator=False,
            accelerator_rho=0.0
        ))

    hw_statuses = []
    for hw in state.hw_types:
        hw_statuses.append(HardwareTypeStatus(
            hw_type_id=hw.hw_type_id,
            hw_type_name=f"HW-{hw.hw_type_id}",
            num_servers=1,
            total_cpus=hw.total_cpus,
            total_memory=hw.total_memory,
            total_storage=hw.total_storage,
            total_network=hw.total_network,
            total_accelerators=hw.total_accelerators,
            available_cpus=hw.available_cpus,
            available_memory=hw.available_memory,
            available_storage=hw.available_storage,
            available_network=hw.available_network,
            available_accelerators=hw.available_accelerators,
            compute_capability_per_cpu=hw.compute_capability,
            accelerator_compute_capability=hw.accelerator_compute_capability,
            cpu_idle_power=hw.power_idle,
            cpu_max_power=hw.power_max,
            acc_idle_power=hw.acc_power_idle,
            acc_max_power=hw.acc_power_max,
            ongoing_tasks=[]
        ))

    return ScoringAllocationRequest(
        timestamp=state.global_state.timestamp,
        task_id=task.task_id,
        num_vms=task.num_vms,
        implementations=implementations,
        hw_types=hw_statuses,
        weights=None
    )


def run_performance_study(
    env: CloudProvisioningEnv,
    strategy_name: str,
    model: Optional[RLAgent] = None,
    allocator: Optional[ScoringAllocator] = None,
    num_episodes: int = 20
) -> PerformanceMetrics:
    metrics = PerformanceMetrics()

    for episode in range(num_episodes):
        state, _ = env.reset(seed=episode)
        episode_energy = 0.0
        episode_accepted = 0
        episode_sla_violations = 0
        episode_total = 0
        episode_reward = 0.0
        done = False

        while not done:
            if strategy_name == "ppo":
                action_result = model.predict(state, deterministic=True)
                action = action_result[0].action
            elif strategy_name == "scoring":
                req = adapt_rl_state_to_scoring_request(state)
                res = allocator.allocate(req)
                action = res.selected_hw_type_id if res.selected_hw_type_id is not None else -1
            else:
                valid_ids = [hw.hw_type_id for hw in state.hw_types] + [-1]
                action = random.choice(valid_ids)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_total += 1
            metrics.total_tasks += 1

            if info.get('accepted', False):
                episode_accepted += 1
                metrics.accepted_tasks += 1
                episode_energy += info.get('energy', 0.0)
                metrics.total_energy_kwh += info.get('energy', 0.0)

                exec_time = info.get('exec_time', 0.0)
                deadline = state.task.deadline
                if deadline is not None and exec_time > deadline:
                    episode_sla_violations += 1
                    metrics.sla_violations += 1

            state = next_state

        metrics.episode_rewards.append(episode_reward)
        metrics.episode_energies.append(episode_energy)
        episode_sla_rate = 1.0 - (episode_sla_violations / max(episode_accepted, 1))
        metrics.episode_sla_rates.append(episode_sla_rate)

    return metrics


def cohens_d(group1: List[float], group2: List[float]) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_statistical_analysis(
    ppo_metrics: PerformanceMetrics,
    scoring_metrics: PerformanceMetrics
) -> Dict:
    energy_t, energy_p = stats.ttest_ind(
        ppo_metrics.episode_energies,
        scoring_metrics.episode_energies,
        equal_var=False
    )
    energy_d = cohens_d(ppo_metrics.episode_energies, scoring_metrics.episode_energies)

    sla_t, sla_p = stats.ttest_ind(
        ppo_metrics.episode_sla_rates,
        scoring_metrics.episode_sla_rates,
        equal_var=False
    )
    sla_d = cohens_d(ppo_metrics.episode_sla_rates, scoring_metrics.episode_sla_rates)

    return {
        'energy': {
            't_statistic': energy_t,
            'p_value': energy_p,
            'cohens_d': energy_d,
            'ppo_mean': np.mean(ppo_metrics.episode_energies),
            'ppo_std': np.std(ppo_metrics.episode_energies),
            'scoring_mean': np.mean(scoring_metrics.episode_energies),
            'scoring_std': np.std(scoring_metrics.episode_energies)
        },
        'sla_compliance': {
            't_statistic': sla_t,
            'p_value': sla_p,
            'cohens_d': sla_d,
            'ppo_mean': np.mean(ppo_metrics.episode_sla_rates),
            'ppo_std': np.std(ppo_metrics.episode_sla_rates),
            'scoring_mean': np.mean(scoring_metrics.episode_sla_rates),
            'scoring_std': np.std(scoring_metrics.episode_sla_rates)
        }
    }


def save_raw_data(
    results: Dict[str, PerformanceMetrics],
    output_path: str
):
    rows = []
    for strategy, metrics in results.items():
        for i in range(len(metrics.episode_energies)):
            rows.append({
                'strategy': strategy,
                'episode': i,
                'energy_kwh': metrics.episode_energies[i],
                'reward': metrics.episode_rewards[i],
                'sla_compliance_rate': metrics.episode_sla_rates[i]
            })
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Raw data saved to: {output_path}")


def print_summary_table(results: Dict[str, PerformanceMetrics], stats_analysis: Dict):
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    headers = ["Metric", "PPO Agent", "Scoring", "Random"]
    print(f"\n{headers[0]:<25} {headers[1]:<18} {headers[2]:<18} {headers[3]:<18}")
    print("-" * 80)

    ppo = results.get('ppo')
    scoring = results.get('scoring')
    random_strat = results.get('random')

    print(f"{'Total Energy (kWh)':<25} {ppo.total_energy_kwh:<18.4f} {scoring.total_energy_kwh:<18.4f} {random_strat.total_energy_kwh:<18.4f}")
    print(f"{'Accepted Tasks':<25} {ppo.accepted_tasks:<18} {scoring.accepted_tasks:<18} {random_strat.accepted_tasks:<18}")
    print(f"{'Total Tasks':<25} {ppo.total_tasks:<18} {scoring.total_tasks:<18} {random_strat.total_tasks:<18}")
    print(f"{'Acceptance Rate (%)':<25} {ppo.acceptance_rate*100:<18.2f} {scoring.acceptance_rate*100:<18.2f} {random_strat.acceptance_rate*100:<18.2f}")
    print(f"{'SLA Compliance Rate (%)':<25} {ppo.sla_compliance_rate*100:<18.2f} {scoring.sla_compliance_rate*100:<18.2f} {random_strat.sla_compliance_rate*100:<18.2f}")
    print(f"{'Efficiency (kWh/task)':<25} {ppo.efficiency_index:<18.6f} {scoring.efficiency_index:<18.6f} {random_strat.efficiency_index:<18.6f}")

    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS (PPO vs Scoring)")
    print("=" * 80)

    energy_stats = stats_analysis['energy']
    sla_stats = stats_analysis['sla_compliance']

    cohens_label = "Cohen's d"
    print(f"\n{'Metric':<20} {'t-statistic':<15} {'p-value':<15} {cohens_label:<15} {'Interpretation'}")
    print("-" * 80)

    def interpret_effect(d):
        d_abs = abs(d)
        if d_abs < 0.2:
            return "Negligible"
        elif d_abs < 0.5:
            return "Small"
        elif d_abs < 0.8:
            return "Medium"
        else:
            return "Large"

    def interpret_significance(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    print(f"{'Energy':<20} {energy_stats['t_statistic']:<15.4f} {energy_stats['p_value']:<15.4f} {energy_stats['cohens_d']:<15.4f} {interpret_effect(energy_stats['cohens_d'])} {interpret_significance(energy_stats['p_value'])}")
    print(f"{'SLA Compliance':<20} {sla_stats['t_statistic']:<15.4f} {sla_stats['p_value']:<15.4f} {sla_stats['cohens_d']:<15.4f} {interpret_effect(sla_stats['cohens_d'])} {interpret_significance(sla_stats['p_value'])}")

    print("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("=" * 80)


def main():
    print("Initializing benchmark evaluation...")

    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "rl" / "ppo" / "model_agnostic.pth"

    num_episodes = 20
    env = CloudProvisioningEnv(preset='medium', episode_length=100)

    print(f"Loading PPO agent from: {model_path}")
    ppo_agent = RLAgent(model_path=str(model_path) if model_path.exists() else None)
    if not model_path.exists():
        print(f"Warning: Model not found at {model_path}, using untrained agent")

    scoring_allocator = ScoringAllocator()

    print(f"\nRunning benchmark with {num_episodes} episodes per strategy...")

    print("\n[1/3] Evaluating PPO Agent...")
    ppo_metrics = run_performance_study(
        env, "ppo", model=ppo_agent, num_episodes=num_episodes
    )

    print("[2/3] Evaluating Scoring Allocator...")
    scoring_metrics = run_performance_study(
        env, "scoring", allocator=scoring_allocator, num_episodes=num_episodes
    )

    print("[3/3] Evaluating Random Strategy...")
    random_metrics = run_performance_study(
        env, "random", num_episodes=num_episodes
    )

    results = {
        'ppo': ppo_metrics,
        'scoring': scoring_metrics,
        'random': random_metrics
    }

    stats_analysis = run_statistical_analysis(ppo_metrics, scoring_metrics)

    output_path = project_root / "results" / "benchmark_raw_data.csv"
    save_raw_data(results, str(output_path))

    print_summary_table(results, stats_analysis)

    print("\nBenchmark evaluation complete.")


if __name__ == "__main__":
    main()
