#!/usr/bin/env python3
"""
Utilization analysis module for RL agent behavior analysis.

Tracks per-step metrics (CPU, memory, network, accelerator utilization per HW type)
and generates detailed figures showing utilization over time with energy and
acceptance rate overlays.

Usage:
    python scripts/utilization_analysis.py --model models/rl/ppo/model.pth --preset medium
    python scripts/utilization_analysis.py --model models/rl/ppo/model.pth --all-presets
"""

import sys
import argparse
import json
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", message="Can't initialize NVML")
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from rl.environment import CloudProvisioningEnv, REALISTIC_HW_CONFIGS
from rl.agent import RLAgent
from rl.state_encoder import StateEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    step: int
    timestamp: float
    task_id: str
    action: int
    accepted: bool
    energy_kwh: float
    cumulative_energy_kwh: float
    cumulative_acceptance_rate: float
    hw_utilizations: Dict[int, Dict[str, float]] = field(default_factory=dict)
    task_requirements: Dict[str, float] = field(default_factory=dict)
    rejection_reason: Optional[str] = None


@dataclass
class EpisodeMetrics:
    episode_id: int
    preset: str
    steps: List[StepMetrics] = field(default_factory=list)
    total_energy_kwh: float = 0.0
    total_accepted: int = 0
    total_rejected: int = 0
    final_acceptance_rate: float = 0.0
    sla_violations: int = 0
    capacity_rejections: int = 0
    policy_rejections: int = 0


class UtilizationTracker:
    def __init__(self, agent: RLAgent, env: CloudProvisioningEnv, preset: str):
        self.agent = agent
        self.env = env
        self.preset = preset
        self.encoder = agent.encoder
        self.device = agent.device

    def run_episode(self, episode_id: int, max_steps: int = 500) -> EpisodeMetrics:
        metrics = EpisodeMetrics(episode_id=episode_id, preset=self.preset)
        state, _ = self.env.reset()

        cumulative_energy = 0.0
        accepted_count = 0
        rejected_count = 0

        for step in range(max_steps):
            task_vec, hw_list = self.encoder.encode(state)
            valid_hw_types = self.encoder.get_valid_hw_types(state)

            hw_type_ids = [hw_id for hw_id, _ in hw_list]
            hw_vecs = [hw_vec for _, hw_vec in hw_list]

            valid_mask = np.array([hw_id in valid_hw_types for hw_id in hw_type_ids], dtype=bool)

            task_tensor = torch.FloatTensor(task_vec).to(self.device)
            hw_tensors = [torch.FloatTensor(hv).to(self.device) for hv in hw_vecs]
            mask_tensor = torch.BoolTensor(valid_mask).to(self.device)

            with torch.no_grad():
                action_probs, _, _ = self.agent.policy.forward(task_tensor, hw_tensors, mask_tensor)

            probs = action_probs.cpu().numpy()
            action_idx = np.argmax(probs)

            if action_idx < len(hw_type_ids):
                selected_hw_id = hw_type_ids[action_idx]
            else:
                selected_hw_id = -1

            next_state, reward, done, truncated, info = self.env.step(selected_hw_id)

            hw_utils_after = self._capture_hw_utilizations(next_state)

            accepted = info.get('accepted', False)
            energy = info.get('energy', 0.0)

            if accepted:
                accepted_count += 1
                cumulative_energy += energy
            else:
                rejected_count += 1

            total_tasks = accepted_count + rejected_count
            cumulative_acc_rate = accepted_count / total_tasks if total_tasks > 0 else 0.0

            rejection_reason = None
            if not accepted:
                if selected_hw_id == -1:
                    rejection_reason = "policy_reject"
                    metrics.policy_rejections += 1
                elif not valid_hw_types:
                    rejection_reason = "no_capacity"
                    metrics.capacity_rejections += 1
                elif selected_hw_id not in valid_hw_types:
                    rejection_reason = "invalid_choice"
                    metrics.capacity_rejections += 1
                else:
                    rejection_reason = "capacity_exhausted"
                    metrics.capacity_rejections += 1

            step_metrics = StepMetrics(
                step=step,
                timestamp=self.env.timestamp,
                task_id=state.task.task_id,
                action=selected_hw_id,
                accepted=accepted,
                energy_kwh=energy,
                cumulative_energy_kwh=cumulative_energy,
                cumulative_acceptance_rate=cumulative_acc_rate,
                hw_utilizations=hw_utils_after,
                task_requirements={
                    'num_vms': state.task.num_vms,
                    'vcpus_per_vm': state.task.vcpus_per_vm,
                    'memory_per_vm': state.task.memory_per_vm,
                    'requires_accelerator': state.task.requires_accelerator,
                    'total_cpu_req': state.task.num_vms * state.task.vcpus_per_vm,
                    'total_mem_req': state.task.num_vms * state.task.memory_per_vm,
                },
                rejection_reason=rejection_reason
            )
            metrics.steps.append(step_metrics)

            state = next_state

            if done or truncated:
                break

        metrics.total_energy_kwh = cumulative_energy
        metrics.total_accepted = accepted_count
        metrics.total_rejected = rejected_count
        metrics.final_acceptance_rate = cumulative_acc_rate

        return metrics

    def _capture_hw_utilizations(self, state) -> Dict[int, Dict[str, float]]:
        utils = {}
        for hw in state.hw_types:
            utils[hw.hw_type_id] = {
                'cpu': hw.utilization_cpu,
                'memory': hw.utilization_memory,
                'storage': hw.utilization_storage,
                'network': hw.utilization_network,
                'accelerator': hw.utilization_accelerator,
                'available_cpus': hw.available_cpus,
                'available_memory': hw.available_memory,
                'total_cpus': hw.total_cpus,
                'total_memory': hw.total_memory,
                'running_tasks': hw.num_running_tasks,
            }
        return utils


class UtilizationVisualizer:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.colors = {
            'cpu': '#2ecc71',
            'memory': '#3498db',
            'storage': '#9b59b6',
            'network': '#e74c3c',
            'accelerator': '#f39c12',
            'energy': '#1abc9c',
            'acceptance': '#e67e22',
        }

        self.hw_colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    def plot_episode_utilization(self, metrics: EpisodeMetrics, save_name: str = None):
        if not metrics.steps:
            logger.warning("No steps to plot")
            return

        hw_ids = sorted(metrics.steps[0].hw_utilizations.keys())
        n_hw = len(hw_ids)

        fig = plt.figure(figsize=(16, 4 + n_hw * 3), constrained_layout=True)
        gs = gridspec.GridSpec(2 + n_hw, 1, height_ratios=[2, 2] + [2] * n_hw, hspace=0.3, figure=fig)

        steps = [s.step for s in metrics.steps]

        ax_energy = fig.add_subplot(gs[0])
        cumulative_energy = [s.cumulative_energy_kwh for s in metrics.steps]
        ax_energy.plot(steps, cumulative_energy, color=self.colors['energy'], linewidth=2, label='Cumulative Energy')
        ax_energy.fill_between(steps, cumulative_energy, alpha=0.3, color=self.colors['energy'])
        ax_energy.set_ylabel('Cumulative Energy (kWh)', fontsize=10)
        ax_energy.set_title(f'Energy Consumption Over Time - {metrics.preset.upper()} Preset', fontsize=12, fontweight='bold')
        ax_energy.legend(loc='upper left')
        ax_energy.grid(True, alpha=0.3)

        ax_acc = ax_energy.twinx()
        acceptance_rate = [s.cumulative_acceptance_rate * 100 for s in metrics.steps]
        ax_acc.plot(steps, acceptance_rate, color=self.colors['acceptance'], linewidth=2, linestyle='--', label='Acceptance Rate')
        ax_acc.set_ylabel('Acceptance Rate (%)', fontsize=10, color=self.colors['acceptance'])
        ax_acc.tick_params(axis='y', labelcolor=self.colors['acceptance'])
        ax_acc.set_ylim(0, 100)
        ax_acc.legend(loc='upper right')

        ax_events = fig.add_subplot(gs[1])
        accept_steps = [s.step for s in metrics.steps if s.accepted]
        reject_steps = [s.step for s in metrics.steps if not s.accepted]
        capacity_reject_steps = [s.step for s in metrics.steps if s.rejection_reason and 'capacity' in s.rejection_reason]
        policy_reject_steps = [s.step for s in metrics.steps if s.rejection_reason == 'policy_reject']

        ax_events.eventplot([accept_steps], lineoffsets=0.8, linelengths=0.3, colors=['#2ecc71'], label='Accepted')
        if capacity_reject_steps:
            ax_events.eventplot([capacity_reject_steps], lineoffsets=0.5, linelengths=0.3, colors=['#e74c3c'], label='Capacity Reject')
        if policy_reject_steps:
            ax_events.eventplot([policy_reject_steps], lineoffsets=0.2, linelengths=0.3, colors=['#f39c12'], label='Policy Reject')

        ax_events.set_ylabel('Events', fontsize=10)
        ax_events.set_yticks([0.2, 0.5, 0.8])
        ax_events.set_yticklabels(['Policy', 'Capacity', 'Accept'])
        ax_events.set_xlim(0, len(steps))
        ax_events.legend(loc='upper right', ncol=3)
        ax_events.set_title('Task Decisions Timeline', fontsize=11, fontweight='bold')
        ax_events.grid(True, alpha=0.3, axis='x')

        for i, hw_id in enumerate(hw_ids):
            ax = fig.add_subplot(gs[2 + i])

            cpu_util = [s.hw_utilizations[hw_id]['cpu'] * 100 for s in metrics.steps]
            mem_util = [s.hw_utilizations[hw_id]['memory'] * 100 for s in metrics.steps]
            net_util = [s.hw_utilizations[hw_id]['network'] * 100 for s in metrics.steps]
            acc_util = [s.hw_utilizations[hw_id]['accelerator'] * 100 for s in metrics.steps]

            ax.plot(steps, cpu_util, color=self.colors['cpu'], linewidth=1.5, label='CPU', alpha=0.9)
            ax.plot(steps, mem_util, color=self.colors['memory'], linewidth=1.5, label='Memory', alpha=0.9)
            ax.plot(steps, net_util, color=self.colors['network'], linewidth=1.5, label='Network', alpha=0.9)

            if any(u > 0 for u in acc_util):
                ax.plot(steps, acc_util, color=self.colors['accelerator'], linewidth=1.5, label='Accelerator', alpha=0.9)

            hw_name = self._get_hw_name(metrics.preset, hw_id)
            ax.set_ylabel(f'Utilization (%)', fontsize=10)
            ax.set_title(f'HW Type {hw_id}: {hw_name}', fontsize=11, fontweight='bold')
            ax.set_ylim(0, 105)
            ax.legend(loc='upper right', ncol=4, fontsize=8)
            ax.grid(True, alpha=0.3)

            ax.axhline(y=80, color='#e74c3c', linestyle=':', alpha=0.5, label='80% threshold')
            ax.axhline(y=95, color='#c0392b', linestyle='--', alpha=0.5, label='95% critical')

        ax.set_xlabel('Step', fontsize=10)

        if save_name:
            for fmt in ['png', 'pdf']:
                fig.savefig(self.output_dir / f'{save_name}.{fmt}', dpi=150, bbox_inches='tight')
            logger.info(f"Saved utilization plot: {save_name}")

        plt.close(fig)

    def plot_comparative_utilization(self, all_metrics: Dict[str, List[EpisodeMetrics]], save_name: str = 'comparative_utilization'):
        n_presets = len(all_metrics)
        fig, axes = plt.subplots(n_presets, 3, figsize=(18, 4 * n_presets), constrained_layout=True)

        if n_presets == 1:
            axes = axes.reshape(1, -1)

        for row, (preset, episodes) in enumerate(all_metrics.items()):
            if not episodes or not episodes[0].steps:
                continue

            avg_metrics = self._average_episode_metrics(episodes)
            steps = list(range(len(avg_metrics['cpu_util'])))

            ax_util = axes[row, 0]
            ax_util.plot(steps, avg_metrics['cpu_util'], color=self.colors['cpu'], linewidth=2, label='CPU')
            ax_util.plot(steps, avg_metrics['mem_util'], color=self.colors['memory'], linewidth=2, label='Memory')
            ax_util.plot(steps, avg_metrics['net_util'], color=self.colors['network'], linewidth=2, label='Network')
            ax_util.fill_between(steps, avg_metrics['cpu_util'], alpha=0.2, color=self.colors['cpu'])
            ax_util.set_ylabel('Avg Utilization (%)', fontsize=10)
            ax_util.set_title(f'{preset.upper()} - Resource Utilization', fontsize=11, fontweight='bold')
            ax_util.set_ylim(0, 100)
            ax_util.legend(loc='upper right')
            ax_util.grid(True, alpha=0.3)

            ax_energy = axes[row, 1]
            ax_energy.plot(steps, avg_metrics['cumulative_energy'], color=self.colors['energy'], linewidth=2)
            ax_energy.fill_between(steps, avg_metrics['cumulative_energy'], alpha=0.3, color=self.colors['energy'])
            ax_energy.set_ylabel('Cumulative Energy (kWh)', fontsize=10)
            ax_energy.set_title(f'{preset.upper()} - Energy Consumption', fontsize=11, fontweight='bold')
            ax_energy.grid(True, alpha=0.3)

            ax_acc = axes[row, 2]
            ax_acc.plot(steps, avg_metrics['acceptance_rate'], color=self.colors['acceptance'], linewidth=2)
            ax_acc.fill_between(steps, avg_metrics['acceptance_rate'], alpha=0.3, color=self.colors['acceptance'])
            ax_acc.set_ylabel('Acceptance Rate (%)', fontsize=10)
            ax_acc.set_title(f'{preset.upper()} - Acceptance Rate', fontsize=11, fontweight='bold')
            ax_acc.set_ylim(0, 100)
            ax_acc.grid(True, alpha=0.3)

            final_acc = avg_metrics['acceptance_rate'][-1] if avg_metrics['acceptance_rate'] else 0
            ax_acc.axhline(y=final_acc, color='#c0392b', linestyle='--', alpha=0.7)
            ax_acc.annotate(f'Final: {final_acc:.1f}%', xy=(len(steps)*0.8, final_acc+3), fontsize=9)

        for ax in axes[-1]:
            ax.set_xlabel('Step', fontsize=10)

        for fmt in ['png', 'pdf']:
            fig.savefig(self.output_dir / f'{save_name}.{fmt}', dpi=150, bbox_inches='tight')

        plt.close(fig)
        logger.info(f"Saved comparative plot: {save_name}")

    def plot_rejection_analysis(self, all_metrics: Dict[str, List[EpisodeMetrics]], save_name: str = 'rejection_analysis'):
        presets = list(all_metrics.keys())
        n_presets = len(presets)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        capacity_rejects = []
        policy_rejects = []
        accepts = []

        for preset in presets:
            episodes = all_metrics[preset]
            cap_rej = np.mean([e.capacity_rejections for e in episodes])
            pol_rej = np.mean([e.policy_rejections for e in episodes])
            acc = np.mean([e.total_accepted for e in episodes])
            capacity_rejects.append(cap_rej)
            policy_rejects.append(pol_rej)
            accepts.append(acc)

        x = np.arange(n_presets)
        width = 0.25

        ax1 = axes[0]
        bars1 = ax1.bar(x - width, accepts, width, label='Accepted', color='#2ecc71')
        bars2 = ax1.bar(x, capacity_rejects, width, label='Capacity Reject', color='#e74c3c')
        bars3 = ax1.bar(x + width, policy_rejects, width, label='Policy Reject', color='#f39c12')

        ax1.set_ylabel('Average Count per Episode', fontsize=11)
        ax1.set_title('Task Outcomes by Preset', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([p.upper() for p in presets])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[1]
        total_tasks = [a + c + p for a, c, p in zip(accepts, capacity_rejects, policy_rejects)]
        acc_pct = [100 * a / t if t > 0 else 0 for a, t in zip(accepts, total_tasks)]
        cap_pct = [100 * c / t if t > 0 else 0 for c, t in zip(capacity_rejects, total_tasks)]
        pol_pct = [100 * p / t if t > 0 else 0 for p, t in zip(policy_rejects, total_tasks)]

        ax2.bar(x, acc_pct, width=0.5, label='Accepted', color='#2ecc71')
        ax2.bar(x, cap_pct, width=0.5, bottom=acc_pct, label='Capacity Reject', color='#e74c3c')
        bottom2 = [a + c for a, c in zip(acc_pct, cap_pct)]
        ax2.bar(x, pol_pct, width=0.5, bottom=bottom2, label='Policy Reject', color='#f39c12')

        ax2.set_ylabel('Percentage (%)', fontsize=11)
        ax2.set_title('Task Outcome Distribution', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([p.upper() for p in presets])
        ax2.legend(loc='upper right')
        ax2.set_ylim(0, 105)
        ax2.grid(True, alpha=0.3, axis='y')

        for fmt in ['png', 'pdf']:
            fig.savefig(self.output_dir / f'{save_name}.{fmt}', dpi=150, bbox_inches='tight')

        plt.close(fig)
        logger.info(f"Saved rejection analysis: {save_name}")

    def _average_episode_metrics(self, episodes: List[EpisodeMetrics]) -> Dict[str, List[float]]:
        if not episodes:
            return {}

        min_steps = min(len(e.steps) for e in episodes)

        cpu_utils = []
        mem_utils = []
        net_utils = []
        energies = []
        acc_rates = []

        for step_idx in range(min_steps):
            cpu_sum = 0
            mem_sum = 0
            net_sum = 0
            energy_sum = 0
            acc_sum = 0

            for ep in episodes:
                step = ep.steps[step_idx]
                hw_ids = list(step.hw_utilizations.keys())

                cpu_sum += np.mean([step.hw_utilizations[h]['cpu'] for h in hw_ids]) * 100
                mem_sum += np.mean([step.hw_utilizations[h]['memory'] for h in hw_ids]) * 100
                net_sum += np.mean([step.hw_utilizations[h]['network'] for h in hw_ids]) * 100
                energy_sum += step.cumulative_energy_kwh
                acc_sum += step.cumulative_acceptance_rate * 100

            n = len(episodes)
            cpu_utils.append(cpu_sum / n)
            mem_utils.append(mem_sum / n)
            net_utils.append(net_sum / n)
            energies.append(energy_sum / n)
            acc_rates.append(acc_sum / n)

        return {
            'cpu_util': cpu_utils,
            'mem_util': mem_utils,
            'net_util': net_utils,
            'cumulative_energy': energies,
            'acceptance_rate': acc_rates,
        }

    def _get_hw_name(self, preset: str, hw_id: int) -> str:
        configs = REALISTIC_HW_CONFIGS.get(preset, [])
        for cfg in configs:
            if cfg.hw_type_id == hw_id:
                return cfg.name
        return f"HW-{hw_id}"


def run_analysis(
    model_path: str,
    presets: List[str],
    output_dir: str,
    n_episodes: int = 5,
    max_steps: int = 500,
    seed: int = 42
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {model_path}")
    agent = RLAgent()
    agent.load(model_path)
    agent.policy.eval()

    visualizer = UtilizationVisualizer(output_path)
    all_metrics: Dict[str, List[EpisodeMetrics]] = {}

    for preset in presets:
        logger.info(f"Running analysis for preset: {preset}")

        env = CloudProvisioningEnv(
            preset=preset,
            max_steps=max_steps,
            seed=seed
        )

        tracker = UtilizationTracker(agent, env, preset)
        episodes = []

        for ep_id in range(n_episodes):
            metrics = tracker.run_episode(ep_id, max_steps)
            episodes.append(metrics)
            logger.info(f"  Episode {ep_id}: accepted={metrics.total_accepted}, "
                       f"rejected={metrics.total_rejected}, energy={metrics.total_energy_kwh:.4f} kWh")

        all_metrics[preset] = episodes

        visualizer.plot_episode_utilization(
            episodes[0],
            save_name=f'utilization_{preset}_episode_0'
        )

    visualizer.plot_comparative_utilization(all_metrics, save_name='comparative_utilization')
    visualizer.plot_rejection_analysis(all_metrics, save_name='rejection_analysis')

    summary = {}
    for preset, episodes in all_metrics.items():
        summary[preset] = {
            'avg_acceptance_rate': np.mean([e.final_acceptance_rate for e in episodes]),
            'std_acceptance_rate': np.std([e.final_acceptance_rate for e in episodes]),
            'avg_energy_kwh': np.mean([e.total_energy_kwh for e in episodes]),
            'std_energy_kwh': np.std([e.total_energy_kwh for e in episodes]),
            'avg_capacity_rejections': np.mean([e.capacity_rejections for e in episodes]),
            'avg_policy_rejections': np.mean([e.policy_rejections for e in episodes]),
            'capacity_rejection_ratio': np.mean([
                e.capacity_rejections / max(e.total_rejected, 1) for e in episodes
            ]),
        }

    summary_path = output_path / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Analysis complete. Results saved to {output_path}")

    return all_metrics, summary


def main():
    parser = argparse.ArgumentParser(
        description='Analyze RL agent utilization patterns',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--preset', type=str, default=None,
                        choices=['small', 'medium', 'large', 'enterprise', 'high_load', 'stress_test'],
                        help='Single preset to analyze')
    parser.add_argument('--all-presets', action='store_true',
                        help='Analyze all presets')
    parser.add_argument('--output-dir', type=str, default='results/utilization_analysis',
                        help='Output directory for figures')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes per preset')
    parser.add_argument('--max-steps', type=int, default=500,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    if args.all_presets:
        presets = ['small', 'medium', 'large', 'high_load', 'stress_test']
    elif args.preset:
        presets = [args.preset]
    else:
        presets = ['medium']

    run_analysis(
        model_path=args.model,
        presets=presets,
        output_dir=args.output_dir,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
