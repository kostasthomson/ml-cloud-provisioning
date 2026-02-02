#!/usr/bin/env python3
"""
Diagnostic Script: State Vector Analysis Across Presets

Logs and compares raw state vectors between medium and stress_test presets
to identify normalization issues and feature distribution differences.

Usage:
    python scripts/diagnose_state_vectors.py
    python scripts/diagnose_state_vectors.py --presets medium stress_test high_load
    python scripts/diagnose_state_vectors.py --num-samples 200 --output-dir results/diagnostics
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.environment import CloudProvisioningEnv, REALISTIC_HW_CONFIGS
from rl.state_encoder import StateEncoder
from rl.schemas import RLState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

FEATURE_NAMES = {
    'task': [
        'num_vms_norm', 'vcpus_norm', 'memory_norm', 'storage_norm',
        'network_norm', 'instructions_log', 'requires_acc', 'acc_rho',
        'num_compatible_norm', 'has_deadline', 'deadline_norm', 'task_size_norm'
    ],
    'global': [
        'total_power_norm', 'queue_length_norm', 'recent_accept_rate',
        'recent_avg_energy_norm', 'time_of_day'
    ],
    'scarcity': [
        'avg_cpu_util', 'avg_mem_util', 'min_cpu_capacity',
        'min_mem_capacity', 'scarcity_indicator'
    ],
    'hw': [
        'util_cpu', 'util_mem', 'util_storage', 'util_network', 'util_acc',
        'capacity_cpu', 'capacity_mem', 'capacity_storage', 'capacity_network',
        'capacity_acc', 'compute_cap_log', 'acc_compute_log',
        'power_idle_norm', 'power_max_norm', 'running_tasks_norm', 'avg_remaining_time_norm'
    ]
}


def collect_state_samples(
    preset: str,
    num_samples: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """Collect state vector samples from an environment preset."""

    env = CloudProvisioningEnv(preset=preset, max_steps=num_samples + 10, seed=seed)
    encoder = StateEncoder(use_scarcity_features=True)

    state, _ = env.reset()

    samples = {
        'task_vecs': [],
        'global_vecs': [],
        'scarcity_vecs': [],
        'hw_vecs': {},
        'raw_task_data': [],
        'raw_hw_data': [],
    }

    for hw in state.hw_types:
        samples['hw_vecs'][hw.hw_type_id] = []

    for i in range(num_samples):
        task_vec = encoder._encode_task(state.task)
        global_vec = encoder._encode_global(state.global_state)
        scarcity_vec = encoder._encode_scarcity(state)

        samples['task_vecs'].append(task_vec.tolist())
        samples['global_vecs'].append(global_vec.tolist())
        samples['scarcity_vecs'].append(scarcity_vec.tolist())

        samples['raw_task_data'].append({
            'num_vms': state.task.num_vms,
            'vcpus_per_vm': state.task.vcpus_per_vm,
            'memory_per_vm': state.task.memory_per_vm,
            'instructions': state.task.instructions,
            'requires_accelerator': state.task.requires_accelerator,
            'compatible_hw_types': list(state.task.compatible_hw_types),
        })

        for hw in state.hw_types:
            hw_vec = encoder._encode_single_hw(hw)
            if hw.hw_type_id not in samples['hw_vecs']:
                samples['hw_vecs'][hw.hw_type_id] = []
            samples['hw_vecs'][hw.hw_type_id].append(hw_vec.tolist())

        if i == 0:
            samples['raw_hw_data'] = [{
                'hw_type_id': hw.hw_type_id,
                'total_cpus': hw.total_cpus,
                'total_memory': hw.total_memory,
                'total_accelerators': hw.total_accelerators,
            } for hw in state.hw_types]

        action = np.random.choice(env.get_hw_type_ids() + [-1])
        state, _, done, truncated, _ = env.step(action)

        if done or truncated:
            state, _ = env.reset()

    return samples


def compute_statistics(samples: Dict[str, Any]) -> Dict[str, Any]:
    """Compute statistics for collected samples."""

    stats = {}

    task_arr = np.array(samples['task_vecs'])
    stats['task'] = {
        'mean': task_arr.mean(axis=0).tolist(),
        'std': task_arr.std(axis=0).tolist(),
        'min': task_arr.min(axis=0).tolist(),
        'max': task_arr.max(axis=0).tolist(),
    }

    global_arr = np.array(samples['global_vecs'])
    stats['global'] = {
        'mean': global_arr.mean(axis=0).tolist(),
        'std': global_arr.std(axis=0).tolist(),
        'min': global_arr.min(axis=0).tolist(),
        'max': global_arr.max(axis=0).tolist(),
    }

    scarcity_arr = np.array(samples['scarcity_vecs'])
    stats['scarcity'] = {
        'mean': scarcity_arr.mean(axis=0).tolist(),
        'std': scarcity_arr.std(axis=0).tolist(),
        'min': scarcity_arr.min(axis=0).tolist(),
        'max': scarcity_arr.max(axis=0).tolist(),
    }

    stats['hw'] = {}
    for hw_id, vecs in samples['hw_vecs'].items():
        hw_arr = np.array(vecs)
        stats['hw'][hw_id] = {
            'mean': hw_arr.mean(axis=0).tolist(),
            'std': hw_arr.std(axis=0).tolist(),
            'min': hw_arr.min(axis=0).tolist(),
            'max': hw_arr.max(axis=0).tolist(),
        }

    raw_tasks = samples['raw_task_data']
    stats['raw_task'] = {
        'avg_num_vms': np.mean([t['num_vms'] for t in raw_tasks]),
        'avg_vcpus': np.mean([t['vcpus_per_vm'] for t in raw_tasks]),
        'avg_memory': np.mean([t['memory_per_vm'] for t in raw_tasks]),
        'avg_total_cpus_needed': np.mean([t['num_vms'] * t['vcpus_per_vm'] for t in raw_tasks]),
        'avg_total_mem_needed': np.mean([t['num_vms'] * t['memory_per_vm'] for t in raw_tasks]),
        'pct_requires_acc': np.mean([t['requires_accelerator'] for t in raw_tasks]) * 100,
    }

    stats['raw_hw'] = samples['raw_hw_data']

    return stats


def compare_presets(all_stats: Dict[str, Dict], presets: List[str]) -> Dict[str, Any]:
    """Compare statistics across presets and identify issues."""

    comparison = {
        'feature_differences': {},
        'task_fit_analysis': {},
        'warnings': [],
    }

    for feat_type in ['task', 'global', 'scarcity']:
        feat_names = FEATURE_NAMES.get(feat_type, [])
        comparison['feature_differences'][feat_type] = {}

        for i, fname in enumerate(feat_names):
            means = {p: all_stats[p][feat_type]['mean'][i] for p in presets}
            stds = {p: all_stats[p][feat_type]['std'][i] for p in presets}

            mean_spread = max(means.values()) - min(means.values())

            comparison['feature_differences'][feat_type][fname] = {
                'means': means,
                'stds': stds,
                'spread': mean_spread,
            }

            if mean_spread < 0.05 and feat_type == 'scarcity':
                comparison['warnings'].append(
                    f"ISSUE: {feat_type}.{fname} has low spread ({mean_spread:.4f}) - "
                    f"presets may look identical to agent"
                )

    for preset in presets:
        raw = all_stats[preset]['raw_task']
        hw_data = all_stats[preset]['raw_hw']

        total_system_cpus = sum(h['total_cpus'] for h in hw_data)
        total_system_mem = sum(h['total_memory'] for h in hw_data)

        avg_task_cpus = raw['avg_total_cpus_needed']
        avg_task_mem = raw['avg_total_mem_needed']

        cpu_fit_ratio = total_system_cpus / avg_task_cpus if avg_task_cpus > 0 else 0
        mem_fit_ratio = total_system_mem / avg_task_mem if avg_task_mem > 0 else 0

        comparison['task_fit_analysis'][preset] = {
            'total_system_cpus': total_system_cpus,
            'total_system_memory': total_system_mem,
            'avg_task_cpus_needed': avg_task_cpus,
            'avg_task_mem_needed': avg_task_mem,
            'cpu_fit_ratio': cpu_fit_ratio,
            'mem_fit_ratio': mem_fit_ratio,
            'theoretical_concurrent_tasks': min(cpu_fit_ratio, mem_fit_ratio),
        }

    fit_ratios = [comparison['task_fit_analysis'][p]['theoretical_concurrent_tasks']
                  for p in presets]
    if max(fit_ratios) > 5 * min(fit_ratios):
        comparison['warnings'].append(
            f"CRITICAL: Task fit ratio varies {min(fit_ratios):.1f}x to {max(fit_ratios):.1f}x "
            f"across presets - agent sees similar states but outcomes differ drastically"
        )

    return comparison


def print_report(all_stats: Dict, comparison: Dict, presets: List[str]):
    """Print human-readable diagnostic report."""

    print("\n" + "=" * 80)
    print("STATE VECTOR DIAGNOSTIC REPORT")
    print("=" * 80)

    print("\n### RAW INFRASTRUCTURE COMPARISON ###\n")
    print(f"{'Preset':<15} {'Total CPUs':>12} {'Total Mem':>12} {'Accelerators':>12}")
    print("-" * 55)
    for preset in presets:
        hw_data = all_stats[preset]['raw_hw']
        total_cpus = sum(h['total_cpus'] for h in hw_data)
        total_mem = sum(h['total_memory'] for h in hw_data)
        total_acc = sum(h['total_accelerators'] for h in hw_data)
        print(f"{preset:<15} {total_cpus:>12} {total_mem:>12.0f} {total_acc:>12}")

    print("\n### TASK DISTRIBUTION (SAME ACROSS PRESETS) ###\n")
    for preset in presets:
        raw = all_stats[preset]['raw_task']
        print(f"{preset}:")
        print(f"  Avg task size: {raw['avg_total_cpus_needed']:.1f} vCPUs, "
              f"{raw['avg_total_mem_needed']:.1f} GB memory")
        print(f"  Accelerator tasks: {raw['pct_requires_acc']:.1f}%")

    print("\n### TASK-TO-INFRASTRUCTURE FIT ANALYSIS ###\n")
    print(f"{'Preset':<15} {'Sys CPUs':>10} {'Task CPUs':>10} {'Fit Ratio':>10} {'Concurrent':>10}")
    print("-" * 60)
    for preset in presets:
        fit = comparison['task_fit_analysis'][preset]
        print(f"{preset:<15} {fit['total_system_cpus']:>10} "
              f"{fit['avg_task_cpus_needed']:>10.1f} "
              f"{fit['cpu_fit_ratio']:>10.1f}x "
              f"{fit['theoretical_concurrent_tasks']:>10.1f}")

    print("\n### SCARCITY FEATURE ANALYSIS ###\n")
    print("These features SHOULD differ across presets to help the agent distinguish them:\n")

    scarcity_diffs = comparison['feature_differences'].get('scarcity', {})
    for fname, data in scarcity_diffs.items():
        spread = data['spread']
        status = "[GOOD]" if spread > 0.1 else "[LOW]" if spread > 0.05 else "[POOR]"
        print(f"  {fname}:")
        for preset in presets:
            mean = data['means'][preset]
            std = data['stds'][preset]
            print(f"    {preset}: mean={mean:.4f}, std={std:.4f}")
        print(f"    Spread: {spread:.4f} [{status}]")
        print()

    print("\n### WARNINGS AND ISSUES ###\n")
    if comparison['warnings']:
        for warning in comparison['warnings']:
            print(f"  [!] {warning}")
    else:
        print("  No critical issues detected.")

    print("\n### RECOMMENDATIONS ###\n")

    fit_ratios = [comparison['task_fit_analysis'][p]['theoretical_concurrent_tasks']
                  for p in presets]
    if max(fit_ratios) > 3 * min(fit_ratios):
        print("  1. ADD ABSOLUTE CAPACITY FEATURES:")
        print("     - total_system_cpus_normalized (tells agent about scale)")
        print("     - task_to_capacity_ratio (how big is task vs system)")
        print()

    scarcity_spreads = [d['spread'] for d in scarcity_diffs.values()]
    if any(s < 0.1 for s in scarcity_spreads):
        print("  2. IMPROVE SCARCITY SIGNALS:")
        print("     - Current utilization-based features have low variance")
        print("     - Add features that capture absolute headroom")
        print()

    print("  3. CONSIDER TASK SCALING:")
    print("     - Same task distribution on all presets")
    print("     - Large tasks impossible on stress_test but common on medium")
    print("     - Agent cannot learn this without capacity signals")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Diagnose state vector differences across presets')
    parser.add_argument('--presets', nargs='+', default=['medium', 'stress_test'],
                        help='Presets to compare')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of state samples per preset')
    parser.add_argument('--output-dir', type=str, default='results/diagnostics',
                        help='Output directory for JSON results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting samples from presets: {args.presets}")

    all_samples = {}
    all_stats = {}

    for preset in args.presets:
        logger.info(f"  Sampling {preset}...")
        samples = collect_state_samples(preset, args.num_samples, args.seed)
        stats = compute_statistics(samples)
        all_samples[preset] = samples
        all_stats[preset] = stats

    comparison = compare_presets(all_stats, args.presets)

    print_report(all_stats, comparison, args.presets)

    output_file = output_dir / 'state_vector_diagnosis.json'
    with open(output_file, 'w') as f:
        json.dump({
            'presets': args.presets,
            'num_samples': args.num_samples,
            'statistics': all_stats,
            'comparison': comparison,
        }, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == '__main__':
    main()
