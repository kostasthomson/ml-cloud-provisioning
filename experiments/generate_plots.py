#!/usr/bin/env python3
"""
Plot Generation for Academic Results.

Generates publication-ready figures from experiment results.

Usage:
    python generate_plots.py [--results-dir results/academic]
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.style.use('seaborn-v0_8-whitegrid')
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Install with: pip install matplotlib")


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load all experiment results from JSON files."""
    results = {}

    for json_file in (results_dir / "data").glob("*.json"):
        try:
            with open(json_file) as f:
                results[json_file.stem] = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load {json_file}: {e}")

    return results


def plot_pareto_front(results: Dict, output_dir: Path):
    """Generate Pareto front plot."""
    if "pareto_results" not in results:
        logger.warning("No Pareto results found, skipping plot")
        return

    data = results["pareto_results"]
    points = data.get("all_points", [])
    frontier = data.get("pareto_frontier", [])

    if not points:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    energies = [p["avg_energy_per_task"] * 1000 for p in points]
    acceptances = [p["acceptance_rate"] * 100 for p in points]
    weights = [p["energy_weight"] for p in points]

    scatter = ax.scatter(acceptances, energies, c=weights, cmap='viridis',
                         s=150, edgecolors='black', linewidths=1.5, zorder=3)

    if frontier:
        frontier_energies = [p["avg_energy_per_task"] * 1000 for p in frontier]
        frontier_acceptances = [p["acceptance_rate"] * 100 for p in frontier]
        sorted_pairs = sorted(zip(frontier_acceptances, frontier_energies))
        f_acc, f_eng = zip(*sorted_pairs)
        ax.plot(f_acc, f_eng, 'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')

    for i, (acc, eng, w) in enumerate(zip(acceptances, energies, weights)):
        ax.annotate(f'w={w:.1f}', (acc, eng), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Energy Weight', fontsize=12)

    ax.set_xlabel('Acceptance Rate (%)', fontsize=12)
    ax.set_ylabel('Energy per Task (Wh)', fontsize=12)
    ax.set_title('Energy-Acceptance Tradeoff (Pareto Front)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'pareto_front.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'pareto_front.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved Pareto front plot")


def plot_ablation_study(results: Dict, output_dir: Path):
    """Generate ablation study bar chart."""
    if "ablation_results" not in results:
        logger.warning("No ablation results found, skipping plot")
        return

    data = results["ablation_results"]
    ablation_results = data.get("results", [])

    if not ablation_results:
        return

    configs = [r["config_name"] for r in ablation_results]
    energies = [r["avg_energy_per_task"] * 1000 for r in ablation_results]
    acceptances = [r["acceptance_rate"] * 100 for r in ablation_results]
    sla_rates = [r["sla_compliance_rate"] * 100 for r in ablation_results]

    x = np.arange(len(configs))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, energies, width, label='Energy (Wh/task)', color='#2ecc71')
    bars2 = ax.bar(x, acceptances, width, label='Acceptance (%)', color='#3498db')
    bars3 = ax.bar(x + width, sla_rates, width, label='SLA Compliance (%)', color='#e74c3c')

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Ablation Study: Impact of Reward Components', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    full_idx = configs.index("full") if "full" in configs else 0
    ax.axhline(y=energies[full_idx], color='#2ecc71', linestyle='--', alpha=0.5)
    ax.axhline(y=acceptances[full_idx], color='#3498db', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'ablation_study.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ablation_study.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved ablation study plot")


def plot_generalization(results: Dict, output_dir: Path):
    """Generate generalization experiment plot."""
    if "generalization_results" not in results:
        logger.warning("No generalization results found, skipping plot")
        return

    data = results["generalization_results"]
    gen_results = data.get("results", [])

    if not gen_results:
        return

    presets = [r["test_preset"] for r in gen_results]
    energies = [r["avg_energy_per_task"] * 1000 for r in gen_results]
    acceptances = [r["acceptance_rate"] * 100 for r in gen_results]
    hw_counts = [r["num_hw_types_test"] for r in gen_results]
    is_same = [r["is_same_config"] for r in gen_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#27ae60' if same else '#3498db' for same in is_same]

    bars1 = ax1.bar(presets, energies, color=colors, edgecolor='black')
    ax1.set_xlabel('Test Configuration', fontsize=12)
    ax1.set_ylabel('Energy per Task (Wh)', fontsize=12)
    ax1.set_title('Energy Consumption Across Configurations', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, count in zip(bars1, hw_counts):
        ax1.annotate(f'{count} HW', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(presets, acceptances, color=colors, edgecolor='black')
    ax2.set_xlabel('Test Configuration', fontsize=12)
    ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax2.set_title('Acceptance Rate Across Configurations', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    train_patch = mpatches.Patch(color='#27ae60', label='Training Config')
    test_patch = mpatches.Patch(color='#3498db', label='Unseen Config')
    ax1.legend(handles=[train_patch, test_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'generalization.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'generalization.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved generalization plot")


def plot_multi_seed_distribution(results: Dict, output_dir: Path):
    """Generate multi-seed results distribution plot."""
    if "multi_seed_results" not in results:
        logger.warning("No multi-seed results found, skipping plot")
        return

    data = results["multi_seed_results"]
    eval_results = data.get("evaluation_results", [])

    if not eval_results:
        return

    energies = [r["energy_per_task"] * 1000 for r in eval_results]
    acceptances = [r["acceptance_rate"] * 100 for r in eval_results]
    rewards = [r["avg_episode_reward"] for r in eval_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].boxplot(energies, patch_artist=True,
                    boxprops=dict(facecolor='#2ecc71', alpha=0.7))
    axes[0].scatter([1] * len(energies), energies, alpha=0.5, color='black', zorder=3)
    axes[0].set_ylabel('Energy per Task (Wh)', fontsize=12)
    axes[0].set_title(f'Energy Distribution\n(n={len(energies)} seeds)', fontsize=12)
    axes[0].set_xticks([])

    axes[1].boxplot(acceptances, patch_artist=True,
                    boxprops=dict(facecolor='#3498db', alpha=0.7))
    axes[1].scatter([1] * len(acceptances), acceptances, alpha=0.5, color='black', zorder=3)
    axes[1].set_ylabel('Acceptance Rate (%)', fontsize=12)
    axes[1].set_title(f'Acceptance Distribution\n(n={len(acceptances)} seeds)', fontsize=12)
    axes[1].set_xticks([])

    axes[2].boxplot(rewards, patch_artist=True,
                    boxprops=dict(facecolor='#e74c3c', alpha=0.7))
    axes[2].scatter([1] * len(rewards), rewards, alpha=0.5, color='black', zorder=3)
    axes[2].set_ylabel('Average Episode Reward', fontsize=12)
    axes[2].set_title(f'Reward Distribution\n(n={len(rewards)} seeds)', fontsize=12)
    axes[2].set_xticks([])

    for ax in axes:
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'multi_seed_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'multi_seed_distribution.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved multi-seed distribution plot")


def plot_baseline_comparison(results: Dict, output_dir: Path):
    """Generate baseline comparison plot."""
    report_path = output_dir.parent / "academic_evaluation_report.json"
    if not report_path.exists():
        logger.warning("No evaluation report found, skipping baseline plot")
        return

    with open(report_path) as f:
        report = json.load(f)

    bc = report.get("experiments", {}).get("baseline_comparison", {})
    if not all(k in bc for k in ["ppo", "scoring", "random"]):
        return

    strategies = ['PPO Agent', 'Scoring', 'Random']
    energies = [
        bc["ppo"]["total_energy_kwh"],
        bc["scoring"]["total_energy_kwh"],
        bc["random"]["total_energy_kwh"]
    ]
    acceptances = [
        bc["ppo"]["acceptance_rate"] * 100,
        bc["scoring"]["acceptance_rate"] * 100,
        bc["random"]["acceptance_rate"] * 100
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ['#27ae60', '#3498db', '#e74c3c']

    bars1 = ax1.bar(strategies, energies, color=colors, edgecolor='black')
    ax1.set_ylabel('Total Energy (kWh)', fontsize=12)
    ax1.set_title('Energy Consumption by Strategy', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, energies):
        ax1.annotate(f'{val:.2f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10)

    bars2 = ax2.bar(strategies, acceptances, color=colors, edgecolor='black')
    ax2.set_ylabel('Acceptance Rate (%)', fontsize=12)
    ax2.set_title('Task Acceptance by Strategy', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, acceptances):
        ax2.annotate(f'{val:.1f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'baseline_comparison.pdf', bbox_inches='tight')
    plt.close()
    logger.info("Saved baseline comparison plot")


def generate_all_plots(results_dir: Path):
    """Generate all plots from experiment results."""
    if not MATPLOTLIB_AVAILABLE:
        logger.error("matplotlib is required for plot generation")
        return

    results = load_results(results_dir)
    output_dir = results_dir / "figures"
    output_dir.mkdir(exist_ok=True)

    logger.info(f"Generating plots to {output_dir}")

    plot_pareto_front(results, output_dir)
    plot_ablation_study(results, output_dir)
    plot_generalization(results, output_dir)
    plot_multi_seed_distribution(results, output_dir)
    plot_baseline_comparison(results, output_dir)

    logger.info("All plots generated successfully")


def main():
    parser = argparse.ArgumentParser(description='Generate plots from experiment results')
    parser.add_argument('--results-dir', type=str, default='results/academic',
                        help='Results directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        results_dir = Path(__file__).parent.parent / args.results_dir

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    generate_all_plots(results_dir)


if __name__ == "__main__":
    main()
