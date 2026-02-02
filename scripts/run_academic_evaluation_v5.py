#!/usr/bin/env python3
"""
Academic Evaluation Suite v5 - Unified Training and Evaluation Pipeline

Combines distributed training with comprehensive evaluation and output generation.

Features:
- Distributed multi-GPU training (torchrun compatible)
- Domain randomization with scarcity-aware rewards
- Multi-preset generalization evaluation
- Utilization analysis with GPU tracking
- Comprehensive output: models, logs, figures, JSON, LaTeX

Usage:
    # Single GPU
    python scripts/run_academic_evaluation_v5.py --timesteps 100000 --output-dir results/academic_v5

    # Multi-GPU (recommended)
    torchrun --nproc_per_node=4 scripts/run_academic_evaluation_v5.py \
        --timesteps 100000 --output-dir results/academic_v5

    # Skip training (evaluate existing model)
    python scripts/run_academic_evaluation_v5.py --skip-training \
        --model models/rl/ppo/model_distributed.pth --output-dir results/academic_v5

Output Structure:
    results/academic_v5/
    ├── models/
    │   └── model_v5.pth
    ├── logs/
    │   └── train_logs.txt
    ├── data/
    │   ├── training_results.json
    │   ├── generalization_results.json
    │   └── utilization_summary.json
    ├── figures/
    │   ├── training_curves.png
    │   ├── {preset}_utilization.png
    │   ├── comparative_utilization.png
    │   └── rejection_analysis.png
    ├── latex/
    │   ├── generalization_table.tex
    │   └── comparison_table.tex
    └── evaluation_report.json
"""

import sys
import argparse
import json
import logging
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", message="Can't initialize NVML")

import torch
import torch.distributed as dist

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from rl.environment import CloudProvisioningEnv, REALISTIC_HW_CONFIGS
from rl.agent import RLAgent
from rl.state_encoder import StateEncoder
from rl.distributed_trainer import DistributedPPOTrainer, cleanup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EVAL_PRESETS = ['small', 'medium', 'large', 'high_load', 'stress_test']


class OutputManager:
    def __init__(self, output_dir: str):
        self.base_dir = Path(output_dir)
        self.models_dir = self.base_dir / 'models'
        self.logs_dir = self.base_dir / 'logs'
        self.data_dir = self.base_dir / 'data'
        self.figures_dir = self.base_dir / 'figures'
        self.latex_dir = self.base_dir / 'latex'

        for d in [self.models_dir, self.logs_dir, self.data_dir,
                  self.figures_dir, self.latex_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_json(self, data: Dict, filename: str, subdir: str = 'data'):
        target_dir = getattr(self, f'{subdir}_dir', self.data_dir)
        path = target_dir / filename
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=float)
        logger.info(f"Saved: {path}")
        return path

    def save_latex_table(self, content: str, filename: str):
        path = self.latex_dir / filename
        with open(path, 'w') as f:
            f.write(content)
        logger.info(f"Saved LaTeX: {path}")
        return path


def is_main_process() -> bool:
    return int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0))) == 0


def train_distributed(
    output: OutputManager,
    timesteps: int,
    domain_preset: str,
    curriculum: bool,
    num_envs: int,
    rollout_steps: int,
    scarcity_aware: bool,
    scarcity_rejection_scale: float,
    scarcity_acceptance_scale: float,
    lr: float,
    batch_size: int,
    epochs: int,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    use_capacity_features: bool = False,
) -> Dict[str, Any]:
    if is_main_process():
        logger.info("=" * 70)
        logger.info("PHASE 1: Distributed Training")
        logger.info("=" * 70)
        logger.info(f"  Timesteps: {timesteps:,}")
        logger.info(f"  Domain preset: {domain_preset}")
        logger.info(f"  Scarcity-aware: {scarcity_aware}")
        logger.info(f"  Capacity features (v3): {use_capacity_features}")
        logger.info(f"  Num envs: {num_envs}")

    reward_config = {
        'scarcity_aware': scarcity_aware,
        'rejection_penalty': 0.8,
        'scarcity_rejection_scale': scarcity_rejection_scale,
        'scarcity_acceptance_scale': scarcity_acceptance_scale,
    }

    trainer = DistributedPPOTrainer(
        learning_rate=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        n_epochs=epochs,
        batch_size=batch_size,
        use_capacity_features=use_capacity_features,
    )

    model_path = str(output.models_dir / 'model_v5.pth')

    start_time = time.time()
    results = trainer.train(
        total_timesteps=timesteps,
        num_envs=num_envs,
        rollout_steps=rollout_steps,
        env_preset='medium',
        log_interval=5000,
        save_path=model_path,
        domain_randomization=True,
        domain_preset=domain_preset,
        curriculum=curriculum,
        reward_config=reward_config,
        checkpoint_interval=50000,
    )
    training_time = time.time() - start_time

    cleanup()

    training_results = {
        'timesteps': results['total_timesteps'],
        'episodes': results['episodes'],
        'avg_reward': results['avg_reward'],
        'training_time_sec': training_time,
        'fps': results['fps'],
        'model_path': model_path,
        'config': {
            'domain_preset': domain_preset,
            'curriculum': curriculum,
            'scarcity_aware': scarcity_aware,
            'scarcity_rejection_scale': scarcity_rejection_scale,
            'scarcity_acceptance_scale': scarcity_acceptance_scale,
            'use_capacity_features': use_capacity_features,
            'lr': lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'gamma': gamma,
            'num_envs': num_envs,
        }
    }

    if is_main_process():
        output.save_json(training_results, 'training_results.json')

        with open(output.logs_dir / 'train_logs.txt', 'w') as f:
            f.write(f"Training completed: {datetime.now().isoformat()}\n")
            f.write(f"Timesteps: {results['total_timesteps']}\n")
            f.write(f"Episodes: {results['episodes']}\n")
            f.write(f"Avg Reward: {results['avg_reward']:.4f}\n")
            f.write(f"Training Time: {training_time:.1f}s\n")
            f.write(f"Throughput: {results['fps']:.0f} steps/sec\n")

    return training_results


def evaluate_generalization(
    model_path: str,
    output: OutputManager,
    presets: List[str],
    episodes_per_preset: int = 10,
    max_steps: int = 500,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("PHASE 2: Generalization Evaluation")
    logger.info("=" * 70)

    agent = RLAgent()
    agent.load(model_path)
    agent.policy.eval()

    from scripts.utilization_analysis import UtilizationTracker

    results = {}

    for preset in presets:
        logger.info(f"  Evaluating: {preset}")

        env = CloudProvisioningEnv(preset=preset, max_steps=max_steps, seed=42)
        tracker = UtilizationTracker(agent, env, preset)

        metrics_list = []
        for ep in range(episodes_per_preset):
            metrics = tracker.run_episode(ep, max_steps)
            metrics_list.append(metrics)

        total_accepted = sum(m.total_accepted for m in metrics_list)
        total_rejected = sum(m.total_rejected for m in metrics_list)
        total_tasks = total_accepted + total_rejected

        capacity_rej = sum(m.capacity_rejections for m in metrics_list)
        policy_rej = sum(m.policy_rejections for m in metrics_list)

        results[preset] = {
            'acceptance_rate': total_accepted / max(total_tasks, 1),
            'avg_energy_kwh': float(np.mean([m.total_energy_kwh for m in metrics_list])),
            'std_energy_kwh': float(np.std([m.total_energy_kwh for m in metrics_list])),
            'energy_per_task': float(np.mean([
                m.total_energy_kwh / max(m.total_accepted, 1) for m in metrics_list
            ])),
            'total_tasks': total_tasks,
            'total_accepted': total_accepted,
            'total_rejected': total_rejected,
            'capacity_rejections': capacity_rej,
            'policy_rejections': policy_rej,
            'capacity_rejection_ratio': capacity_rej / max(total_rejected, 1),
            'policy_rejection_pct': policy_rej / max(total_rejected, 1) * 100,
        }

        logger.info(f"    Acceptance: {results[preset]['acceptance_rate']:.3f}, "
                   f"Policy Rej%: {results[preset]['policy_rejection_pct']:.1f}%")

    output.save_json(results, 'generalization_results.json')
    return results


def run_utilization_analysis(
    model_path: str,
    output: OutputManager,
    presets: List[str],
    episodes: int = 5,
    max_steps: int = 500,
) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("PHASE 3: Utilization Analysis")
    logger.info("=" * 70)

    from scripts.utilization_analysis import (
        UtilizationTracker, UtilizationVisualizer, run_analysis
    )

    _, summary = run_analysis(
        model_path=model_path,
        presets=presets,
        output_dir=str(output.figures_dir),
        n_episodes=episodes,
        max_steps=max_steps,
        seed=42
    )

    summary_path = output.figures_dir / 'analysis_summary.json'
    if summary_path.exists():
        import shutil
        shutil.move(str(summary_path), str(output.data_dir / 'utilization_summary.json'))

    return summary


def generate_latex_tables(
    output: OutputManager,
    generalization: Dict[str, Any],
    v4_baseline: Optional[Dict[str, Any]] = None,
):
    logger.info("=" * 70)
    logger.info("PHASE 4: LaTeX Table Generation")
    logger.info("=" * 70)

    gen_table = r"""\begin{table}[h]
\centering
\caption{Generalization Results Across Environment Presets}
\label{tab:generalization}
\begin{tabular}{lccccc}
\toprule
Preset & Acceptance & Energy/Task & Policy Rej\% & Capacity Ratio \\
\midrule
"""
    for preset in EVAL_PRESETS:
        if preset in generalization:
            m = generalization[preset]
            gen_table += f"{preset.replace('_', ' ').title()} & "
            gen_table += f"{m['acceptance_rate']:.3f} & "
            gen_table += f"{m['energy_per_task']:.6f} & "
            gen_table += f"{m['policy_rejection_pct']:.1f}\\% & "
            gen_table += f"{m['capacity_rejection_ratio']:.3f} \\\\\n"

    gen_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
    output.save_latex_table(gen_table, 'generalization_table.tex')

    if v4_baseline:
        comp_table = r"""\begin{table}[h]
\centering
\caption{V5 vs V4 Comparison}
\label{tab:comparison}
\begin{tabular}{lcccc}
\toprule
Preset & V4 Accept & V5 Accept & V4 Policy Rej\% & V5 Policy Rej\% \\
\midrule
"""
        for preset in EVAL_PRESETS:
            if preset in generalization and preset in v4_baseline:
                v4 = v4_baseline[preset]
                v5 = generalization[preset]

                v4_acc = v4.get('avg_acceptance_rate', v4.get('acceptance_rate', 0))
                v5_acc = v5['acceptance_rate']

                v4_policy = v4.get('avg_policy_rejections', 0)
                v4_total_rej = v4_policy + v4.get('avg_capacity_rejections', 0)
                v4_policy_pct = (v4_policy / max(v4_total_rej, 1)) * 100 if v4_total_rej > 0 else 0

                v5_policy_pct = v5['policy_rejection_pct']

                comp_table += f"{preset.replace('_', ' ').title()} & "
                comp_table += f"{v4_acc:.3f} & {v5_acc:.3f} & "
                comp_table += f"{v4_policy_pct:.1f}\\% & {v5_policy_pct:.1f}\\% \\\\\n"

        comp_table += r"""\bottomrule
\end{tabular}
\end{table}
"""
        output.save_latex_table(comp_table, 'comparison_table.tex')


def plot_training_curves(output: OutputManager, training_results: Dict[str, Any]):
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping training curves")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_title('Training Summary')
    ax.text(0.5, 0.5,
            f"Timesteps: {training_results['timesteps']:,}\n"
            f"Episodes: {training_results['episodes']}\n"
            f"Avg Reward: {training_results['avg_reward']:.2f}\n"
            f"Time: {training_results['training_time_sec']:.1f}s\n"
            f"Throughput: {training_results['fps']:.0f} steps/sec",
            transform=ax.transAxes, fontsize=14, verticalalignment='center',
            horizontalalignment='center', fontfamily='monospace')
    ax.axis('off')

    fig.savefig(output.figures_dir / 'training_summary.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_final_report(
    output: OutputManager,
    training_results: Optional[Dict[str, Any]],
    generalization_results: Dict[str, Any],
    utilization_summary: Dict[str, Any],
    v4_baseline: Optional[Dict[str, Any]] = None,
):
    logger.info("=" * 70)
    logger.info("PHASE 5: Final Report Generation")
    logger.info("=" * 70)

    report = {
        'version': 'v5',
        'timestamp': datetime.now().isoformat(),
        'training': training_results,
        'generalization': generalization_results,
        'utilization': utilization_summary,
    }

    if v4_baseline:
        improvements = {}
        for preset in EVAL_PRESETS:
            if preset in generalization_results and preset in v4_baseline:
                v4_acc = v4_baseline[preset].get('avg_acceptance_rate',
                         v4_baseline[preset].get('acceptance_rate', 0))
                v5_acc = generalization_results[preset]['acceptance_rate']
                improvements[preset] = {
                    'v4_acceptance': v4_acc,
                    'v5_acceptance': v5_acc,
                    'absolute_improvement': v5_acc - v4_acc,
                    'relative_improvement_pct': ((v5_acc - v4_acc) / max(v4_acc, 0.01)) * 100,
                }
        report['v4_comparison'] = improvements

        avg_improvement = np.mean([v['relative_improvement_pct'] for v in improvements.values()])
        report['avg_improvement_pct'] = avg_improvement

    output.save_json(report, 'evaluation_report.json', 'base')

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\nGeneralization Results:")
    for preset in EVAL_PRESETS:
        if preset in generalization_results:
            m = generalization_results[preset]
            logger.info(f"  {preset:15s}: accept={m['acceptance_rate']:.3f}, "
                       f"policy_rej={m['policy_rejection_pct']:.1f}%")

    if v4_baseline and 'avg_improvement_pct' in report:
        logger.info(f"\nAverage improvement over v4: {report['avg_improvement_pct']:.1f}%")

    logger.info(f"\nOutputs saved to: {output.base_dir}")
    logger.info("=" * 70)

    return report


def load_v4_baseline(v4_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not v4_path:
        default_paths = [
            'results/academic_v4/utilization_analysis/analysis_summary.json',
            'results/academic_v4/data/generalization_results.json',
        ]
        for p in default_paths:
            if Path(p).exists():
                v4_path = p
                break

    if v4_path and Path(v4_path).exists():
        logger.info(f"Loading v4 baseline from: {v4_path}")
        with open(v4_path) as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Academic Evaluation Suite v5 - Unified Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Training timesteps (default: 100000)')
    parser.add_argument('--output-dir', type=str, default='results/academic_v5',
                        help='Output directory')
    parser.add_argument('--domain-preset', type=str, default='constrained_first',
                        choices=['mixed_capacity', 'constrained_first', 'full_spectrum', 'production'],
                        help='Domain randomization preset (default: constrained_first)')
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning')

    parser.add_argument('--scarcity-aware', action='store_true', default=True,
                        help='Enable scarcity-aware rewards (default: True)')
    parser.add_argument('--scarcity-rejection-scale', type=float, default=1.5,
                        help='Rejection penalty scale when resources available')
    parser.add_argument('--scarcity-acceptance-scale', type=float, default=2.0,
                        help='Acceptance bonus scale under scarcity')

    parser.add_argument('--num-envs', type=int, default=8,
                        help='Parallel environments per GPU')
    parser.add_argument('--rollout-steps', type=int, default=256,
                        help='Steps per rollout')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='PPO epochs')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--clip-range', type=float, default=0.2, help='PPO clip range')

    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Episodes per preset for evaluation')
    parser.add_argument('--eval-steps', type=int, default=500,
                        help='Max steps per evaluation episode')
    parser.add_argument('--util-episodes', type=int, default=5,
                        help='Episodes for utilization analysis')

    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, evaluate existing model')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to existing model (for --skip-training)')
    parser.add_argument('--v4-baseline', type=str, default=None,
                        help='Path to v4 baseline JSON for comparison')
    parser.add_argument('--use-capacity-features', action='store_true',
                        help='Enable v3 capacity scale features (addresses scale blindness)')

    args = parser.parse_args()

    output = OutputManager(args.output_dir)

    if is_main_process():
        logger.info("=" * 70)
        logger.info("Academic Evaluation Suite v5")
        logger.info("=" * 70)
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Timesteps: {args.timesteps}")
        logger.info(f"Domain preset: {args.domain_preset}")
        logger.info(f"Scarcity-aware: {args.scarcity_aware}")
        logger.info(f"Capacity features (v3): {args.use_capacity_features}")
        logger.info("=" * 70)

    training_results = None

    if args.skip_training:
        if not args.model:
            logger.error("--model required when using --skip-training")
            sys.exit(1)
        model_path = args.model
        logger.info(f"Skipping training, using model: {model_path}")
    else:
        training_results = train_distributed(
            output=output,
            timesteps=args.timesteps,
            domain_preset=args.domain_preset,
            curriculum=args.curriculum,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            scarcity_aware=args.scarcity_aware,
            scarcity_rejection_scale=args.scarcity_rejection_scale,
            scarcity_acceptance_scale=args.scarcity_acceptance_scale,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            use_capacity_features=args.use_capacity_features,
        )
        model_path = training_results['model_path']

    if not is_main_process():
        return

    if training_results:
        plot_training_curves(output, training_results)

    generalization_results = evaluate_generalization(
        model_path=model_path,
        output=output,
        presets=EVAL_PRESETS,
        episodes_per_preset=args.eval_episodes,
        max_steps=args.eval_steps,
    )

    utilization_summary = run_utilization_analysis(
        model_path=model_path,
        output=output,
        presets=EVAL_PRESETS,
        episodes=args.util_episodes,
        max_steps=args.eval_steps,
    )

    v4_baseline = load_v4_baseline(args.v4_baseline)

    generate_latex_tables(output, generalization_results, v4_baseline)

    generate_final_report(
        output=output,
        training_results=training_results,
        generalization_results=generalization_results,
        utilization_summary=utilization_summary,
        v4_baseline=v4_baseline,
    )


if __name__ == '__main__':
    main()
