#!/usr/bin/env python3
"""
Academic Evaluation Suite v4 - With Domain Randomization and Utilization Analysis

This script runs a comprehensive evaluation including:
1. Domain randomization training (improved generalization)
2. Multi-seed training for statistical validity
3. Generalization testing across all presets
4. Detailed utilization analysis and visualizations

Usage:
    python scripts/run_academic_evaluation_v4.py --timesteps 100000 --output-dir results/academic_v4
"""

import sys
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from rl.environment import CloudProvisioningEnv, DomainRandomizedEnv, REALISTIC_HW_CONFIGS, DOMAIN_RANDOM_PRESETS
from rl.agent import RLAgent
from rl.trainer import PPOTrainer
from rl.schemas import RLTrainingConfig
from rl.state_encoder import StateEncoder
from scripts.utilization_analysis import UtilizationTracker, UtilizationVisualizer, EpisodeMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_with_domain_randomization(
    output_dir: Path,
    timesteps: int = 100000,
    domain_preset: str = 'mixed_capacity',
    curriculum: bool = False,
    seed: int = 42
) -> Dict[str, Any]:
    """Train agent with domain randomization."""
    logger.info(f"Training with domain randomization: {domain_preset}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    presets = DOMAIN_RANDOM_PRESETS.get(domain_preset, ['medium'])
    logger.info(f"Using presets: {presets}")

    env = DomainRandomizedEnv(
        domain_preset=domain_preset,
        curriculum=curriculum,
        max_steps=2048,
        seed=seed,
        exec_time_noise=0.15,
        energy_noise=0.1
    )

    agent = RLAgent()
    config = RLTrainingConfig(
        total_timesteps=timesteps,
        learning_rate=3e-4,
        gamma=0.99,
        batch_size=64,
        n_epochs=10,
        max_steps_per_episode=2048
    )

    trainer = PPOTrainer(agent, config)
    start_time = time.time()

    results = trainer.train(env, total_timesteps=timesteps)

    training_time = time.time() - start_time

    model_path = output_dir / 'models' / 'domain_random' / f'model_{domain_preset}.pth'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(model_path))

    return {
        'seed': seed,
        'domain_preset': domain_preset,
        'presets': presets,
        'curriculum': curriculum,
        'timesteps': timesteps,
        'training_time_sec': training_time,
        'episodes_completed': trainer.episodes_completed,
        'final_reward': results['episode_rewards'][-1] if results['episode_rewards'] else 0,
        'avg_reward_last_100': np.mean(results['episode_rewards'][-100:]) if results['episode_rewards'] else 0,
        'model_path': str(model_path),
        'policy_losses': results['policy_losses'][-10:] if results['policy_losses'] else [],
        'value_losses': results['value_losses'][-10:] if results['value_losses'] else [],
    }


def evaluate_generalization(
    model_path: str,
    output_dir: Path,
    presets: List[str] = None,
    episodes_per_preset: int = 10,
    max_steps: int = 500
) -> Dict[str, Any]:
    """Evaluate model generalization across presets."""
    if presets is None:
        presets = ['small', 'medium', 'large', 'high_load', 'stress_test', 'enterprise']

    logger.info(f"Evaluating generalization on presets: {presets}")

    agent = RLAgent()
    agent.load(model_path)
    agent.policy.eval()

    results = {}

    for preset in presets:
        logger.info(f"  Testing on: {preset}")

        env = CloudProvisioningEnv(preset=preset, max_steps=max_steps)
        tracker = UtilizationTracker(agent, env, preset)

        metrics = []
        for ep in range(episodes_per_preset):
            episode_metrics = tracker.run_episode(ep, max_steps)
            metrics.append(episode_metrics)

        total_accepted = sum(m.total_accepted for m in metrics)
        total_tasks = sum(m.total_accepted + m.total_rejected for m in metrics)

        results[preset] = {
            'acceptance_rate': total_accepted / max(total_tasks, 1),
            'avg_energy_kwh': np.mean([m.total_energy_kwh for m in metrics]),
            'std_energy_kwh': np.std([m.total_energy_kwh for m in metrics]),
            'energy_per_task': np.mean([
                m.total_energy_kwh / max(m.total_accepted, 1) for m in metrics
            ]),
            'capacity_rejections': np.mean([m.capacity_rejections for m in metrics]),
            'policy_rejections': np.mean([m.policy_rejections for m in metrics]),
            'total_tasks': total_tasks,
            'total_accepted': total_accepted,
        }

    return results


def run_utilization_analysis(
    model_path: str,
    output_dir: Path,
    presets: List[str] = None,
    episodes: int = 5,
    max_steps: int = 500
):
    """Run detailed utilization analysis with visualizations."""
    if presets is None:
        presets = ['small', 'medium', 'large', 'high_load']

    logger.info("Running utilization analysis")

    agent = RLAgent()
    agent.load(model_path)
    agent.policy.eval()

    figures_dir = output_dir / 'figures' / 'utilization'
    figures_dir.mkdir(parents=True, exist_ok=True)

    visualizer = UtilizationVisualizer(figures_dir)
    all_metrics: Dict[str, List[EpisodeMetrics]] = {}

    for preset in presets:
        logger.info(f"  Analyzing: {preset}")

        env = CloudProvisioningEnv(preset=preset, max_steps=max_steps)
        tracker = UtilizationTracker(agent, env, preset)

        episodes_data = []
        for ep_id in range(episodes):
            metrics = tracker.run_episode(ep_id, max_steps)
            episodes_data.append(metrics)
            logger.info(f"    Episode {ep_id}: accepted={metrics.total_accepted}, "
                       f"rejected={metrics.total_rejected}, "
                       f"capacity_rej={metrics.capacity_rejections}, "
                       f"policy_rej={metrics.policy_rejections}")

        all_metrics[preset] = episodes_data

        visualizer.plot_episode_utilization(
            episodes_data[0],
            save_name=f'utilization_{preset}_detailed'
        )

    visualizer.plot_comparative_utilization(all_metrics, save_name='comparative_utilization_dr')
    visualizer.plot_rejection_analysis(all_metrics, save_name='rejection_analysis_dr')

    return all_metrics


def compare_training_methods(
    output_dir: Path,
    timesteps: int = 50000,
    eval_episodes: int = 10
) -> Dict[str, Any]:
    """Compare single-preset vs domain-randomized training."""
    logger.info("Comparing training methods")

    results = {
        'single_preset': {},
        'domain_random': {},
    }

    single_env = CloudProvisioningEnv(preset='medium', max_steps=2048, seed=42)
    single_agent = RLAgent()
    single_config = RLTrainingConfig(total_timesteps=timesteps, max_steps_per_episode=2048)
    single_trainer = PPOTrainer(single_agent, single_config)

    logger.info("Training single-preset model...")
    single_results = single_trainer.train(single_env, total_timesteps=timesteps)

    single_model_path = output_dir / 'models' / 'comparison' / 'single_preset.pth'
    single_model_path.parent.mkdir(parents=True, exist_ok=True)
    single_agent.save(str(single_model_path))

    logger.info("Training domain-random model...")
    dr_env = DomainRandomizedEnv(domain_preset='mixed_capacity', max_steps=2048, seed=42)
    dr_agent = RLAgent()
    dr_config = RLTrainingConfig(total_timesteps=timesteps, max_steps_per_episode=2048)
    dr_trainer = PPOTrainer(dr_agent, dr_config)

    dr_results = dr_trainer.train(dr_env, total_timesteps=timesteps)

    dr_model_path = output_dir / 'models' / 'comparison' / 'domain_random.pth'
    dr_agent.save(str(dr_model_path))

    test_presets = ['small', 'medium', 'large', 'high_load', 'stress_test']

    logger.info("Evaluating single-preset model...")
    results['single_preset'] = evaluate_generalization(
        str(single_model_path), output_dir, test_presets, eval_episodes
    )
    results['single_preset']['training_reward'] = np.mean(single_results['episode_rewards'][-50:]) if single_results['episode_rewards'] else 0

    logger.info("Evaluating domain-random model...")
    results['domain_random'] = evaluate_generalization(
        str(dr_model_path), output_dir, test_presets, eval_episodes
    )
    results['domain_random']['training_reward'] = np.mean(dr_results['episode_rewards'][-50:]) if dr_results['episode_rewards'] else 0

    comparison = {'presets': {}}
    for preset in test_presets:
        single_acc = results['single_preset'][preset]['acceptance_rate']
        dr_acc = results['domain_random'][preset]['acceptance_rate']
        improvement = ((dr_acc - single_acc) / max(single_acc, 0.01)) * 100

        comparison['presets'][preset] = {
            'single_preset_acceptance': single_acc,
            'domain_random_acceptance': dr_acc,
            'improvement_pct': improvement,
        }

    avg_improvement = np.mean([
        comparison['presets'][p]['improvement_pct'] for p in test_presets
    ])
    comparison['avg_improvement_pct'] = avg_improvement

    return {'results': results, 'comparison': comparison}


def main():
    parser = argparse.ArgumentParser(
        description='Academic Evaluation Suite v4',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Training timesteps')
    parser.add_argument('--output-dir', type=str, default='results/academic_v4',
                        help='Output directory')
    parser.add_argument('--domain-preset', type=str, default='mixed_capacity',
                        choices=['mixed_capacity', 'constrained_first', 'full_spectrum', 'production'],
                        help='Domain randomization preset')
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning')
    parser.add_argument('--skip-comparison', action='store_true',
                        help='Skip method comparison (faster)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'models').mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("Academic Evaluation Suite v4")
    logger.info("=" * 70)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Timesteps: {args.timesteps}")
    logger.info(f"Domain preset: {args.domain_preset}")
    logger.info(f"Curriculum: {args.curriculum}")
    logger.info("=" * 70)

    final_results = {
        'config': {
            'timesteps': args.timesteps,
            'domain_preset': args.domain_preset,
            'curriculum': args.curriculum,
            'seed': args.seed,
        }
    }

    logger.info("\n[1/4] Training with domain randomization...")
    training_results = train_with_domain_randomization(
        output_dir,
        timesteps=args.timesteps,
        domain_preset=args.domain_preset,
        curriculum=args.curriculum,
        seed=args.seed
    )
    final_results['training'] = training_results

    with open(output_dir / 'data' / 'training_results.json', 'w') as f:
        json.dump(training_results, f, indent=2)

    logger.info("\n[2/4] Evaluating generalization...")
    model_path = training_results['model_path']
    generalization_results = evaluate_generalization(
        model_path, output_dir,
        episodes_per_preset=10
    )
    final_results['generalization'] = generalization_results

    with open(output_dir / 'data' / 'generalization_results.json', 'w') as f:
        json.dump(generalization_results, f, indent=2)

    logger.info("\n[3/4] Running utilization analysis...")
    run_utilization_analysis(
        model_path, output_dir,
        presets=['small', 'medium', 'large', 'high_load'],
        episodes=3
    )

    if not args.skip_comparison:
        logger.info("\n[4/4] Comparing training methods...")
        comparison_results = compare_training_methods(
            output_dir,
            timesteps=min(args.timesteps, 50000),
            eval_episodes=5
        )
        final_results['comparison'] = comparison_results

        with open(output_dir / 'data' / 'comparison_results.json', 'w') as f:
            json.dump(comparison_results, f, indent=2, default=float)
    else:
        logger.info("\n[4/4] Skipping method comparison")

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=float)

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)

    logger.info("\nGeneralization Results:")
    for preset, metrics in generalization_results.items():
        logger.info(f"  {preset:15s}: acceptance={metrics['acceptance_rate']:.3f}, "
                   f"energy/task={metrics['energy_per_task']:.6f}")

    if 'comparison' in final_results:
        logger.info("\nTraining Method Comparison:")
        logger.info(f"  Average improvement with domain randomization: "
                   f"{final_results['comparison']['comparison']['avg_improvement_pct']:.1f}%")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
