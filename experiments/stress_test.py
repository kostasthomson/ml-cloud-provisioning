#!/usr/bin/env python3
"""
Stress Test for High-Load Scenarios.

Tests the RL agent under increasingly difficult conditions to prove
it can make intelligent tradeoffs when resources are constrained.

Usage:
    python stress_test.py [--presets high_load,stress_test]
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.environment import CloudProvisioningEnv, REALISTIC_HW_CONFIGS
from rl.agent import RLAgent
from rl.trainer import PPOTrainer
from rl.schemas import RLTrainingConfig
from entities.allocator.scoring_allocator import ScoringAllocator
from scripts.benchmark_performance import run_performance_study
from experiments.config import ExperimentConfig, setup_experiment_logging

logger = None


@dataclass
class StressTestResult:
    """Results from a stress test scenario."""
    preset: str
    num_hw_types: int
    ppo_energy_per_task: float
    ppo_acceptance_rate: float
    ppo_sla_compliance: float
    scoring_energy_per_task: float
    scoring_acceptance_rate: float
    scoring_sla_compliance: float
    ppo_wins_energy: bool
    ppo_wins_acceptance: bool
    energy_improvement_pct: float
    acceptance_improvement_pct: float


def evaluate_agent_on_preset(
    agent: RLAgent,
    preset: str,
    config: ExperimentConfig
) -> Dict[str, float]:
    """Evaluate trained agent on a specific preset."""
    env = CloudProvisioningEnv(
        preset=preset,
        episode_length=config.episode_length,
        exec_time_noise=config.exec_time_noise,
        energy_noise=config.energy_noise
    )

    metrics = run_performance_study(
        env, "ppo", model=agent, num_episodes=config.evaluation_episodes
    )

    return {
        "energy_per_task": metrics.total_energy_kwh / max(metrics.total_accepted, 1),
        "acceptance_rate": metrics.acceptance_rate,
        "sla_compliance": metrics.sla_compliance_rate,
        "total_energy": metrics.total_energy_kwh,
        "total_accepted": metrics.total_accepted,
        "total_tasks": metrics.total_tasks
    }


def evaluate_scoring_on_preset(
    preset: str,
    config: ExperimentConfig
) -> Dict[str, float]:
    """Evaluate scoring allocator on a specific preset."""
    env = CloudProvisioningEnv(
        preset=preset,
        episode_length=config.episode_length,
        exec_time_noise=config.exec_time_noise,
        energy_noise=config.energy_noise
    )
    allocator = ScoringAllocator()

    metrics = run_performance_study(
        env, "scoring", allocator=allocator, num_episodes=config.evaluation_episodes
    )

    return {
        "energy_per_task": metrics.total_energy_kwh / max(metrics.total_accepted, 1),
        "acceptance_rate": metrics.acceptance_rate,
        "sla_compliance": metrics.sla_compliance_rate,
        "total_energy": metrics.total_energy_kwh,
        "total_accepted": metrics.total_accepted,
        "total_tasks": metrics.total_tasks
    }


def _save_stress_checkpoint(config: ExperimentConfig, results: List[StressTestResult], failed_presets: List[str]):
    """Save intermediate checkpoint."""
    checkpoint = {
        "experiment": "stress_test",
        "timestamp": datetime.now().isoformat(),
        "status": "in_progress",
        "completed_presets": [r.preset for r in results],
        "failed_presets": failed_presets,
        "results": [asdict(r) for r in results]
    }
    checkpoint_path = config.results_dir / "data" / "stress_test_checkpoint.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    return checkpoint_path


def run_stress_test(
    config: ExperimentConfig,
    presets: List[str] = None,
    model_path: str = None
) -> Dict[str, Any]:
    """Run stress test comparing PPO vs baselines under high load."""
    global logger
    if logger is None:
        logger = setup_experiment_logging(config, "stress_test")

    presets = presets or config.stress_test_presets
    logger.info(f"Running stress test on presets: {presets}")

    if model_path:
        agent = RLAgent(model_path=model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.info("Training new agent on medium preset...")
        train_env = CloudProvisioningEnv(
            preset="medium",
            episode_length=config.episode_length,
            max_steps=config.training_timesteps,
            exec_time_noise=config.exec_time_noise,
            energy_noise=config.energy_noise
        )

        agent = RLAgent(device="auto", embed_dim=64)
        training_config = RLTrainingConfig(
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            batch_size=config.batch_size,
            n_epochs=config.ppo_epochs,
            clip_range=config.clip_range,
            total_timesteps=config.training_timesteps
        )

        trainer = PPOTrainer(agent, training_config)
        trainer.train(train_env, total_timesteps=config.training_timesteps)

        model_dir = config.results_dir / "models" / "stress_test"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(model_dir / "model_stress_test.pth")
        agent.save(model_path)
        logger.info(f"Model saved to {model_path}")

    results = []
    failed_presets = []

    for i, preset in enumerate(presets):
        logger.info(f"[{i+1}/{len(presets)}] Testing on '{preset}'...")
        try:
            ppo_metrics = evaluate_agent_on_preset(agent, preset, config)
            scoring_metrics = evaluate_scoring_on_preset(preset, config)

            energy_improvement = (
                (scoring_metrics["energy_per_task"] - ppo_metrics["energy_per_task"])
                / max(scoring_metrics["energy_per_task"], 1e-9) * 100
            )
            acceptance_improvement = (
                (ppo_metrics["acceptance_rate"] - scoring_metrics["acceptance_rate"])
                / max(scoring_metrics["acceptance_rate"], 1e-9) * 100
            )

            result = StressTestResult(
                preset=preset,
                num_hw_types=len(REALISTIC_HW_CONFIGS.get(preset, [])),
                ppo_energy_per_task=ppo_metrics["energy_per_task"],
                ppo_acceptance_rate=ppo_metrics["acceptance_rate"],
                ppo_sla_compliance=ppo_metrics["sla_compliance"],
                scoring_energy_per_task=scoring_metrics["energy_per_task"],
                scoring_acceptance_rate=scoring_metrics["acceptance_rate"],
                scoring_sla_compliance=scoring_metrics["sla_compliance"],
                ppo_wins_energy=ppo_metrics["energy_per_task"] < scoring_metrics["energy_per_task"],
                ppo_wins_acceptance=ppo_metrics["acceptance_rate"] > scoring_metrics["acceptance_rate"],
                energy_improvement_pct=energy_improvement,
                acceptance_improvement_pct=acceptance_improvement
            )
            results.append(result)

            logger.info(f"  PPO:     energy={ppo_metrics['energy_per_task']:.6f}, "
                       f"accept={ppo_metrics['acceptance_rate']:.2%}")
            logger.info(f"  Scoring: energy={scoring_metrics['energy_per_task']:.6f}, "
                       f"accept={scoring_metrics['acceptance_rate']:.2%}")
            logger.info(f"  PPO wins energy: {result.ppo_wins_energy}, "
                       f"PPO wins acceptance: {result.ppo_wins_acceptance}")

            _save_stress_checkpoint(config, results, failed_presets)

        except Exception as e:
            logger.error(f"Preset '{preset}' FAILED: {e}")
            failed_presets.append(preset)
            _save_stress_checkpoint(config, results, failed_presets)
            continue

    if not results:
        logger.error("All presets failed!")
        return {"error": "All presets failed", "failed_presets": failed_presets}

    ppo_wins_count = sum(1 for r in results if r.ppo_wins_energy)
    avg_energy_improvement = np.mean([r.energy_improvement_pct for r in results])
    avg_acceptance_improvement = np.mean([r.acceptance_improvement_pct for r in results])

    output = {
        "experiment": "stress_test",
        "timestamp": datetime.now().isoformat(),
        "status": "completed" if not failed_presets else "partial",
        "config": {
            "presets": presets,
            "timesteps": config.training_timesteps,
            "evaluation_episodes": config.evaluation_episodes
        },
        "completed_presets": [r.preset for r in results],
        "failed_presets": failed_presets,
        "results": [asdict(r) for r in results],
        "summary": {
            "ppo_wins_energy": ppo_wins_count,
            "total_presets": len(results),
            "win_rate": ppo_wins_count / len(results) if results else 0,
            "avg_energy_improvement_pct": avg_energy_improvement,
            "avg_acceptance_improvement_pct": avg_acceptance_improvement
        },
        "model_path": model_path
    }

    output_path = config.results_dir / "data" / "stress_test_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    print_stress_summary(results, output["summary"])

    return output


def print_stress_summary(results: List[StressTestResult], summary: Dict):
    """Print formatted summary."""
    print("\n" + "=" * 100)
    print("STRESS TEST RESULTS - PPO vs Scoring Allocator")
    print("=" * 100)
    print(f"\n{'Preset':<15} {'HW Types':<10} {'PPO Energy':<12} {'Scoring Energy':<15} {'PPO Wins?':<10} {'Improvement'}")
    print("-" * 100)

    for r in results:
        win_str = "YES" if r.ppo_wins_energy else "NO"
        print(f"{r.preset:<15} {r.num_hw_types:<10} {r.ppo_energy_per_task:<12.6f} "
              f"{r.scoring_energy_per_task:<15.6f} {win_str:<10} {r.energy_improvement_pct:+.1f}%")

    print("=" * 100)
    print(f"\nSummary:")
    print(f"  PPO wins on energy: {summary['ppo_wins_energy']}/{summary['total_presets']} "
          f"({summary['win_rate']:.0%})")
    print(f"  Avg energy improvement: {summary['avg_energy_improvement_pct']:+.1f}%")
    print(f"  Avg acceptance improvement: {summary['avg_acceptance_improvement_pct']:+.1f}%")
    print("=" * 100)


def main():
    global logger
    parser = argparse.ArgumentParser(description='Stress test under high load')
    parser.add_argument('--presets', type=str, default=None,
                        help='Comma-separated presets to test')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pre-trained model')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Training timesteps')
    parser.add_argument('--eval-episodes', type=int, default=None,
                        help='Evaluation episodes')
    args = parser.parse_args()

    config = ExperimentConfig()
    logger = setup_experiment_logging(config, "stress_test")

    if args.timesteps:
        config.training_timesteps = args.timesteps
    if args.eval_episodes:
        config.evaluation_episodes = args.eval_episodes

    presets = None
    if args.presets:
        presets = [p.strip() for p in args.presets.split(',')]

    run_stress_test(config, presets=presets, model_path=args.model)


if __name__ == "__main__":
    main()
