#!/usr/bin/env python3
"""
Pareto Front Analysis for Energy-Acceptance Tradeoff.

Trains models with different energy weights to map out the tradeoff curve
between energy efficiency and task acceptance rate.

Usage:
    python pareto_analysis.py [--weights 0.5,0.6,0.7,0.8,0.9]
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.environment import CloudProvisioningEnv
from rl.agent import RLAgent
from rl.trainer import PPOTrainer
from rl.reward import RewardCalculator
from rl.schemas import RLTrainingConfig
from experiments.config import ExperimentConfig, setup_experiment_logging

logger = None


@dataclass
class ParetoPoint:
    """A single point on the Pareto front."""
    energy_weight: float
    avg_energy_per_task: float
    acceptance_rate: float
    sla_compliance_rate: float
    avg_reward: float
    total_energy: float
    total_accepted: int
    total_tasks: int
    model_path: str


def train_with_energy_weight(
    energy_weight: float,
    config: ExperimentConfig,
    save_dir: Path,
    seed: int = 42
) -> str:
    """Train a model with specific energy weight."""
    logger.info(f"Training with energy_weight={energy_weight:.2f}...")

    np.random.seed(seed)

    env = CloudProvisioningEnv(
        preset=config.env_preset,
        episode_length=config.episode_length,
        max_steps=config.training_timesteps,
        seed=seed,
        exec_time_noise=config.exec_time_noise,
        energy_noise=config.energy_noise
    )

    env.reward_calculator = RewardCalculator(
        energy_weight=energy_weight,
        sla_weight=1.0 - energy_weight - 0.05,
        rejection_penalty=0.3
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
    trainer.train(env, total_timesteps=config.training_timesteps)

    model_path = save_dir / f"model_energy_weight_{energy_weight:.2f}.pth"
    agent.save(str(model_path))

    return str(model_path)


def evaluate_pareto_point(
    model_path: str,
    energy_weight: float,
    config: ExperimentConfig
) -> ParetoPoint:
    """Evaluate a trained model for Pareto analysis."""
    logger.info(f"Evaluating model with energy_weight={energy_weight:.2f}...")

    agent = RLAgent(model_path=model_path)

    env = CloudProvisioningEnv(
        preset=config.env_preset,
        episode_length=config.episode_length,
        max_steps=config.episode_length,
        exec_time_noise=config.exec_time_noise,
        energy_noise=config.energy_noise
    )

    total_energy = 0.0
    total_accepted = 0
    total_tasks = 0
    total_sla_violations = 0
    total_reward = 0.0

    for episode in range(config.evaluation_episodes):
        state, _ = env.reset(seed=episode)
        done = False

        while not done:
            action_result, _, _ = agent.predict(state, deterministic=True)
            action = action_result.action

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            total_tasks += 1

            if info.get('accepted', False):
                total_accepted += 1
                total_energy += info.get('energy', 0.0)

                if state.task.deadline is not None:
                    exec_time = info.get('exec_time', 0.0)
                    if exec_time > state.task.deadline:
                        total_sla_violations += 1

            state = next_state

    return ParetoPoint(
        energy_weight=energy_weight,
        avg_energy_per_task=total_energy / max(total_accepted, 1),
        acceptance_rate=total_accepted / max(total_tasks, 1),
        sla_compliance_rate=1.0 - (total_sla_violations / max(total_accepted, 1)),
        avg_reward=total_reward / config.evaluation_episodes,
        total_energy=total_energy,
        total_accepted=total_accepted,
        total_tasks=total_tasks,
        model_path=model_path
    )


def compute_pareto_frontier(points: List[ParetoPoint]) -> List[ParetoPoint]:
    """Extract Pareto-optimal points (minimizing energy, maximizing acceptance)."""
    pareto_points = []

    for point in points:
        is_dominated = False
        for other in points:
            if other == point:
                continue
            if (other.avg_energy_per_task <= point.avg_energy_per_task and
                other.acceptance_rate >= point.acceptance_rate and
                (other.avg_energy_per_task < point.avg_energy_per_task or
                 other.acceptance_rate > point.acceptance_rate)):
                is_dominated = True
                break

        if not is_dominated:
            pareto_points.append(point)

    pareto_points.sort(key=lambda p: p.acceptance_rate)
    return pareto_points


def _save_pareto_checkpoint(config: ExperimentConfig, pareto_points: List[ParetoPoint], failed_weights: List[float]):
    """Save intermediate checkpoint to preserve progress."""
    checkpoint = {
        "experiment": "pareto_analysis",
        "timestamp": datetime.now().isoformat(),
        "status": "in_progress",
        "completed_weights": [p.energy_weight for p in pareto_points],
        "failed_weights": failed_weights,
        "all_points": [asdict(p) for p in pareto_points]
    }
    checkpoint_path = config.results_dir / "data" / "pareto_checkpoint.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    return checkpoint_path


def run_pareto_analysis(
    config: ExperimentConfig,
    energy_weights: List[float] = None
) -> Dict[str, Any]:
    """Run complete Pareto front analysis."""
    global logger
    if logger is None:
        logger = setup_experiment_logging(config, "pareto_analysis")

    weights = energy_weights or config.pareto_energy_weights
    logger.info(f"Running Pareto analysis with energy weights: {weights}")

    save_dir = config.results_dir / "models" / "pareto"
    save_dir.mkdir(parents=True, exist_ok=True)

    pareto_points = []
    failed_weights = []

    for i, weight in enumerate(weights):
        logger.info(f"[{i+1}/{len(weights)}] Training with energy_weight={weight:.2f}...")
        try:
            model_path = train_with_energy_weight(weight, config, save_dir)
            point = evaluate_pareto_point(model_path, weight, config)
            pareto_points.append(point)
            logger.info(
                f"  Energy weight {weight:.2f}: "
                f"energy={point.avg_energy_per_task:.6f}, "
                f"acceptance={point.acceptance_rate:.2%}"
            )
            _save_pareto_checkpoint(config, pareto_points, failed_weights)

        except Exception as e:
            logger.error(f"Energy weight {weight:.2f} FAILED: {e}")
            failed_weights.append(weight)
            _save_pareto_checkpoint(config, pareto_points, failed_weights)
            continue

    if not pareto_points:
        logger.error("All weights failed! No results to compute.")
        return {"error": "All weights failed", "failed_weights": failed_weights}

    pareto_frontier = compute_pareto_frontier(pareto_points)

    results = {
        "experiment": "pareto_analysis",
        "timestamp": datetime.now().isoformat(),
        "status": "completed" if not failed_weights else "partial",
        "config": {
            "energy_weights": weights,
            "timesteps": config.training_timesteps,
            "evaluation_episodes": config.evaluation_episodes,
            "env_preset": config.env_preset
        },
        "completed_weights": [p.energy_weight for p in pareto_points],
        "failed_weights": failed_weights,
        "all_points": [asdict(p) for p in pareto_points],
        "pareto_frontier": [asdict(p) for p in pareto_frontier],
        "summary": {
            "num_points": len(pareto_points),
            "num_pareto_optimal": len(pareto_frontier),
            "energy_range": {
                "min": min(p.avg_energy_per_task for p in pareto_points),
                "max": max(p.avg_energy_per_task for p in pareto_points)
            },
            "acceptance_range": {
                "min": min(p.acceptance_rate for p in pareto_points),
                "max": max(p.acceptance_rate for p in pareto_points)
            }
        }
    }

    output_path = config.results_dir / "data" / "pareto_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    try:
        generate_pareto_csv(pareto_points, config.results_dir / "data" / "pareto_data.csv")
    except Exception as e:
        logger.warning(f"CSV generation failed: {e}")

    if failed_weights:
        logger.warning(f"Completed with {len(failed_weights)} failed weights: {failed_weights}")

    print_pareto_summary(pareto_points, pareto_frontier)

    return results


def generate_pareto_csv(points: List[ParetoPoint], output_path: Path):
    """Generate CSV file for Pareto data (for plotting)."""
    import csv

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'energy_weight', 'avg_energy_per_task', 'acceptance_rate',
            'sla_compliance_rate', 'avg_reward', 'total_energy',
            'total_accepted', 'total_tasks'
        ])
        for p in points:
            writer.writerow([
                p.energy_weight, p.avg_energy_per_task, p.acceptance_rate,
                p.sla_compliance_rate, p.avg_reward, p.total_energy,
                p.total_accepted, p.total_tasks
            ])
    logger.info(f"Pareto CSV saved to {output_path}")


def print_pareto_summary(
    all_points: List[ParetoPoint],
    frontier: List[ParetoPoint]
):
    """Print formatted summary of Pareto analysis."""
    print("\n" + "=" * 80)
    print("PARETO FRONT ANALYSIS RESULTS")
    print("=" * 80)
    print(f"\n{'Energy Weight':<15} {'Energy/Task':<15} {'Acceptance':<15} {'SLA Compliance':<15} {'Pareto?'}")
    print("-" * 80)

    for point in sorted(all_points, key=lambda p: p.energy_weight):
        is_pareto = point in frontier
        pareto_str = "Yes" if is_pareto else "No"
        print(
            f"{point.energy_weight:<15.2f} "
            f"{point.avg_energy_per_task:<15.6f} "
            f"{point.acceptance_rate:<15.2%} "
            f"{point.sla_compliance_rate:<15.2%} "
            f"{pareto_str}"
        )

    print("=" * 80)
    print(f"\nPareto-optimal points: {len(frontier)}/{len(all_points)}")
    print("\nTradeoff Summary:")
    print(f"  Lowest energy:    {min(p.avg_energy_per_task for p in all_points):.6f} kWh/task")
    print(f"  Highest energy:   {max(p.avg_energy_per_task for p in all_points):.6f} kWh/task")
    print(f"  Lowest acceptance:  {min(p.acceptance_rate for p in all_points):.2%}")
    print(f"  Highest acceptance: {max(p.acceptance_rate for p in all_points):.2%}")
    print("=" * 80)


def main():
    global logger
    parser = argparse.ArgumentParser(description='Pareto front analysis')
    parser.add_argument('--weights', type=str, default=None,
                        help='Comma-separated energy weights (e.g., 0.5,0.7,0.9)')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Training timesteps per weight')
    parser.add_argument('--eval-episodes', type=int, default=None,
                        help='Evaluation episodes')
    args = parser.parse_args()

    config = ExperimentConfig()
    logger = setup_experiment_logging(config, "pareto_analysis")

    if args.timesteps:
        config.training_timesteps = args.timesteps
    if args.eval_episodes:
        config.evaluation_episodes = args.eval_episodes

    weights = None
    if args.weights:
        weights = [float(w.strip()) for w in args.weights.split(',')]

    run_pareto_analysis(config, energy_weights=weights)


if __name__ == "__main__":
    main()
