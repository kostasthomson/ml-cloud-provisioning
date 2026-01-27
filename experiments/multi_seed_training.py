#!/usr/bin/env python3
"""
Multi-Seed Training for Statistical Validity.

Trains the RL agent multiple times with different random seeds to establish
statistical significance of results. Produces mean ± std for all metrics.

Usage:
    python multi_seed_training.py [--seeds 5] [--timesteps 200000]
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
from rl.schemas import RLTrainingConfig
from experiments.config import ExperimentConfig, DEFAULT_CONFIG, setup_experiment_logging

logger = None


@dataclass
class TrainingResult:
    """Results from a single training run."""
    seed: int
    final_reward: float
    episodes_completed: int
    training_time_sec: float
    policy_loss_final: float
    value_loss_final: float
    model_path: str


@dataclass
class EvaluationResult:
    """Evaluation results for a trained model."""
    seed: int
    total_energy_kwh: float
    energy_per_task: float
    acceptance_rate: float
    sla_compliance_rate: float
    avg_episode_reward: float
    episode_energies: List[float]
    episode_rewards: List[float]


def train_single_seed(
    seed: int,
    config: ExperimentConfig,
    save_dir: Path
) -> TrainingResult:
    """Train a model with a specific seed."""
    logger.info(f"Training with seed {seed}...")

    np.random.seed(seed)

    env = CloudProvisioningEnv(
        preset=config.env_preset,
        episode_length=config.episode_length,
        max_steps=config.training_timesteps,
        seed=seed,
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

    start_time = datetime.now()
    stats = trainer.train(env, total_timesteps=config.training_timesteps)
    training_time = (datetime.now() - start_time).total_seconds()

    model_path = save_dir / f"model_seed_{seed}.pth"
    agent.save(str(model_path))

    return TrainingResult(
        seed=seed,
        final_reward=float(np.mean(stats['episode_rewards'][-100:])) if stats['episode_rewards'] else 0.0,
        episodes_completed=len(stats['episode_rewards']),
        training_time_sec=training_time,
        policy_loss_final=float(np.mean(stats['policy_losses'][-10:])) if stats['policy_losses'] else 0.0,
        value_loss_final=float(np.mean(stats['value_losses'][-10:])) if stats['value_losses'] else 0.0,
        model_path=str(model_path)
    )


def evaluate_model(
    model_path: str,
    seed: int,
    config: ExperimentConfig
) -> EvaluationResult:
    """Evaluate a trained model."""
    logger.info(f"Evaluating model from seed {seed}...")

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
    episode_energies = []
    episode_rewards = []

    for episode in range(config.evaluation_episodes):
        state, _ = env.reset(seed=episode + seed * 1000)
        episode_energy = 0.0
        episode_reward = 0.0
        episode_accepted = 0
        episode_violations = 0
        done = False

        while not done:
            action_result, _, _ = agent.predict(state, deterministic=True)
            action = action_result.action

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            total_tasks += 1

            if info.get('accepted', False):
                episode_accepted += 1
                total_accepted += 1
                energy = info.get('energy', 0.0)
                episode_energy += energy
                total_energy += energy

                if state.task.deadline is not None:
                    exec_time = info.get('exec_time', 0.0)
                    if exec_time > state.task.deadline:
                        episode_violations += 1
                        total_sla_violations += 1

            state = next_state

        episode_energies.append(episode_energy)
        episode_rewards.append(episode_reward)

    return EvaluationResult(
        seed=seed,
        total_energy_kwh=total_energy,
        energy_per_task=total_energy / max(total_accepted, 1),
        acceptance_rate=total_accepted / max(total_tasks, 1),
        sla_compliance_rate=1.0 - (total_sla_violations / max(total_accepted, 1)),
        avg_episode_reward=float(np.mean(episode_rewards)),
        episode_energies=episode_energies,
        episode_rewards=episode_rewards
    )


def compute_statistics(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Compute aggregate statistics across all seeds."""
    from scipy import stats as scipy_stats

    energies = [r.energy_per_task for r in results]
    acceptance_rates = [r.acceptance_rate for r in results]
    sla_rates = [r.sla_compliance_rate for r in results]
    rewards = [r.avg_episode_reward for r in results]

    def compute_ci(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        se = scipy_stats.sem(data)
        ci = scipy_stats.t.interval(confidence, n-1, loc=mean, scale=se)
        return ci

    return {
        "n_seeds": len(results),
        "energy_per_task": {
            "mean": float(np.mean(energies)),
            "std": float(np.std(energies)),
            "ci_95": compute_ci(energies) if len(energies) > 1 else (np.mean(energies), np.mean(energies)),
            "min": float(np.min(energies)),
            "max": float(np.max(energies))
        },
        "acceptance_rate": {
            "mean": float(np.mean(acceptance_rates)),
            "std": float(np.std(acceptance_rates)),
            "ci_95": compute_ci(acceptance_rates) if len(acceptance_rates) > 1 else (np.mean(acceptance_rates), np.mean(acceptance_rates)),
        },
        "sla_compliance_rate": {
            "mean": float(np.mean(sla_rates)),
            "std": float(np.std(sla_rates)),
            "ci_95": compute_ci(sla_rates) if len(sla_rates) > 1 else (np.mean(sla_rates), np.mean(sla_rates)),
        },
        "avg_reward": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "ci_95": compute_ci(rewards) if len(rewards) > 1 else (np.mean(rewards), np.mean(rewards)),
        }
    }


def _save_checkpoint(config: ExperimentConfig, training_results: List, evaluation_results: List, seeds: List[int], failed_seeds: List[int]):
    """Save intermediate checkpoint to preserve progress."""
    checkpoint = {
        "experiment": "multi_seed_training",
        "timestamp": datetime.now().isoformat(),
        "status": "in_progress",
        "completed_seeds": [r.seed for r in training_results],
        "failed_seeds": failed_seeds,
        "training_results": [asdict(r) for r in training_results],
        "evaluation_results": [
            {
                "seed": r.seed,
                "total_energy_kwh": r.total_energy_kwh,
                "energy_per_task": r.energy_per_task,
                "acceptance_rate": r.acceptance_rate,
                "sla_compliance_rate": r.sla_compliance_rate,
                "avg_episode_reward": r.avg_episode_reward
            }
            for r in evaluation_results
        ]
    }
    checkpoint_path = config.results_dir / "data" / "multi_seed_checkpoint.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)
    return checkpoint_path


def run_multi_seed_experiment(
    config: ExperimentConfig,
    num_seeds: int = None
) -> Dict[str, Any]:
    """Run complete multi-seed training and evaluation."""
    global logger
    if logger is None:
        logger = setup_experiment_logging(config, "multi_seed_training")

    seeds = config.seeds[:num_seeds] if num_seeds else config.seeds
    logger.info(f"Running multi-seed experiment with {len(seeds)} seeds")

    save_dir = config.results_dir / "models" / "multi_seed"
    save_dir.mkdir(parents=True, exist_ok=True)

    training_results = []
    evaluation_results = []
    failed_seeds = []

    for i, seed in enumerate(seeds):
        logger.info(f"[{i+1}/{len(seeds)}] Training seed {seed}...")
        try:
            result = train_single_seed(seed, config, save_dir)
            training_results.append(result)
            logger.info(f"Seed {seed}: reward={result.final_reward:.3f}, episodes={result.episodes_completed}")

            logger.info(f"[{i+1}/{len(seeds)}] Evaluating seed {seed}...")
            eval_result = evaluate_model(result.model_path, seed, config)
            evaluation_results.append(eval_result)
            logger.info(f"Seed {seed} evaluation: energy={eval_result.energy_per_task:.6f}, acceptance={eval_result.acceptance_rate:.2%}")

            checkpoint_path = _save_checkpoint(config, training_results, evaluation_results, seeds, failed_seeds)
            logger.debug(f"Checkpoint saved: {checkpoint_path}")

        except Exception as e:
            logger.error(f"Seed {seed} FAILED: {e}")
            failed_seeds.append(seed)
            _save_checkpoint(config, training_results, evaluation_results, seeds, failed_seeds)
            continue

    if not evaluation_results:
        logger.error("All seeds failed! No results to compute.")
        return {"error": "All seeds failed", "failed_seeds": failed_seeds}

    try:
        statistics = compute_statistics(evaluation_results)
    except Exception as e:
        logger.warning(f"Statistics computation failed: {e}. Using basic stats.")
        statistics = {
            "n_seeds": len(evaluation_results),
            "energy_per_task": {"mean": np.mean([r.energy_per_task for r in evaluation_results]), "std": 0},
            "acceptance_rate": {"mean": np.mean([r.acceptance_rate for r in evaluation_results]), "std": 0},
            "sla_compliance_rate": {"mean": np.mean([r.sla_compliance_rate for r in evaluation_results]), "std": 0},
            "avg_reward": {"mean": np.mean([r.avg_episode_reward for r in evaluation_results]), "std": 0}
        }

    results = {
        "experiment": "multi_seed_training",
        "timestamp": datetime.now().isoformat(),
        "status": "completed" if not failed_seeds else "partial",
        "config": {
            "seeds": seeds,
            "timesteps": config.training_timesteps,
            "evaluation_episodes": config.evaluation_episodes,
            "env_preset": config.env_preset,
            "exec_time_noise": config.exec_time_noise,
            "energy_noise": config.energy_noise
        },
        "completed_seeds": [r.seed for r in training_results],
        "failed_seeds": failed_seeds,
        "training_results": [asdict(r) for r in training_results],
        "evaluation_results": [
            {
                "seed": r.seed,
                "total_energy_kwh": r.total_energy_kwh,
                "energy_per_task": r.energy_per_task,
                "acceptance_rate": r.acceptance_rate,
                "sla_compliance_rate": r.sla_compliance_rate,
                "avg_episode_reward": r.avg_episode_reward
            }
            for r in evaluation_results
        ],
        "statistics": statistics
    }

    output_path = config.results_dir / "data" / "multi_seed_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {output_path}")

    if failed_seeds:
        logger.warning(f"Completed with {len(failed_seeds)} failed seeds: {failed_seeds}")

    print_summary(statistics, [r.seed for r in training_results])

    return results


def print_summary(stats: Dict, seeds: List[int]):
    """Print formatted summary of results."""
    print("\n" + "=" * 70)
    print("MULTI-SEED TRAINING RESULTS")
    print("=" * 70)
    print(f"Number of seeds: {len(seeds)}")
    print(f"Seeds used: {seeds}")
    print("-" * 70)
    print(f"{'Metric':<25} {'Mean':<15} {'Std':<15} {'95% CI'}")
    print("-" * 70)

    for metric, data in stats.items():
        if isinstance(data, dict) and 'mean' in data:
            ci = data.get('ci_95', (data['mean'], data['mean']))
            ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            print(f"{metric:<25} {data['mean']:<15.4f} {data['std']:<15.4f} {ci_str}")

    print("=" * 70)


def main():
    global logger
    parser = argparse.ArgumentParser(description='Multi-seed training for statistical validity')
    parser.add_argument('--seeds', type=int, default=None, help='Number of seeds to use')
    parser.add_argument('--timesteps', type=int, default=None, help='Training timesteps per seed')
    parser.add_argument('--eval-episodes', type=int, default=None, help='Evaluation episodes per model')
    parser.add_argument('--preset', type=str, default=None, help='Environment preset')
    parser.add_argument('--noise', type=float, default=None, help='Execution time noise factor')
    args = parser.parse_args()

    config = ExperimentConfig()
    logger = setup_experiment_logging(config, "multi_seed_training")

    if args.timesteps:
        config.training_timesteps = args.timesteps
    if args.eval_episodes:
        config.evaluation_episodes = args.eval_episodes
    if args.preset:
        config.env_preset = args.preset
    if args.noise:
        config.exec_time_noise = args.noise
        config.energy_noise = args.noise * 0.7

    run_multi_seed_experiment(config, num_seeds=args.seeds)


if __name__ == "__main__":
    main()
