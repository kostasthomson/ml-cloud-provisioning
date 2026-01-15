"""
Enhanced training script for the RL-based resource allocation agent.

Features:
- Aggressive energy-focused reward shaping
- Extended training duration (1M+ timesteps)
- Comprehensive logging and visualization
- Training curve plots
- Checkpoint saving

Usage:
    python train_rl_enhanced.py [--timesteps N] [--save-path PATH]
"""

import sys
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl import RLAgent, PPOTrainer, CloudProvisioningEnv, RLTrainingConfig

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingLogger:
    """Comprehensive logging for RL training."""

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = {
            'timesteps': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': [],
            'avg_energy': [],
            'acceptance_rate': [],
            'sla_compliance': [],
            'rolling_reward': []
        }

        self.start_time = time.time()
        self.checkpoint_interval = 100000

    def log_episode(self, timestep: int, reward: float, length: int, info: Dict):
        """Log episode completion."""
        self.metrics['timesteps'].append(timestep)
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(length)

        if 'acceptance_rate' in info:
            self.metrics['acceptance_rate'].append(info['acceptance_rate'])
        if 'total_energy' in info:
            self.metrics['avg_energy'].append(info['total_energy'])

        if len(self.metrics['episode_rewards']) >= 100:
            rolling = np.mean(self.metrics['episode_rewards'][-100:])
            self.metrics['rolling_reward'].append(rolling)

    def log_update(self, timestep: int, policy_loss: float, value_loss: float, entropy_loss: float):
        """Log training update."""
        self.metrics['policy_losses'].append(policy_loss)
        self.metrics['value_losses'].append(value_loss)
        self.metrics['entropy_losses'].append(entropy_loss)

    def save_metrics(self, filename: str = 'training_metrics.json'):
        """Save metrics to JSON file."""
        save_data = {k: [float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for v in vals] for k, vals in self.metrics.items()}
        save_data['training_time_seconds'] = time.time() - self.start_time

        with open(self.log_dir / filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        logger.info(f"Metrics saved to {self.log_dir / filename}")

    def plot_training_curves(self, filename: str = 'training_curves.png'):
        """Generate training visualization plots."""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available, skipping plot generation")
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('RL Training Progress', fontsize=14)

        if self.metrics['episode_rewards']:
            ax = axes[0, 0]
            ax.plot(self.metrics['episode_rewards'], alpha=0.3, label='Episode')
            if len(self.metrics['episode_rewards']) >= 100:
                rolling = np.convolve(self.metrics['episode_rewards'],
                                     np.ones(100)/100, mode='valid')
                ax.plot(range(99, len(self.metrics['episode_rewards'])), rolling,
                       label='100-ep Rolling Mean', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title('Episode Rewards')
            ax.legend()
            ax.grid(True, alpha=0.3)

        if self.metrics['policy_losses']:
            ax = axes[0, 1]
            ax.plot(self.metrics['policy_losses'])
            ax.set_xlabel('Update')
            ax.set_ylabel('Loss')
            ax.set_title('Policy Loss')
            ax.grid(True, alpha=0.3)

        if self.metrics['value_losses']:
            ax = axes[0, 2]
            ax.plot(self.metrics['value_losses'])
            ax.set_xlabel('Update')
            ax.set_ylabel('Loss')
            ax.set_title('Value Loss')
            ax.grid(True, alpha=0.3)

        if self.metrics['episode_lengths']:
            ax = axes[1, 0]
            ax.plot(self.metrics['episode_lengths'], alpha=0.5)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Steps')
            ax.set_title('Episode Length')
            ax.grid(True, alpha=0.3)

        if self.metrics['entropy_losses']:
            ax = axes[1, 1]
            ax.plot(self.metrics['entropy_losses'])
            ax.set_xlabel('Update')
            ax.set_ylabel('Entropy')
            ax.set_title('Entropy Loss')
            ax.grid(True, alpha=0.3)

        if self.metrics['acceptance_rate']:
            ax = axes[1, 2]
            ax.plot(self.metrics['acceptance_rate'])
            ax.set_xlabel('Episode')
            ax.set_ylabel('Rate')
            ax.set_title('Acceptance Rate')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.log_dir / filename, dpi=150)
        plt.close()
        logger.info(f"Training curves saved to {self.log_dir / filename}")


def train_enhanced(
    total_timesteps: int = 1000000,
    save_path: str = "models/rl/ppo/model_agnostic.pth",
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    log_dir: str = "logs/rl_training",
    checkpoint_interval: int = 100000,
    env_preset: str = 'medium'
):
    """Enhanced training with comprehensive logging."""

    project_root = Path(__file__).parent.parent
    log_dir_full = project_root / log_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path_full = project_root / save_path

    logger.info("=" * 60)
    logger.info("ENHANCED RL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"PPO epochs: {n_epochs}")
    logger.info(f"Gamma: {gamma}")
    logger.info(f"Environment preset: {env_preset}")
    logger.info(f"Log directory: {log_dir_full}")
    logger.info(f"Model save path: {save_path_full}")
    logger.info("=" * 60)

    training_logger = TrainingLogger(str(log_dir_full))

    agent = RLAgent()
    logger.info(f"Agent initialized. Task dim: {agent.task_dim}, HW dim: {agent.hw_dim}")

    reward_config = agent.reward_calculator.get_config()
    logger.info(f"Reward configuration: {reward_config}")

    config = RLTrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        total_timesteps=total_timesteps
    )

    trainer = PPOTrainer(agent, config)
    env = CloudProvisioningEnv(preset=env_preset, episode_length=100)

    logger.info(f"Environment HW types: {env.get_num_hw_types()}")
    logger.info(f"HW type IDs: {env.get_hw_type_ids()}")

    last_checkpoint = 0
    episode_count = 0

    def training_callback(timestep, info):
        nonlocal last_checkpoint, episode_count

        training_logger.log_update(
            timestep,
            info.get('policy_loss', 0),
            info.get('value_loss', 0),
            info.get('entropy_loss', 0)
        )

        if timestep % 10000 == 0:
            status = trainer.get_status()
            elapsed = time.time() - training_logger.start_time
            steps_per_sec = timestep / max(elapsed, 1)
            remaining = (total_timesteps - timestep) / max(steps_per_sec, 1)

            logger.info(
                f"Step {timestep:,}/{total_timesteps:,} ({100*timestep/total_timesteps:.1f}%) | "
                f"Episodes: {status.episodes_completed} | "
                f"Avg Reward: {status.avg_reward:.3f} | "
                f"Speed: {steps_per_sec:.0f} steps/s | "
                f"ETA: {remaining/60:.1f} min"
            )

        if timestep - last_checkpoint >= checkpoint_interval:
            checkpoint_path = log_dir_full / f"checkpoint_{timestep}.pth"
            agent.save(str(checkpoint_path))
            training_logger.save_metrics()
            training_logger.plot_training_curves()
            last_checkpoint = timestep
            logger.info(f"Checkpoint saved at step {timestep:,}")

    logger.info("Starting training...")
    start_time = time.time()

    stats = trainer.train(env, total_timesteps=total_timesteps, callback=training_callback)

    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time/60:.2f} minutes")

    save_path_full.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(save_path_full))

    for i, reward in enumerate(stats['episode_rewards']):
        training_logger.log_episode(i * 100, reward, stats['episode_lengths'][i] if i < len(stats['episode_lengths']) else 100, {})

    training_logger.save_metrics()
    training_logger.plot_training_curves()

    summary = {
        'total_timesteps': total_timesteps,
        'episodes_completed': trainer.episodes_completed,
        'training_time_seconds': training_time,
        'final_avg_reward': agent.last_training_reward,
        'model_path': str(save_path_full),
        'config': {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'n_epochs': n_epochs,
            'gamma': gamma,
            'env_preset': env_preset
        },
        'reward_config': reward_config
    }

    with open(log_dir_full / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total timesteps: {total_timesteps:,}")
    logger.info(f"Episodes completed: {trainer.episodes_completed}")
    logger.info(f"Training time: {training_time/60:.2f} minutes")
    logger.info(f"Final avg reward: {agent.last_training_reward:.4f}")
    logger.info(f"Model saved to: {save_path_full}")
    logger.info(f"Logs saved to: {log_dir_full}")
    logger.info("=" * 60)

    return agent, stats, training_logger


def main():
    parser = argparse.ArgumentParser(description='Enhanced RL training for cloud provisioning')
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Total training timesteps (default: 1M)')
    parser.add_argument('--save-path', type=str, default='models/rl/ppo/model_agnostic.pth',
                       help='Model save path')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--log-dir', type=str, default='logs/rl_training',
                       help='Log directory')
    parser.add_argument('--checkpoint-interval', type=int, default=100000,
                       help='Checkpoint save interval')
    parser.add_argument('--env-preset', type=str, default='medium',
                       choices=['small', 'medium', 'large', 'enterprise'],
                       help='Environment preset')

    args = parser.parse_args()

    train_enhanced(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma,
        log_dir=args.log_dir,
        checkpoint_interval=args.checkpoint_interval,
        env_preset=args.env_preset
    )


if __name__ == '__main__':
    main()
