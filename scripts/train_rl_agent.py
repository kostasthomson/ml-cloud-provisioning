"""
Training script for the RL-based resource allocation agent.

Usage:
    python train_rl_agent.py [--timesteps N] [--save-path PATH]
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl import RLAgent, PPOTrainer, CloudProvisioningEnv, RLTrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train(
    total_timesteps: int = 50000,
    save_path: str = "models/rl/ppo/model.pth",
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99
):
    """Train the RL agent."""
    logger.info("Initializing RL agent and environment...")

    agent = RLAgent()

    config = RLTrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        total_timesteps=total_timesteps
    )

    trainer = PPOTrainer(agent, config)

    env = CloudProvisioningEnv()

    logger.info(f"Starting training for {total_timesteps} timesteps...")
    logger.info(f"Config: LR={learning_rate}, batch={batch_size}, epochs={n_epochs}")

    def callback(timestep, info):
        if timestep % 1000 == 0:
            status = trainer.get_status()
            logger.info(
                f"Step {timestep}/{total_timesteps} | "
                f"Episodes: {status.episodes_completed} | "
                f"Avg Reward: {status.avg_reward:.3f}"
            )

    stats = trainer.train(env, total_timesteps=total_timesteps, callback=callback)

    logger.info(f"Training completed. Saving model to {save_path}...")
    agent.save(save_path)

    logger.info("=== Training Summary ===")
    logger.info(f"Total timesteps: {agent.training_timesteps}")
    logger.info(f"Episodes completed: {trainer.episodes_completed}")
    avg_reward = agent.last_training_reward or 0.0
    logger.info(f"Final avg reward: {avg_reward:.3f}")

    if stats['episode_rewards']:
        logger.info(f"Episode rewards: min={min(stats['episode_rewards']):.3f}, "
                   f"max={max(stats['episode_rewards']):.3f}")

    return agent, stats


def main():
    parser = argparse.ArgumentParser(description='Train RL agent for cloud provisioning')
    parser.add_argument('--timesteps', type=int, default=50000, help='Total training timesteps')
    parser.add_argument('--save-path', type=str, default='models/rl/ppo/model.pth', help='Model save path')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='PPO epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    args = parser.parse_args()

    train(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma
    )


if __name__ == '__main__':
    main()
