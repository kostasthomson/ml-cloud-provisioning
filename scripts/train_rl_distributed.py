#!/usr/bin/env python3
"""
Distributed training script for the RL-based resource allocation agent.

Best practice: Use torchrun to launch multi-GPU training.

Usage:
    # Single GPU
    python train_rl_distributed.py --timesteps 100000

    # Multi-GPU with torchrun (RECOMMENDED)
    torchrun --nproc_per_node=4 train_rl_distributed.py --timesteps 200000

    # Multi-node training
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_backend=c10d \
             --rdzv_endpoint=MASTER_IP:29500 train_rl_distributed.py --timesteps 500000
"""

import sys
import argparse
import logging
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.distributed_trainer import DistributedPPOTrainer, cleanup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Distributed RL training for cloud provisioning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU:
    python train_rl_distributed.py --timesteps 100000 --num-envs 8

  Multi-GPU (4 GPUs):
    torchrun --nproc_per_node=4 train_rl_distributed.py --timesteps 200000

  Multi-GPU with custom settings:
    torchrun --nproc_per_node=4 train_rl_distributed.py \\
        --timesteps 200000 --num-envs 8 --batch-size 128 --lr 1e-4
        """
    )

    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps across all GPUs (default: 100000)')
    parser.add_argument('--save-path', type=str, default='models/rl/ppo/model_distributed.pth',
                        help='Model save path (default: models/rl/ppo/model_distributed.pth)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='PPO mini-batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='PPO epochs per update (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda (default: 0.95)')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clip range (default: 0.2)')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='Parallel environments per GPU (default: 8)')
    parser.add_argument('--rollout-steps', type=int, default=256,
                        help='Steps per rollout before update (default: 256)')
    parser.add_argument('--env-preset', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'enterprise'],
                        help='Environment preset (default: medium)')
    parser.add_argument('--log-interval', type=int, default=5000,
                        help='Log progress every N timesteps (default: 5000)')

    args = parser.parse_args()

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', 0)))

    if rank == 0:
        logger.info("=" * 60)
        logger.info("Distributed PPO Training")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size} GPU(s)")
        logger.info(f"Total timesteps: {args.timesteps}")
        logger.info(f"Environments per GPU: {args.num_envs}")
        logger.info(f"Effective parallel envs: {args.num_envs * world_size}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"PPO epochs: {args.epochs}")
        logger.info(f"Environment preset: {args.env_preset}")
        logger.info("=" * 60)

    try:
        trainer = DistributedPPOTrainer(
            learning_rate=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
        )

        results = trainer.train(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            env_preset=args.env_preset,
            log_interval=args.log_interval,
            save_path=args.save_path,
        )

        if trainer.is_main_process:
            logger.info("=" * 60)
            logger.info("Training Complete")
            logger.info("=" * 60)
            logger.info(f"Total timesteps: {results['total_timesteps']}")
            logger.info(f"Episodes completed: {results['episodes']}")
            logger.info(f"Average reward (last 100): {results['avg_reward']:.2f}")
            logger.info(f"Training time: {results['elapsed_time']:.1f}s")
            logger.info(f"Throughput: {results['fps']:.0f} steps/sec")
            logger.info(f"Model saved to: {args.save_path}")
            logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        cleanup()


if __name__ == '__main__':
    main()
