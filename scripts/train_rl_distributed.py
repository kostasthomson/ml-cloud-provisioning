"""
Distributed training script for the RL-based resource allocation agent.

Supports both single-GPU and multi-GPU training using PyTorch DistributedDataParallel.

Usage:
    # Single GPU (fallback)
    python train_rl_distributed.py --timesteps 100000

    # Multi-GPU (automatic detection)
    python train_rl_distributed.py --timesteps 100000 --distributed

    # Specify number of GPUs
    python train_rl_distributed.py --timesteps 100000 --distributed --num-gpus 4

    # With torchrun (recommended for multi-node)
    torchrun --nproc_per_node=4 train_rl_distributed.py --timesteps 100000 --use-torchrun
"""

import sys
import argparse
import logging
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl import RLAgent, PPOTrainer, CloudProvisioningEnv, RLTrainingConfig
from rl.distributed_trainer import (
    DistributedPPOTrainer,
    VectorizedEnv,
    run_distributed_training,
    setup_distributed,
    cleanup_distributed
)
from rl.agent import PolicyNetwork
from rl.state_encoder import StateEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_single_gpu(
    total_timesteps: int,
    save_path: str,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    gamma: float,
    max_steps_per_episode: int,
    num_envs: int,
    env_preset: str
):
    """Train on a single GPU with vectorized environments."""
    logger.info("Starting single-GPU training with vectorized environments...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    encoder = StateEncoder()
    policy = PolicyNetwork(
        task_dim=encoder.task_dim,
        hw_dim=encoder.hw_dim,
        embed_dim=64
    ).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

    vec_env = VectorizedEnv(num_envs, {'preset': env_preset, 'episode_length': max_steps_per_episode})
    states = vec_env.reset()

    config = RLTrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        total_timesteps=total_timesteps
    )

    from rl.distributed_trainer import ParallelPPOBuffer, Experience

    buffer = ParallelPPOBuffer(gamma=gamma)
    current_timestep = 0
    episodes_completed = 0
    episode_rewards = []
    current_ep_rewards = [0.0] * num_envs
    rollout_steps = 512

    training_stats = {
        'episode_rewards': [],
        'policy_losses': [],
        'value_losses': []
    }

    logger.info(f"Training for {total_timesteps} timesteps with {num_envs} parallel environments")
    start_time = time.time()

    while current_timestep < total_timesteps:
        buffer.clear()

        for _ in range(rollout_steps):
            task_vecs = []
            hw_vecs_list = []
            hw_ids_list = []
            valid_masks = []
            action_indices = []
            log_probs = []
            values = []

            for state in states:
                task_vec, hw_list = encoder.encode(state)
                valid_hw_types = encoder.get_valid_hw_types(state)
                hw_type_ids = [hw_id for hw_id, _ in hw_list]
                hw_vecs = [hw_vec for _, hw_vec in hw_list]
                valid_mask = np.array([hw_id in valid_hw_types for hw_id in hw_type_ids], dtype=bool)

                task_vecs.append(task_vec)
                hw_vecs_list.append(hw_vecs)
                hw_ids_list.append(hw_type_ids)
                valid_masks.append(valid_mask)

            with torch.no_grad():
                for i, (task_vec, hw_vecs, valid_mask) in enumerate(zip(task_vecs, hw_vecs_list, valid_masks)):
                    task_tensor = torch.FloatTensor(task_vec).to(device)
                    hw_tensors = [torch.FloatTensor(hv).to(device) for hv in hw_vecs]
                    mask_tensor = torch.BoolTensor(valid_mask).to(device)

                    action_probs, value, _ = policy.forward(task_tensor, hw_tensors, mask_tensor)
                    probs = action_probs.cpu().numpy()
                    action_idx = np.random.choice(len(probs), p=probs)
                    log_prob = np.log(probs[action_idx] + 1e-8)

                    action_indices.append(action_idx)
                    log_probs.append(log_prob)
                    values.append(value.item())

            actions = []
            for action_idx, hw_ids in zip(action_indices, hw_ids_list):
                if action_idx < len(hw_ids):
                    actions.append(hw_ids[action_idx])
                else:
                    actions.append(-1)

            next_states, rewards, dones, truncateds, infos = vec_env.step(actions)

            for i in range(num_envs):
                exp = Experience(
                    task_vec=task_vecs[i],
                    hw_vecs=hw_vecs_list[i],
                    hw_type_ids=hw_ids_list[i],
                    valid_mask=valid_masks[i],
                    action_idx=action_indices[i],
                    reward=rewards[i],
                    value=values[i],
                    log_prob=log_probs[i],
                    done=dones[i] or truncateds[i]
                )
                buffer.store(exp)
                current_ep_rewards[i] += rewards[i]

                if dones[i] or truncateds[i]:
                    episode_rewards.append(current_ep_rewards[i])
                    training_stats['episode_rewards'].append(current_ep_rewards[i])
                    current_ep_rewards[i] = 0.0
                    episodes_completed += 1

            states = next_states
            current_timestep += num_envs

            if current_timestep >= total_timesteps:
                break

        buffer.compute_advantages()
        data = buffer.get_all()

        if data:
            total_policy_loss = 0.0
            total_value_loss = 0.0
            n_updates = 0

            for _ in range(n_epochs):
                indices = np.random.permutation(len(data))

                for start in range(0, len(data), batch_size):
                    end = min(start + batch_size, len(data))
                    batch_indices = indices[start:end]

                    batch_policy_loss = torch.tensor(0.0, device=device)
                    batch_value_loss = torch.tensor(0.0, device=device)

                    for idx in batch_indices:
                        exp, advantage, ret = data[idx]

                        task_tensor = torch.FloatTensor(exp.task_vec).to(device)
                        hw_tensors = [torch.FloatTensor(hv).to(device) for hv in exp.hw_vecs]
                        mask_tensor = torch.BoolTensor(exp.valid_mask).to(device)

                        action_probs, value, _ = policy.forward(task_tensor, hw_tensors, mask_tensor)

                        new_log_prob = torch.log(action_probs[exp.action_idx] + 1e-8)
                        ratio = torch.exp(new_log_prob - exp.log_prob)
                        adv_tensor = torch.tensor(advantage, device=device)

                        surr1 = ratio * adv_tensor
                        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_tensor
                        policy_loss = -torch.min(surr1, surr2)

                        ret_tensor = torch.tensor(ret, device=device)
                        value_loss = 0.5 * (value - ret_tensor) ** 2

                        batch_policy_loss = batch_policy_loss + policy_loss
                        batch_value_loss = batch_value_loss + value_loss

                    bs = len(batch_indices)
                    loss = (batch_policy_loss + 0.5 * batch_value_loss) / bs

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()

                    total_policy_loss += batch_policy_loss.item() / bs
                    total_value_loss += batch_value_loss.item() / bs
                    n_updates += 1

            training_stats['policy_losses'].append(total_policy_loss / max(n_updates, 1))
            training_stats['value_losses'].append(total_value_loss / max(n_updates, 1))

        if current_timestep % 5000 < rollout_steps * num_envs:
            elapsed = time.time() - start_time
            steps_per_sec = current_timestep / elapsed
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
            logger.info(
                f"Step {current_timestep}/{total_timesteps} | "
                f"Episodes: {episodes_completed} | "
                f"Avg Reward: {avg_reward:.3f} | "
                f"Speed: {steps_per_sec:.0f} steps/s"
            )

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s ({current_timestep/elapsed:.0f} steps/s)")

    save_dict = {
        'policy_state_dict': policy.state_dict(),
        'is_trained': True,
        'training_timesteps': current_timestep,
        'task_dim': encoder.task_dim,
        'hw_dim': encoder.hw_dim,
        'embed_dim': 64,
        'infrastructure_agnostic': True,
        'last_training_reward': np.mean(episode_rewards[-100:]) if episode_rewards else 0,
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_dict, save_path)
    logger.info(f"Model saved to {save_path}")

    return training_stats


def train_torchrun(
    total_timesteps: int,
    save_path: str,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    gamma: float,
    num_envs_per_gpu: int,
    rollout_steps: int,
    env_preset: str
):
    """Train using torchrun launcher (for multi-node support)."""
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    setup_distributed(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    if rank == 0:
        logger.info(f"Starting torchrun training with {world_size} processes")

    encoder = StateEncoder()
    policy = PolicyNetwork(
        task_dim=encoder.task_dim,
        hw_dim=encoder.hw_dim,
        embed_dim=64
    )

    config = RLTrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        total_timesteps=total_timesteps
    )

    trainer = DistributedPPOTrainer(
        policy=policy,
        config=config,
        rank=rank,
        world_size=world_size,
        device=device
    )

    stats = trainer.train(
        total_timesteps=total_timesteps,
        num_envs_per_gpu=num_envs_per_gpu,
        rollout_steps=rollout_steps,
        env_kwargs={'preset': env_preset}
    )

    if rank == 0:
        save_dict = {
            'policy_state_dict': trainer.get_policy_state_dict(),
            'is_trained': True,
            'training_timesteps': total_timesteps,
            'task_dim': encoder.task_dim,
            'hw_dim': encoder.hw_dim,
            'embed_dim': 64,
            'infrastructure_agnostic': True,
            'distributed_training': True,
            'world_size': world_size,
        }
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, save_path)
        logger.info(f"Model saved to {save_path}")

    cleanup_distributed()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Distributed RL training for cloud provisioning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single GPU with 8 parallel environments:
    python train_rl_distributed.py --timesteps 100000 --num-envs 8

  Multi-GPU with automatic GPU detection:
    python train_rl_distributed.py --timesteps 100000 --distributed

  Multi-GPU with specific GPU count:
    python train_rl_distributed.py --timesteps 100000 --distributed --num-gpus 4

  Using torchrun for multi-node:
    torchrun --nproc_per_node=4 train_rl_distributed.py --timesteps 100000 --use-torchrun
        """
    )

    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps (default: 100000)')
    parser.add_argument('--save-path', type=str, default='models/rl/ppo/model_distributed.pth',
                        help='Model save path')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='PPO batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='PPO epochs per update (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='Parallel environments per GPU (default: 8)')
    parser.add_argument('--rollout-steps', type=int, default=512,
                        help='Steps per rollout before update (default: 512)')
    parser.add_argument('--env-preset', type=str, default='medium',
                        choices=['small', 'medium', 'large', 'enterprise'],
                        help='Environment preset (default: medium)')
    parser.add_argument('--max-steps', type=int, default=2048,
                        help='Max steps per episode (default: 2048)')

    parser.add_argument('--distributed', action='store_true',
                        help='Enable multi-GPU distributed training')
    parser.add_argument('--num-gpus', type=int, default=None,
                        help='Number of GPUs (default: all available)')
    parser.add_argument('--use-torchrun', action='store_true',
                        help='Use torchrun launcher (set automatically by torchrun)')

    args = parser.parse_args()

    if 'LOCAL_RANK' in os.environ or args.use_torchrun:
        train_torchrun(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            num_envs_per_gpu=args.num_envs,
            rollout_steps=args.rollout_steps,
            env_preset=args.env_preset
        )
    elif args.distributed:
        num_gpus = args.num_gpus or torch.cuda.device_count()
        if num_gpus < 2:
            logger.warning(f"Only {num_gpus} GPU(s) available. Falling back to single-GPU training.")
            train_single_gpu(
                total_timesteps=args.timesteps,
                save_path=args.save_path,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                gamma=args.gamma,
                max_steps_per_episode=args.max_steps,
                num_envs=args.num_envs,
                env_preset=args.env_preset
            )
        else:
            result = run_distributed_training(
                total_timesteps=args.timesteps,
                save_path=args.save_path,
                num_gpus=num_gpus,
                num_envs_per_gpu=args.num_envs,
                rollout_steps=args.rollout_steps,
                learning_rate=args.lr,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                gamma=args.gamma,
                env_preset=args.env_preset
            )
            if result.get('success'):
                logger.info(f"Training completed. Episodes: {result['episodes']}, "
                           f"Avg reward: {result['avg_reward']:.3f}")
            else:
                logger.error(f"Training failed: {result.get('error', 'Unknown error')}")
    else:
        train_single_gpu(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            max_steps_per_episode=args.max_steps,
            num_envs=args.num_envs,
            env_preset=args.env_preset
        )


if __name__ == '__main__':
    main()
