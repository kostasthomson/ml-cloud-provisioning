"""
Multi-GPU Distributed PPO Trainer using PyTorch DistributedDataParallel.

Implements parallel environment rollouts and synchronized gradient updates
across multiple GPUs for faster training.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from queue import Empty
import time

from .agent import RLAgent, PolicyNetwork
from .schemas import RLTrainingConfig, RLState
from .state_encoder import StateEncoder
from .environment import CloudProvisioningEnv

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience for training."""
    task_vec: np.ndarray
    hw_vecs: List[np.ndarray]
    hw_type_ids: List[int]
    valid_mask: np.ndarray
    action_idx: int
    reward: float
    value: float
    log_prob: float
    done: bool


class ParallelPPOBuffer:
    """Buffer for storing trajectories from multiple parallel environments."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.experiences: List[Experience] = []
        self.advantages: List[float] = []
        self.returns: List[float] = []

    def store(self, exp: Experience):
        self.experiences.append(exp)

    def store_batch(self, exps: List[Experience]):
        self.experiences.extend(exps)

    def compute_advantages(self, last_values: List[float] = None):
        """Compute GAE-Lambda advantages for all trajectories."""
        if not self.experiences:
            return

        episode_starts = [0]
        for i, exp in enumerate(self.experiences):
            if exp.done and i < len(self.experiences) - 1:
                episode_starts.append(i + 1)

        all_advantages = []
        all_returns = []

        for ep_idx, start_idx in enumerate(episode_starts):
            if ep_idx < len(episode_starts) - 1:
                end_idx = episode_starts[ep_idx + 1]
            else:
                end_idx = len(self.experiences)

            episode_exps = self.experiences[start_idx:end_idx]
            n = len(episode_exps)

            if n == 0:
                continue

            last_val = 0.0 if episode_exps[-1].done else (last_values[ep_idx] if last_values else 0.0)

            rewards = np.array([e.reward for e in episode_exps] + [last_val])
            values = np.array([e.value for e in episode_exps] + [last_val])
            dones = np.array([float(e.done) for e in episode_exps] + [0.0])

            deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]

            advantages = np.zeros(n)
            last_adv = 0.0
            for t in reversed(range(n)):
                advantages[t] = deltas[t] + self.gamma * self.lam * (1 - dones[t]) * last_adv
                last_adv = advantages[t]

            returns = advantages + values[:-1]
            all_advantages.extend(advantages.tolist())
            all_returns.extend(returns.tolist())

        if all_advantages:
            adv_mean = np.mean(all_advantages)
            adv_std = np.std(all_advantages) + 1e-8
            self.advantages = [(a - adv_mean) / adv_std for a in all_advantages]
            self.returns = all_returns

    def get_all(self) -> List[Tuple[Experience, float, float]]:
        return list(zip(self.experiences, self.advantages, self.returns))

    def clear(self):
        self.experiences = []
        self.advantages = []
        self.returns = []

    def __len__(self):
        return len(self.experiences)


class VectorizedEnv:
    """Vectorized environment wrapper for parallel rollouts."""

    def __init__(self, num_envs: int, env_kwargs: Dict = None):
        self.num_envs = num_envs
        env_kwargs = env_kwargs or {}
        self.envs = [CloudProvisioningEnv(**env_kwargs) for _ in range(num_envs)]
        self.states = [None] * num_envs

    def reset(self, seed: int = None) -> List[RLState]:
        self.states = []
        for i, env in enumerate(self.envs):
            state, _ = env.reset(seed=seed + i if seed else None)
            self.states.append(state)
        return self.states

    def step(self, actions: List[int]) -> Tuple[List[RLState], List[float], List[bool], List[bool], List[Dict]]:
        next_states = []
        rewards = []
        dones = []
        truncateds = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            next_state, reward, done, truncated, info = env.step(action)

            if done or truncated:
                reset_state, _ = env.reset()
                next_states.append(reset_state)
            else:
                next_states.append(next_state)

            rewards.append(reward)
            dones.append(done)
            truncateds.append(truncated)
            infos.append(info)

        self.states = next_states
        return next_states, rewards, dones, truncateds, infos

    def get_states(self) -> List[RLState]:
        return self.states


class DistributedPPOTrainer:
    """
    Multi-GPU PPO Trainer using DistributedDataParallel.

    Features:
    - Parallel environment rollouts
    - Synchronized gradient updates across GPUs
    - Efficient data collection with vectorized environments
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        config: RLTrainingConfig,
        rank: int,
        world_size: int,
        device: torch.device
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.is_main = rank == 0

        self.policy = policy.to(device)
        self.policy = DDP(self.policy, device_ids=[rank], output_device=rank)

        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )

        self.encoder = StateEncoder()
        self.buffer = ParallelPPOBuffer(gamma=config.gamma)

        self.current_timestep = 0
        self.episodes_completed = 0
        self.episode_rewards: List[float] = []

    def _encode_states(self, states: List[RLState]) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[List[int]], List[np.ndarray]]:
        """Encode multiple states."""
        task_vecs = []
        hw_vecs_list = []
        hw_ids_list = []
        valid_masks = []

        for state in states:
            task_vec, hw_list = self.encoder.encode(state)
            valid_hw_types = self.encoder.get_valid_hw_types(state)

            hw_type_ids = [hw_id for hw_id, _ in hw_list]
            hw_vecs = [hw_vec for _, hw_vec in hw_list]
            valid_mask = np.array([hw_id in valid_hw_types for hw_id in hw_type_ids], dtype=bool)

            task_vecs.append(task_vec)
            hw_vecs_list.append(hw_vecs)
            hw_ids_list.append(hw_type_ids)
            valid_masks.append(valid_mask)

        return task_vecs, hw_vecs_list, hw_ids_list, valid_masks

    def _select_actions(
        self,
        task_vecs: List[np.ndarray],
        hw_vecs_list: List[List[np.ndarray]],
        valid_masks: List[np.ndarray]
    ) -> Tuple[List[int], List[float], List[float]]:
        """Select actions for multiple states."""
        action_indices = []
        log_probs = []
        values = []

        with torch.no_grad():
            for task_vec, hw_vecs, valid_mask in zip(task_vecs, hw_vecs_list, valid_masks):
                task_tensor = torch.FloatTensor(task_vec).to(self.device)
                hw_tensors = [torch.FloatTensor(hv).to(self.device) for hv in hw_vecs]
                mask_tensor = torch.BoolTensor(valid_mask).to(self.device)

                action_probs, value, _ = self.policy.module.forward(
                    task_tensor, hw_tensors, mask_tensor
                )

                probs = action_probs.cpu().numpy()
                action_idx = np.random.choice(len(probs), p=probs)
                log_prob = np.log(probs[action_idx] + 1e-8)

                action_indices.append(action_idx)
                log_probs.append(log_prob)
                values.append(value.item())

        return action_indices, log_probs, values

    def collect_rollouts(
        self,
        vec_env: VectorizedEnv,
        rollout_steps: int
    ) -> Tuple[int, List[float]]:
        """Collect rollouts from vectorized environment."""
        states = vec_env.get_states()
        episode_rewards_this_rollout = []
        current_episode_rewards = [0.0] * vec_env.num_envs
        steps_collected = 0

        for _ in range(rollout_steps):
            task_vecs, hw_vecs_list, hw_ids_list, valid_masks = self._encode_states(states)
            action_indices, log_probs, values = self._select_actions(task_vecs, hw_vecs_list, valid_masks)

            actions = []
            for action_idx, hw_ids in zip(action_indices, hw_ids_list):
                if action_idx < len(hw_ids):
                    actions.append(hw_ids[action_idx])
                else:
                    actions.append(-1)

            next_states, rewards, dones, truncateds, infos = vec_env.step(actions)

            for i in range(vec_env.num_envs):
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
                self.buffer.store(exp)
                current_episode_rewards[i] += rewards[i]

                if dones[i] or truncateds[i]:
                    episode_rewards_this_rollout.append(current_episode_rewards[i])
                    current_episode_rewards[i] = 0.0
                    self.episodes_completed += 1

            states = next_states
            steps_collected += vec_env.num_envs
            self.current_timestep += vec_env.num_envs

        return steps_collected, episode_rewards_this_rollout

    def update(self) -> Dict[str, float]:
        """Perform PPO update with gradient synchronization."""
        self.buffer.compute_advantages()
        data = self.buffer.get_all()

        if not data:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy_loss': 0.0}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy_loss = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(len(data))

            for start in range(0, len(data), self.config.batch_size):
                end = min(start + self.config.batch_size, len(data))
                batch_indices = indices[start:end]

                batch_policy_loss = torch.tensor(0.0, device=self.device)
                batch_value_loss = torch.tensor(0.0, device=self.device)
                batch_entropy_loss = torch.tensor(0.0, device=self.device)

                for idx in batch_indices:
                    exp, advantage, ret = data[idx]

                    task_tensor = torch.FloatTensor(exp.task_vec).to(self.device)
                    hw_tensors = [torch.FloatTensor(hv).to(self.device) for hv in exp.hw_vecs]
                    mask_tensor = torch.BoolTensor(exp.valid_mask).to(self.device)

                    action_probs, value, _ = self.policy.module.forward(
                        task_tensor, hw_tensors, mask_tensor
                    )

                    new_log_prob = torch.log(action_probs[exp.action_idx] + 1e-8)
                    old_log_prob = exp.log_prob

                    ratio = torch.exp(new_log_prob - old_log_prob)
                    adv_tensor = torch.tensor(advantage, device=self.device)

                    surr1 = ratio * adv_tensor
                    surr2 = torch.clamp(
                        ratio,
                        1 - self.config.clip_range,
                        1 + self.config.clip_range
                    ) * adv_tensor

                    policy_loss = -torch.min(surr1, surr2)

                    ret_tensor = torch.tensor(ret, device=self.device)
                    value_loss = 0.5 * (value - ret_tensor) ** 2

                    entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum()
                    entropy_loss = -0.01 * entropy

                    batch_policy_loss = batch_policy_loss + policy_loss
                    batch_value_loss = batch_value_loss + value_loss
                    batch_entropy_loss = batch_entropy_loss + entropy_loss

                batch_size = len(batch_indices)
                loss = (batch_policy_loss + 0.5 * batch_value_loss + batch_entropy_loss) / batch_size

                self.optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += batch_policy_loss.item() / batch_size
                total_value_loss += batch_value_loss.item() / batch_size
                total_entropy_loss += batch_entropy_loss.item() / batch_size
                n_updates += 1

        self.buffer.clear()

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy_loss': total_entropy_loss / max(n_updates, 1)
        }

    def train(
        self,
        total_timesteps: int,
        num_envs_per_gpu: int = 4,
        rollout_steps: int = 512,
        env_kwargs: Dict = None,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run distributed training.

        Args:
            total_timesteps: Total timesteps across all processes
            num_envs_per_gpu: Number of parallel environments per GPU
            rollout_steps: Steps per rollout before update
            env_kwargs: Arguments for environment creation
            callback: Optional callback(timestep, info)
        """
        vec_env = VectorizedEnv(num_envs_per_gpu, env_kwargs or {})
        vec_env.reset(seed=self.rank * 1000)

        timesteps_per_process = total_timesteps // self.world_size
        training_stats = {
            'episode_rewards': [],
            'policy_losses': [],
            'value_losses': [],
            'entropy_losses': []
        }

        if self.is_main:
            logger.info(f"Starting distributed training: {self.world_size} GPUs, "
                       f"{num_envs_per_gpu} envs/GPU, {total_timesteps} total timesteps")

        while self.current_timestep < timesteps_per_process:
            steps, ep_rewards = self.collect_rollouts(vec_env, rollout_steps)
            self.episode_rewards.extend(ep_rewards)
            training_stats['episode_rewards'].extend(ep_rewards)

            update_stats = self.update()
            training_stats['policy_losses'].append(update_stats['policy_loss'])
            training_stats['value_losses'].append(update_stats['value_loss'])
            training_stats['entropy_losses'].append(update_stats['entropy_loss'])

            if self.is_main and self.current_timestep % 5000 < rollout_steps * num_envs_per_gpu:
                avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                logger.info(
                    f"Step {self.current_timestep * self.world_size}/{total_timesteps} | "
                    f"Episodes: {self.episodes_completed} | "
                    f"Avg Reward: {avg_reward:.3f}"
                )

            if callback and self.is_main:
                callback(self.current_timestep * self.world_size, update_stats)

        return training_stats

    def get_policy_state_dict(self) -> Dict:
        """Get policy state dict (unwrapped from DDP)."""
        return self.policy.module.state_dict()


def setup_distributed(rank: int, world_size: int, backend: str = 'nccl'):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def train_worker(
    rank: int,
    world_size: int,
    config: RLTrainingConfig,
    total_timesteps: int,
    save_path: str,
    num_envs_per_gpu: int,
    rollout_steps: int,
    env_kwargs: Dict,
    result_queue: mp.Queue
):
    """Worker function for distributed training."""
    try:
        setup_distributed(rank, world_size)
        device = torch.device(f'cuda:{rank}')

        encoder = StateEncoder()
        policy = PolicyNetwork(
            task_dim=encoder.task_dim,
            hw_dim=encoder.hw_dim,
            embed_dim=64
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
            env_kwargs=env_kwargs
        )

        if rank == 0:
            policy_state = trainer.get_policy_state_dict()
            save_dict = {
                'policy_state_dict': policy_state,
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

            result_queue.put({
                'success': True,
                'stats': stats,
                'episodes': trainer.episodes_completed,
                'avg_reward': np.mean(trainer.episode_rewards[-100:]) if trainer.episode_rewards else 0
            })

        cleanup_distributed()

    except Exception as e:
        logger.error(f"Worker {rank} failed: {e}")
        if rank == 0:
            result_queue.put({'success': False, 'error': str(e)})
        raise


def run_distributed_training(
    total_timesteps: int = 100000,
    save_path: str = "models/rl/ppo/model_distributed.pth",
    num_gpus: int = None,
    num_envs_per_gpu: int = 4,
    rollout_steps: int = 512,
    learning_rate: float = 3e-4,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    env_preset: str = 'medium'
) -> Dict[str, Any]:
    """
    Launch distributed training across multiple GPUs.

    Args:
        total_timesteps: Total training timesteps
        save_path: Path to save the trained model
        num_gpus: Number of GPUs to use (None = all available)
        num_envs_per_gpu: Parallel environments per GPU
        rollout_steps: Steps per rollout before PPO update
        learning_rate: Optimizer learning rate
        batch_size: PPO batch size
        n_epochs: PPO epochs per update
        gamma: Discount factor
        env_preset: Environment preset ('small', 'medium', 'large')

    Returns:
        Training results dictionary
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available. Multi-GPU training requires GPUs.")

    logger.info(f"Launching distributed training on {num_gpus} GPUs")
    logger.info(f"Total timesteps: {total_timesteps}, Envs per GPU: {num_envs_per_gpu}")

    config = RLTrainingConfig(
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        total_timesteps=total_timesteps
    )

    env_kwargs = {'preset': env_preset}
    result_queue = mp.Queue()

    mp.spawn(
        train_worker,
        args=(
            num_gpus,
            config,
            total_timesteps,
            save_path,
            num_envs_per_gpu,
            rollout_steps,
            env_kwargs,
            result_queue
        ),
        nprocs=num_gpus,
        join=True
    )

    try:
        result = result_queue.get(timeout=10)
        return result
    except Empty:
        return {'success': False, 'error': 'No result from workers'}
