"""
Infrastructure-agnostic Gymnasium environment for cloud resource allocation.

This environment supports any number of hardware types and generates
realistic cloud provisioning scenarios for RL training.
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
import logging

from .schemas import (
    RLState, TaskState, HWTypeState, GlobalState,
    RLExperience, TaskOutcome
)
from .reward import RewardCalculator

logger = logging.getLogger(__name__)


class HWTypeConfig:
    """Configuration for a hardware type with realistic parameters."""

    def __init__(
        self,
        hw_type_id: int,
        name: str,
        total_cpus: int,
        total_memory: float,
        total_storage: float,
        total_network: float,
        total_accelerators: int,
        compute_capability: float,
        accelerator_compute: float,
        power_idle: float,
        power_max: float,
        acc_power_idle: float = 0.0,
        acc_power_max: float = 0.0
    ):
        self.hw_type_id = hw_type_id
        self.name = name
        self.total_cpus = total_cpus
        self.total_memory = total_memory
        self.total_storage = total_storage
        self.total_network = total_network
        self.total_accelerators = total_accelerators
        self.compute_capability = compute_capability
        self.accelerator_compute = accelerator_compute
        self.power_idle = power_idle
        self.power_max = power_max
        self.acc_power_idle = acc_power_idle
        self.acc_power_max = acc_power_max


REALISTIC_HW_CONFIGS = {
    'small': [
        HWTypeConfig(1, "CPU-Standard", 256, 1024, 10, 10, 0, 4400, 0, 163, 220),
        HWTypeConfig(2, "CPU-HighMem", 128, 2048, 20, 10, 0, 4400, 0, 180, 250),
    ],
    'medium': [
        HWTypeConfig(1, "CPU-Standard", 512, 2048, 50, 25, 0, 4400, 0, 163, 220),
        HWTypeConfig(2, "GPU-Tesla-V100", 256, 1024, 25, 25, 16, 4400, 125000, 200, 300, 50, 300),
        HWTypeConfig(3, "CPU-HighMem", 256, 4096, 100, 25, 0, 4400, 0, 180, 250),
    ],
    'large': [
        HWTypeConfig(1, "CPU-Standard", 1000, 4000, 100, 50, 0, 4400, 0, 163, 220),
        HWTypeConfig(2, "GPU-Tesla-V100", 500, 2000, 50, 50, 32, 4400, 125000, 200, 300, 50, 300),
        HWTypeConfig(3, "GPU-A100", 250, 1000, 50, 50, 16, 4400, 312000, 250, 400, 100, 400),
        HWTypeConfig(4, "CPU-HighMem", 500, 8000, 200, 50, 0, 4400, 0, 180, 250),
    ],
    'enterprise': [
        HWTypeConfig(1, "CPU-Standard", 2000, 8000, 200, 100, 0, 4400, 0, 163, 220),
        HWTypeConfig(2, "GPU-V100", 1000, 4000, 100, 100, 64, 4400, 125000, 200, 300, 50, 300),
        HWTypeConfig(3, "GPU-A100", 500, 2000, 100, 100, 32, 4400, 312000, 250, 400, 100, 400),
        HWTypeConfig(4, "DFE-Maxeler", 200, 800, 50, 50, 16, 4400, 50000, 150, 200, 30, 150),
        HWTypeConfig(5, "MIC-KnightsLanding", 400, 1600, 100, 50, 32, 4400, 80000, 180, 260, 40, 200),
        HWTypeConfig(6, "CPU-HighMem", 1000, 16000, 400, 100, 0, 4400, 0, 200, 280),
    ],
    'high_load': [
        HWTypeConfig(1, "CPU-Standard", 128, 512, 20, 10, 0, 4400, 0, 163, 220),
        HWTypeConfig(2, "GPU-Tesla-V100", 64, 256, 10, 10, 4, 4400, 125000, 200, 300, 50, 300),
        HWTypeConfig(3, "CPU-HighMem", 64, 1024, 40, 10, 0, 4400, 0, 180, 250),
    ],
    'stress_test': [
        HWTypeConfig(1, "CPU-Limited", 64, 256, 10, 5, 0, 4400, 0, 163, 220),
        HWTypeConfig(2, "GPU-Limited", 32, 128, 5, 5, 2, 4400, 125000, 200, 300, 50, 300),
    ],
}

DOMAIN_RANDOM_PRESETS = {
    'mixed_capacity': ['small', 'medium', 'large'],
    'constrained_first': ['stress_test', 'high_load', 'medium'],
    'full_spectrum': ['stress_test', 'high_load', 'small', 'medium', 'large'],
    'production': ['high_load', 'medium', 'large', 'enterprise'],
}


class CloudProvisioningEnv:
    """
    Infrastructure-agnostic cloud provisioning environment.

    Supports any number of HW types and realistic workload generation.
    """

    def __init__(
        self,
        hw_configs: Optional[List[HWTypeConfig]] = None,
        preset: str = 'medium',
        episode_length: int = 100,
        max_steps: int = 2048,
        experiences: Optional[List[RLExperience]] = None,
        seed: Optional[int] = None,
        exec_time_noise: float = 0.0,
        energy_noise: float = 0.0,
        task_arrival_noise: float = 0.0,
        reward_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize environment.

        Args:
            hw_configs: Custom hardware configurations
            preset: Hardware preset ('small', 'medium', 'large', 'enterprise')
            episode_length: Steps per episode
            max_steps: Maximum steps before truncation
            experiences: Pre-collected experiences for offline training
            seed: Random seed
            exec_time_noise: Execution time noise factor (0.0-0.5, e.g., 0.15 = ±15%)
            energy_noise: Energy estimation noise factor (0.0-0.5)
            task_arrival_noise: Task characteristic noise factor (0.0-0.3)
            reward_config: Optional dict of RewardCalculator parameters
        """
        if hw_configs:
            self.hw_configs = hw_configs
        else:
            self.hw_configs = REALISTIC_HW_CONFIGS.get(preset, REALISTIC_HW_CONFIGS['medium'])

        self.max_steps = max_steps
        self.episode_length = max_steps
        self.experiences = experiences
        self.experience_idx = 0
        self.reward_calculator = RewardCalculator(**(reward_config or {}))

        self.exec_time_noise = max(0.0, min(0.5, exec_time_noise))
        self.energy_noise = max(0.0, min(0.5, energy_noise))
        self.task_arrival_noise = max(0.0, min(0.3, task_arrival_noise))

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.hw_states: Dict[int, Dict[str, float]] = {}
        self.running_tasks: List[Dict] = []
        self.total_energy = 0.0
        self.accepted_count = 0
        self.rejected_count = 0
        self.timestamp = 0.0

        self._init_hw_states()

    def _init_hw_states(self):
        """Initialize HW states from configs."""
        self.hw_states = {}
        for cfg in self.hw_configs:
            self.hw_states[cfg.hw_type_id] = {
                'available_cpus': float(cfg.total_cpus),
                'available_memory': float(cfg.total_memory),
                'available_storage': float(cfg.total_storage),
                'available_network': float(cfg.total_network),
                'available_accelerators': cfg.total_accelerators,
                'running_tasks': 0,
                'total_remaining_time': 0.0
            }

    def reset(self, seed: Optional[int] = None) -> Tuple[RLState, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.running_tasks = []
        self.total_energy = 0.0
        self.accepted_count = 0
        self.rejected_count = 0
        self.timestamp = np.random.uniform(0, 86400)
        self.experience_idx = 0

        self._init_hw_states()

        state = self._generate_state()
        return state, {}

    def step(self, action: int) -> Tuple[RLState, float, bool, bool, Dict]:
        """
        Execute action and return next state.

        Args:
            action: HW type ID to allocate to, or -1 for reject

        Returns:
            Tuple of (next_state, reward, done, truncated, info)
        """
        current_state = self._generate_state()
        task = current_state.task

        if action == -1:
            accepted = False
            energy = 0.0
            exec_time = 0.0
            self.rejected_count += 1
        else:
            hw_state = self.hw_states.get(action)
            if hw_state is None:
                accepted = False
                energy = 0.0
                exec_time = 0.0
                self.rejected_count += 1
            else:
                cpus_needed = task.num_vms * task.vcpus_per_vm
                mem_needed = task.num_vms * task.memory_per_vm

                can_allocate = (
                    hw_state['available_cpus'] >= cpus_needed and
                    hw_state['available_memory'] >= mem_needed and
                    action in task.compatible_hw_types
                )

                if can_allocate and task.requires_accelerator:
                    can_allocate = hw_state['available_accelerators'] >= task.num_vms

                if can_allocate:
                    accepted = True
                    hw_state['available_cpus'] -= cpus_needed
                    hw_state['available_memory'] -= mem_needed
                    hw_state['running_tasks'] += 1
                    self.accepted_count += 1

                    accs_needed = task.num_vms if task.requires_accelerator else 0
                    if accs_needed > 0:
                        hw_state['available_accelerators'] -= accs_needed

                    cfg = next(c for c in self.hw_configs if c.hw_type_id == action)
                    exec_time = self._estimate_exec_time(task, cfg)
                    energy = self._estimate_energy(task, cfg, exec_time)
                    self.total_energy += energy

                    self.running_tasks.append({
                        'hw_type_id': action,
                        'cpus': cpus_needed,
                        'memory': mem_needed,
                        'accelerators': accs_needed,
                        'remaining_time': exec_time,
                        'energy': energy
                    })
                else:
                    accepted = False
                    energy = 0.0
                    exec_time = 0.0
                    self.rejected_count += 1

        outcome = TaskOutcome(
            task_id=task.task_id,
            action_taken=action,
            hw_type_id=action if action != -1 and accepted else None,
            accepted=accepted,
            energy_consumed_kwh=energy,
            execution_time_sec=exec_time,
            deadline_met=exec_time <= task.deadline if task.deadline and accepted else None,
            sla_violation=exec_time > task.deadline if task.deadline and accepted else False
        )

        reward = self.reward_calculator.compute_reward(outcome, current_state)

        self.current_step += 1
        self.timestamp += np.random.uniform(0.5, 5.0)
        self._update_running_tasks()

        truncated = self.current_step >= self.max_steps
        done = False

        if self.experiences and self.experience_idx >= len(self.experiences):
            done = True

        terminated = done or truncated

        next_state = self._generate_state()

        info = {
            'accepted': accepted,
            'energy': energy,
            'exec_time': exec_time,
            'total_energy': self.total_energy,
            'acceptance_rate': self.accepted_count / max(self.current_step, 1),
            'steps': self.current_step,
            'episode_done': terminated
        }

        return next_state, reward, done, truncated, info

    def _generate_state(self) -> RLState:
        """Generate current state."""
        if self.experiences and self.experience_idx < len(self.experiences):
            state = self.experiences[self.experience_idx].state
            self.experience_idx += 1
            return state

        task = self._generate_task()
        hw_types = self._get_hw_type_states()
        global_state = self._get_global_state()

        return RLState(task=task, hw_types=hw_types, global_state=global_state)

    def _generate_task(self) -> TaskState:
        """Generate a realistic task."""
        task_type = np.random.choice(['small', 'medium', 'large', 'gpu', 'memory_intensive'])

        if task_type == 'small':
            num_vms = np.random.randint(1, 4)
            vcpus = np.random.choice([2, 4])
            memory = np.random.choice([4, 8, 16])
            instructions = np.random.uniform(1e8, 1e10)
            requires_acc = False
        elif task_type == 'medium':
            num_vms = np.random.randint(2, 8)
            vcpus = np.random.choice([4, 8, 16])
            memory = np.random.choice([16, 32, 64])
            instructions = np.random.uniform(1e10, 1e12)
            requires_acc = False
        elif task_type == 'large':
            num_vms = np.random.randint(4, 16)
            vcpus = np.random.choice([8, 16, 32])
            memory = np.random.choice([64, 128, 256])
            instructions = np.random.uniform(1e12, 1e14)
            requires_acc = np.random.random() < 0.3
        elif task_type == 'gpu':
            num_vms = np.random.randint(1, 8)
            vcpus = np.random.choice([8, 16])
            memory = np.random.choice([32, 64, 128])
            instructions = np.random.uniform(1e11, 1e13)
            requires_acc = True
        else:
            num_vms = np.random.randint(1, 4)
            vcpus = np.random.choice([4, 8])
            memory = np.random.choice([128, 256, 512])
            instructions = np.random.uniform(1e9, 1e11)
            requires_acc = False

        compatible = []
        for cfg in self.hw_configs:
            if requires_acc and cfg.total_accelerators > 0:
                compatible.append(cfg.hw_type_id)
            elif not requires_acc:
                compatible.append(cfg.hw_type_id)

        if not compatible:
            compatible = [self.hw_configs[0].hw_type_id]

        has_deadline = np.random.random() < 0.6
        if has_deadline:
            base_time = instructions / 4.4e9
            deadline = base_time * np.random.uniform(1.5, 5.0)
        else:
            deadline = None

        return TaskState(
            task_id=f"task_{self.current_step}_{np.random.randint(1000, 9999)}",
            num_vms=int(num_vms),
            vcpus_per_vm=int(vcpus),
            memory_per_vm=float(memory),
            storage_per_vm=float(np.random.uniform(0.01, 0.5)),
            network_per_vm=float(np.random.uniform(0.001, 0.1)),
            instructions=float(instructions),
            compatible_hw_types=compatible,
            requires_accelerator=requires_acc,
            accelerator_rho=float(np.random.uniform(0.3, 0.9)) if requires_acc else 0.0,
            deadline=deadline
        )

    def _get_hw_type_states(self) -> List[HWTypeState]:
        """Get current HW type states."""
        states = []
        for cfg in self.hw_configs:
            hw = self.hw_states[cfg.hw_type_id]

            util_cpu = 1 - (hw['available_cpus'] / cfg.total_cpus)
            util_mem = 1 - (hw['available_memory'] / cfg.total_memory)
            util_storage = 1 - (hw['available_storage'] / cfg.total_storage)
            util_network = 1 - (hw['available_network'] / cfg.total_network)
            util_acc = 1 - (hw['available_accelerators'] / cfg.total_accelerators) if cfg.total_accelerators > 0 else 0

            avg_remaining = hw['total_remaining_time'] / max(hw['running_tasks'], 1)

            states.append(HWTypeState(
                hw_type_id=cfg.hw_type_id,
                utilization_cpu=float(min(1.0, max(0.0, util_cpu))),
                utilization_memory=float(min(1.0, max(0.0, util_mem))),
                utilization_storage=float(min(1.0, max(0.0, util_storage))),
                utilization_network=float(min(1.0, max(0.0, util_network))),
                utilization_accelerator=float(min(1.0, max(0.0, util_acc))),
                available_cpus=float(hw['available_cpus']),
                available_memory=float(hw['available_memory']),
                available_storage=float(hw['available_storage']),
                available_network=float(hw['available_network']),
                available_accelerators=int(hw['available_accelerators']),
                total_cpus=float(cfg.total_cpus),
                total_memory=float(cfg.total_memory),
                total_storage=float(cfg.total_storage),
                total_network=float(cfg.total_network),
                total_accelerators=cfg.total_accelerators,
                compute_capability=float(cfg.compute_capability),
                accelerator_compute_capability=float(cfg.accelerator_compute),
                power_idle=float(cfg.power_idle),
                power_max=float(cfg.power_max),
                acc_power_idle=float(cfg.acc_power_idle),
                acc_power_max=float(cfg.acc_power_max),
                num_running_tasks=hw['running_tasks'],
                avg_remaining_time=float(avg_remaining)
            ))

        return states

    def _get_global_state(self) -> GlobalState:
        """Get current global state."""
        total_power = sum(
            self._estimate_current_power(cfg)
            for cfg in self.hw_configs
        )

        acceptance_rate = self.accepted_count / max(self.accepted_count + self.rejected_count, 1)
        avg_energy = self.total_energy / max(self.accepted_count, 1)

        return GlobalState(
            timestamp=float(self.timestamp),
            total_power_consumption=float(total_power),
            queue_length=int(np.random.randint(0, 20)),
            recent_acceptance_rate=float(acceptance_rate),
            recent_avg_energy=float(avg_energy)
        )

    def _estimate_current_power(self, cfg: HWTypeConfig) -> float:
        """Estimate current power consumption for a HW type."""
        hw = self.hw_states[cfg.hw_type_id]
        util = 1 - (hw['available_cpus'] / cfg.total_cpus)
        return cfg.power_idle + util * (cfg.power_max - cfg.power_idle)

    def _estimate_exec_time(self, task: TaskState, cfg: HWTypeConfig) -> float:
        """Estimate execution time for task on HW type with optional stochastic noise."""
        if task.requires_accelerator and cfg.accelerator_compute > 0:
            compute = cfg.accelerator_compute * task.num_vms * task.accelerator_rho
        else:
            compute = cfg.compute_capability * task.num_vms * task.vcpus_per_vm

        base_time = task.instructions / max(compute * 1e6, 1)

        min_exec_time = 5.0 + np.random.uniform(0, 10.0)
        base_time = max(base_time, min_exec_time)

        if self.exec_time_noise > 0:
            noise_factor = np.random.uniform(1 - self.exec_time_noise, 1 + self.exec_time_noise)
            return base_time * noise_factor

        return base_time

    def _estimate_energy(self, task: TaskState, cfg: HWTypeConfig, exec_time: float) -> float:
        """Estimate energy consumption in kWh with optional stochastic noise."""
        util_increase = (task.num_vms * task.vcpus_per_vm) / cfg.total_cpus
        power = cfg.power_idle + util_increase * (cfg.power_max - cfg.power_idle)

        if task.requires_accelerator and cfg.acc_power_max > 0:
            power += cfg.acc_power_idle + task.accelerator_rho * (cfg.acc_power_max - cfg.acc_power_idle)

        base_energy = (power * exec_time) / 3600000

        if self.energy_noise > 0:
            noise_factor = np.random.uniform(1 - self.energy_noise, 1 + self.energy_noise)
            return base_energy * noise_factor

        return base_energy

    def _update_running_tasks(self):
        """Update running tasks and release resources."""
        time_step = np.random.uniform(0.5, 2.0)
        completed = []

        for i, task in enumerate(self.running_tasks):
            task['remaining_time'] -= time_step
            if task['remaining_time'] <= 0:
                completed.append(i)

        for i in reversed(completed):
            task = self.running_tasks.pop(i)
            hw = self.hw_states[task['hw_type_id']]
            hw['available_cpus'] += task['cpus']
            hw['available_memory'] += task['memory']
            hw['available_accelerators'] += task.get('accelerators', 0)
            hw['running_tasks'] -= 1

        for hw_id, hw in self.hw_states.items():
            hw['total_remaining_time'] = sum(
                t['remaining_time'] for t in self.running_tasks
                if t['hw_type_id'] == hw_id
            )

    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions."""
        return np.ones(len(self.hw_configs) + 1, dtype=bool)

    def get_num_hw_types(self) -> int:
        """Get number of HW types in this environment."""
        return len(self.hw_configs)

    def get_hw_type_ids(self) -> List[int]:
        """Get list of HW type IDs."""
        return [cfg.hw_type_id for cfg in self.hw_configs]


class DomainRandomizedEnv(CloudProvisioningEnv):
    """
    Environment that samples from multiple presets for domain randomization.

    This improves generalization by exposing the agent to varied resource
    capacities during training.
    """

    def __init__(
        self,
        presets: Optional[List[str]] = None,
        domain_preset: str = 'mixed_capacity',
        preset_weights: Optional[List[float]] = None,
        curriculum: bool = False,
        curriculum_threshold: float = 0.6,
        **kwargs
    ):
        """
        Initialize domain-randomized environment.

        Args:
            presets: List of preset names to sample from
            domain_preset: Named preset group from DOMAIN_RANDOM_PRESETS
            preset_weights: Sampling weights for each preset (uniform if None)
            curriculum: If True, start with harder presets and progress to easier
            curriculum_threshold: Acceptance rate threshold to advance curriculum
            **kwargs: Additional arguments for CloudProvisioningEnv
        """
        if presets is None:
            presets = DOMAIN_RANDOM_PRESETS.get(domain_preset, ['medium'])

        self.available_presets = presets
        self.preset_weights = preset_weights
        self.curriculum = curriculum
        self.curriculum_threshold = curriculum_threshold
        self.curriculum_stage = 0
        self.recent_acceptance_rates: List[float] = []

        initial_preset = self._select_preset()
        self.current_preset = initial_preset

        kwargs.pop('preset', None)
        kwargs.pop('hw_configs', None)

        super().__init__(preset=initial_preset, **kwargs)

        logger.info(f"DomainRandomizedEnv initialized with presets: {self.available_presets}")
        if self.curriculum:
            logger.info(f"Curriculum learning enabled, threshold: {self.curriculum_threshold}")

    def _select_preset(self) -> str:
        """Select a preset based on current strategy."""
        if self.curriculum:
            if self.curriculum_stage < len(self.available_presets):
                return self.available_presets[self.curriculum_stage]
            return self.available_presets[-1]

        if self.preset_weights:
            weights = np.array(self.preset_weights)
            weights = weights / weights.sum()
            return np.random.choice(self.available_presets, p=weights)

        return np.random.choice(self.available_presets)

    def _update_curriculum(self, acceptance_rate: float):
        """Update curriculum stage based on recent performance."""
        self.recent_acceptance_rates.append(acceptance_rate)

        if len(self.recent_acceptance_rates) >= 10:
            avg_rate = np.mean(self.recent_acceptance_rates[-10:])

            if avg_rate >= self.curriculum_threshold:
                if self.curriculum_stage < len(self.available_presets) - 1:
                    self.curriculum_stage += 1
                    logger.info(f"Curriculum advanced to stage {self.curriculum_stage}: "
                               f"{self.available_presets[self.curriculum_stage]}")
                    self.recent_acceptance_rates = []

    def reset(self, seed: Optional[int] = None) -> Tuple[RLState, Dict]:
        """Reset with a potentially new preset."""
        if seed is not None:
            np.random.seed(seed)

        new_preset = self._select_preset()

        if new_preset != self.current_preset:
            self.hw_configs = REALISTIC_HW_CONFIGS.get(new_preset, REALISTIC_HW_CONFIGS['medium'])
            self.current_preset = new_preset

        self.current_step = 0
        self.running_tasks = []
        self.total_energy = 0.0
        self.accepted_count = 0
        self.rejected_count = 0
        self.timestamp = np.random.uniform(0, 86400)
        self.experience_idx = 0

        self._init_hw_states()

        state = self._generate_state()
        return state, {'preset': self.current_preset}

    def step(self, action: int) -> Tuple[RLState, float, bool, bool, Dict]:
        """Execute action and track acceptance for curriculum."""
        next_state, reward, done, truncated, info = super().step(action)

        info['preset'] = self.current_preset

        if done or truncated:
            acceptance_rate = self.accepted_count / max(self.current_step, 1)
            info['episode_acceptance_rate'] = acceptance_rate

            if self.curriculum:
                self._update_curriculum(acceptance_rate)

        return next_state, reward, done, truncated, info

    def get_current_preset(self) -> str:
        """Get the current preset being used."""
        return self.current_preset

    def get_curriculum_stage(self) -> int:
        """Get current curriculum stage."""
        return self.curriculum_stage
