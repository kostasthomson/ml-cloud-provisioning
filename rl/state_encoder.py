"""
Infrastructure-agnostic state encoder for RL agent.

Encodes task and HW type features into fixed-size embeddings that can be
processed by the policy network regardless of the number of HW types.
"""

import numpy as np
from typing import List, Tuple, Dict
import logging

from .schemas import RLState, HWTypeState, TaskState, GlobalState

logger = logging.getLogger(__name__)


class StateEncoder:
    """
    Encodes RLState into separate task and HW embeddings.

    This encoder is infrastructure-agnostic:
    - Task encoding is fixed-size regardless of HW types
    - Each HW type is encoded independently with fixed size
    - Works with any number of HW types (1, 2, 10, 100, etc.)

    Version history:
    - v1: Basic task + global features (17-dim)
    - v2: Added scarcity signals (22-dim)
    - v3: Added capacity scale features (28-dim) - addresses scale blindness
    """

    TASK_FEATURES = 12
    GLOBAL_FEATURES = 5
    SCARCITY_FEATURES = 5
    CAPACITY_FEATURES = 6
    TASK_GLOBAL_DIM_V1 = TASK_FEATURES + GLOBAL_FEATURES  # 17 (legacy)
    TASK_GLOBAL_DIM_V2 = TASK_FEATURES + GLOBAL_FEATURES + SCARCITY_FEATURES  # 22
    TASK_GLOBAL_DIM_V3 = TASK_FEATURES + GLOBAL_FEATURES + SCARCITY_FEATURES + CAPACITY_FEATURES  # 28
    HW_FEATURES = 16

    def __init__(self, use_scarcity_features: bool = True, use_capacity_features: bool = False):
        """
        Initialize state encoder.

        Args:
            use_scarcity_features: If True, include scarcity signals (v2).
            use_capacity_features: If True, include capacity scale signals (v3).
                                   Requires use_scarcity_features=True.
        """
        self.use_scarcity_features = use_scarcity_features
        self.use_capacity_features = use_capacity_features and use_scarcity_features

        if self.use_capacity_features:
            self.task_dim = self.TASK_GLOBAL_DIM_V3
        elif use_scarcity_features:
            self.task_dim = self.TASK_GLOBAL_DIM_V2
        else:
            self.task_dim = self.TASK_GLOBAL_DIM_V1
        self.hw_dim = self.HW_FEATURES

    def encode(self, state: RLState) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
        """
        Encode RLState into task embedding and list of HW embeddings.

        Args:
            state: RLState object

        Returns:
            Tuple of:
            - task_vec: np.ndarray of shape (TASK_GLOBAL_DIM,) - task + global + scarcity features
            - hw_list: List of (hw_type_id, hw_vec) tuples where hw_vec is (HW_FEATURES,)
        """
        task_vec = self._encode_task(state.task)
        global_vec = self._encode_global(state.global_state)

        if self.use_capacity_features:
            scarcity_vec = self._encode_scarcity(state)
            capacity_vec = self._encode_capacity(state)
            task_global_vec = np.concatenate([task_vec, global_vec, scarcity_vec, capacity_vec])
        elif self.use_scarcity_features:
            scarcity_vec = self._encode_scarcity(state)
            task_global_vec = np.concatenate([task_vec, global_vec, scarcity_vec])
        else:
            task_global_vec = np.concatenate([task_vec, global_vec])
        task_global_vec = np.nan_to_num(task_global_vec, nan=0.0, posinf=1.0, neginf=0.0)
        task_global_vec = np.clip(task_global_vec, 0.0, 1.0).astype(np.float32)

        hw_list = []
        for hw in state.hw_types:
            hw_vec = self._encode_single_hw(hw)
            hw_vec = np.nan_to_num(hw_vec, nan=0.0, posinf=1.0, neginf=0.0)
            hw_vec = np.clip(hw_vec, 0.0, 1.0).astype(np.float32)
            hw_list.append((hw.hw_type_id, hw_vec))

        return task_global_vec, hw_list

    def encode_flat(self, state: RLState, max_hw_types: int = 8) -> np.ndarray:
        """
        Encode state into a flat vector for backward compatibility.
        Pads or truncates to max_hw_types HW slots.
        """
        task_vec, hw_list = self.encode(state)

        hw_matrix = np.zeros((max_hw_types, self.HW_FEATURES), dtype=np.float32)
        for i, (_, hw_vec) in enumerate(hw_list[:max_hw_types]):
            hw_matrix[i] = hw_vec

        return np.concatenate([task_vec, hw_matrix.flatten()])

    def _encode_task(self, task: TaskState) -> np.ndarray:
        """Encode task features (12 values)."""
        max_compatible = max(task.compatible_hw_types) if task.compatible_hw_types else 0
        min_compatible = min(task.compatible_hw_types) if task.compatible_hw_types else 0
        num_compatible = len(task.compatible_hw_types)

        features = [
            task.num_vms / 16.0,
            task.vcpus_per_vm / 64.0,
            task.memory_per_vm / 512.0,
            task.storage_per_vm / 10.0,
            task.network_per_vm / 1.0,
            np.log10(max(task.instructions, 1)) / 15.0,
            float(task.requires_accelerator),
            task.accelerator_rho,
            num_compatible / 10.0,
            float(task.deadline is not None),
            min(task.deadline or 0, 3600) / 3600.0 if task.deadline else 0.0,
            (task.num_vms * task.vcpus_per_vm * task.memory_per_vm) / (16 * 64 * 512),
        ]

        return np.array(features, dtype=np.float32)

    def _encode_global(self, global_state: GlobalState) -> np.ndarray:
        """Encode global state (5 values)."""
        time_of_day = (global_state.timestamp % 86400) / 86400.0

        features = [
            global_state.total_power_consumption / 100000.0,
            global_state.queue_length / 100.0,
            global_state.recent_acceptance_rate,
            global_state.recent_avg_energy / 10.0,
            time_of_day,
        ]

        return np.array(features, dtype=np.float32)

    def _encode_scarcity(self, state: RLState) -> np.ndarray:
        """
        Encode scarcity signals (5 values) to help generalization.

        These features help the agent distinguish resource-constrained
        states from abundant ones, improving generalization across
        different capacity environments.
        """
        if not state.hw_types:
            return np.zeros(self.SCARCITY_FEATURES, dtype=np.float32)

        cpu_utils = [hw.utilization_cpu for hw in state.hw_types]
        mem_utils = [hw.utilization_memory for hw in state.hw_types]

        capacity_ratios = [
            hw.available_cpus / max(hw.total_cpus, 1) for hw in state.hw_types
        ]
        mem_ratios = [
            hw.available_memory / max(hw.total_memory, 1) for hw in state.hw_types
        ]

        avg_cpu_util = np.mean(cpu_utils)
        avg_mem_util = np.mean(mem_utils)
        min_cpu_capacity = min(capacity_ratios)
        min_mem_capacity = min(mem_ratios)
        scarcity_indicator = float(any(u > 0.8 for u in cpu_utils) or any(u > 0.8 for u in mem_utils))

        features = [
            avg_cpu_util,
            avg_mem_util,
            min_cpu_capacity,
            min_mem_capacity,
            scarcity_indicator,
        ]

        return np.array(features, dtype=np.float32)

    def _encode_capacity(self, state: RLState) -> np.ndarray:
        """
        Encode capacity scale features (6 values) to address scale blindness.

        These features provide absolute capacity information that helps the agent
        distinguish between environments of different scales (e.g., medium vs stress_test).

        v3 feature set addresses the core issue: utilization ratios hide absolute scale.
        """
        if not state.hw_types:
            return np.zeros(self.CAPACITY_FEATURES, dtype=np.float32)

        total_cpus = sum(hw.total_cpus for hw in state.hw_types)
        total_memory = sum(hw.total_memory for hw in state.hw_types)
        available_cpus = sum(hw.available_cpus for hw in state.hw_types)
        available_memory = sum(hw.available_memory for hw in state.hw_types)

        task = state.task
        task_cpus_needed = task.num_vms * task.vcpus_per_vm
        task_memory_needed = task.num_vms * task.memory_per_vm

        cpu_fit_ratio = available_cpus / max(task_cpus_needed, 1)
        mem_fit_ratio = available_memory / max(task_memory_needed, 1)
        task_fit_ratio = min(cpu_fit_ratio, mem_fit_ratio)

        if total_cpus >= 2000:
            scale_bucket = 1.0  # enterprise
        elif total_cpus >= 1000:
            scale_bucket = 0.75  # large
        elif total_cpus >= 500:
            scale_bucket = 0.5  # medium
        elif total_cpus >= 200:
            scale_bucket = 0.25  # small
        else:
            scale_bucket = 0.1  # stress_test/high_load

        task_relative_size = task_cpus_needed / max(total_cpus, 1)

        features = [
            min(total_cpus / 2000.0, 1.0),
            min(total_memory / 8000.0, 1.0),
            min(cpu_fit_ratio / 10.0, 1.0),
            min(mem_fit_ratio / 10.0, 1.0),
            scale_bucket,
            min(task_relative_size * 10.0, 1.0),
        ]

        return np.array(features, dtype=np.float32)

    def _encode_single_hw(self, hw: HWTypeState) -> np.ndarray:
        """Encode single HW type (16 values)."""
        capacity_ratio = hw.available_cpus / max(hw.total_cpus, 1)
        memory_ratio = hw.available_memory / max(hw.total_memory, 1)
        storage_ratio = hw.available_storage / max(hw.total_storage, 1)
        network_ratio = hw.available_network / max(hw.total_network, 1)
        acc_ratio = hw.available_accelerators / max(hw.total_accelerators, 1) if hw.total_accelerators > 0 else 0.0

        features = [
            hw.utilization_cpu,
            hw.utilization_memory,
            hw.utilization_storage,
            hw.utilization_network,
            hw.utilization_accelerator,
            capacity_ratio,
            memory_ratio,
            storage_ratio,
            network_ratio,
            acc_ratio,
            np.log10(max(hw.compute_capability, 1)) / 10.0,
            np.log10(max(hw.accelerator_compute_capability, 1)) / 10.0 if hw.accelerator_compute_capability > 0 else 0.0,
            hw.power_idle / 500.0,
            hw.power_max / 500.0,
            hw.num_running_tasks / 100.0,
            hw.avg_remaining_time / 3600.0,
        ]

        return np.array(features, dtype=np.float32)

    def get_valid_hw_types(self, state: RLState) -> List[int]:
        """
        Get list of valid HW type IDs that can handle the task.

        Returns:
            List of hw_type_ids that have sufficient resources and are compatible
        """
        task = state.task
        valid_types = []

        for hw in state.hw_types:
            if hw.hw_type_id not in task.compatible_hw_types:
                continue

            total_cpus_needed = task.num_vms * task.vcpus_per_vm
            total_memory_needed = task.num_vms * task.memory_per_vm

            if hw.available_cpus < total_cpus_needed:
                continue
            if hw.available_memory < total_memory_needed:
                continue

            if task.requires_accelerator:
                if hw.available_accelerators < task.num_vms:
                    continue

            valid_types.append(hw.hw_type_id)

        return valid_types

    def can_reject(self, state: RLState) -> bool:
        """Check if rejection is a valid action (always True, but may have penalties)."""
        return True


def create_dummy_state(num_hw_types: int = 3) -> RLState:
    """Create a dummy state for testing/initialization."""
    from .schemas import TaskState, HWTypeState, GlobalState

    task = TaskState(
        task_id="dummy",
        num_vms=2,
        vcpus_per_vm=4,
        memory_per_vm=16.0,
        instructions=1e9,
        compatible_hw_types=list(range(1, num_hw_types + 1))
    )

    hw_types = []
    for i in range(1, num_hw_types + 1):
        hw_types.append(HWTypeState(
            hw_type_id=i,
            utilization_cpu=0.3,
            utilization_memory=0.25,
            available_cpus=500 + i * 100,
            available_memory=2000 + i * 500,
            total_cpus=1000,
            total_memory=5000,
            compute_capability=4400 * i
        ))

    global_state = GlobalState(timestamp=3600.0)

    return RLState(task=task, hw_types=hw_types, global_state=global_state)
