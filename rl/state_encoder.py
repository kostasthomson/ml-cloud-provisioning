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
    """

    TASK_FEATURES = 12
    GLOBAL_FEATURES = 5
    TASK_GLOBAL_DIM = TASK_FEATURES + GLOBAL_FEATURES  # 17
    HW_FEATURES = 16

    def __init__(self):
        self.task_dim = self.TASK_GLOBAL_DIM
        self.hw_dim = self.HW_FEATURES

    def encode(self, state: RLState) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
        """
        Encode RLState into task embedding and list of HW embeddings.

        Args:
            state: RLState object

        Returns:
            Tuple of:
            - task_vec: np.ndarray of shape (TASK_GLOBAL_DIM,) - task + global features
            - hw_list: List of (hw_type_id, hw_vec) tuples where hw_vec is (HW_FEATURES,)
        """
        task_vec = self._encode_task(state.task)
        global_vec = self._encode_global(state.global_state)
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
