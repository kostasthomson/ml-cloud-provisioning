"""
RL-based task allocator using infrastructure-agnostic PPO agent.

This allocator adapts the RL agent to the BaseAllocator interface,
allowing seamless integration with the C++ simulator via /allocate_task endpoint.
"""
from typing import Optional, Tuple, List
import logging

from entities.schemas import AllocationRequest, VMAllocation, CellStatus, HardwareType
from .base_allocator import BaseAllocator
from rl.agent import RLAgent
from rl.schemas import RLState, TaskState, HWTypeState, GlobalState

logger = logging.getLogger(__name__)


class RLAllocator(BaseAllocator):
    """
    Allocator using reinforcement learning agent for energy-efficient decisions.

    The RL agent scores each hardware type independently and selects the one
    with highest expected reward (considering energy efficiency and SLA).
    """

    def __init__(self, model_path: str = "models/rl/ppo/model_agnostic.pth"):
        self.agent = RLAgent(model_path=model_path)
        super().__init__()

        logger.info(f"RLAllocator initialized with model: {model_path}")
        logger.info(f"Model trained: {self.agent.is_trained}, timesteps: {self.agent.training_timesteps}")

    def get_method_name(self) -> str:
        return "rl_ppo_agnostic"

    def _perform_allocation(
            self,
            request: AllocationRequest
    ) -> Optional[Tuple[List[VMAllocation], float, str]]:
        """
        Use RL agent to decide which HW type to allocate task to.
        """
        rl_state = self._convert_to_rl_state(request)

        if not rl_state.hw_types:
            logger.warning("No valid HW types available for allocation")
            return None

        action, state_value, inference_time_ms = self.agent.predict(
            state=rl_state,
            deterministic=True
        )

        logger.debug(
            f"RL prediction: action={action.action_name}, "
            f"confidence={action.confidence:.3f}, "
            f"value={state_value:.3f}, "
            f"time={inference_time_ms:.2f}ms"
        )

        if action.action == -1 or action.hw_type_id is None:
            logger.info(f"RL agent rejected task {request.task.task_id}")
            return None

        selected_hw_type_id = action.hw_type_id

        selected_cell = None
        selected_hw_type = None

        for cell in request.cells:
            for hw_type in cell.hw_types:
                if hw_type.hw_type_id == selected_hw_type_id:
                    if self._has_sufficient_resources(request.task, hw_type, cell):
                        selected_cell = cell
                        selected_hw_type = hw_type
                        break
            if selected_cell:
                break

        if not selected_cell or not selected_hw_type:
            logger.warning(
                f"RL selected HW type {selected_hw_type_id} but no resources available. "
                f"Probs: {action.action_probs}"
            )
            return None

        if not self._is_compatible(request.task, selected_hw_type):
            logger.warning(
                f"RL selected incompatible HW type {selected_hw_type_id} for task implementation "
                f"{request.task.implementation_id}"
            )
            return None

        vm_allocations = self._allocate_vms_to_servers(
            request.task, selected_cell, selected_hw_type
        )

        if not vm_allocations:
            logger.warning("Failed to allocate VMs to servers")
            return None

        energy_cost = self._estimate_energy_cost(request.task, selected_hw_type, selected_cell)

        reason = (
            f"RL agent selected HW{selected_hw_type_id} "
            f"(confidence={action.confidence:.2f}, value={state_value:.2f})"
        )

        return vm_allocations, energy_cost, reason

    def _convert_to_rl_state(self, request: AllocationRequest) -> RLState:
        """
        Convert AllocationRequest to RLState format expected by RL agent.
        """
        task = request.task

        compatible_hw_types = self._get_compatible_hw_types(request)

        instructions = 1e10
        if task.estimated_duration:
            instructions = task.estimated_duration * 4400 * task.vcpus_per_vm * task.num_vms

        task_state = TaskState(
            task_id=task.task_id,
            num_vms=task.num_vms,
            vcpus_per_vm=task.vcpus_per_vm,
            memory_per_vm=task.memory_per_vm,
            storage_per_vm=task.storage_per_vm,
            network_per_vm=task.network_per_vm,
            instructions=instructions,
            compatible_hw_types=compatible_hw_types,
            requires_accelerator=task.requires_accelerator,
            accelerator_rho=task.accelerator_utilization,
            deadline=task.estimated_duration
        )

        hw_type_states = []
        for cell in request.cells:
            for hw_type in cell.hw_types:
                available = cell.available_resources.get(hw_type.hw_type_id, {})
                utilization = cell.current_utilization.get(hw_type.hw_type_id, {})

                total_cpus = hw_type.num_servers * hw_type.num_cpus_per_server
                total_memory = hw_type.num_servers * hw_type.memory_per_server
                total_storage = hw_type.num_servers * hw_type.storage_per_server
                total_accelerators = hw_type.num_servers * hw_type.num_accelerators_per_server

                avail_cpus = available.get('cpu', total_cpus)
                avail_memory = available.get('memory', total_memory)
                avail_storage = available.get('storage', total_storage)
                avail_network = available.get('network', 1.0)
                avail_accelerators = int(available.get('accelerators', total_accelerators))

                util_cpu = utilization.get('cpu', 0.0)
                util_memory = utilization.get('memory', 0.0)
                util_network = utilization.get('network', 0.0)

                hw_state = HWTypeState(
                    hw_type_id=hw_type.hw_type_id,
                    utilization_cpu=util_cpu,
                    utilization_memory=util_memory,
                    utilization_network=util_network,
                    utilization_accelerator=0.0,
                    available_cpus=avail_cpus,
                    available_memory=avail_memory,
                    available_storage=avail_storage,
                    available_network=avail_network,
                    available_accelerators=avail_accelerators,
                    total_cpus=total_cpus,
                    total_memory=total_memory,
                    total_storage=total_storage,
                    total_network=1.0,
                    total_accelerators=total_accelerators,
                    compute_capability=hw_type.compute_capability,
                    accelerator_compute_capability=hw_type.accelerator_compute_capability,
                    power_idle=hw_type.cpu_idle_power,
                    power_max=hw_type.cpu_power_consumption[-1] if hw_type.cpu_power_consumption else 220.0,
                    acc_power_idle=hw_type.accelerator_idle_power,
                    acc_power_max=hw_type.accelerator_max_power,
                    num_running_tasks=0,
                    avg_remaining_time=0.0
                )
                hw_type_states.append(hw_state)

        global_state = GlobalState(
            timestamp=request.timestamp,
            total_power_consumption=0.0,
            queue_length=0,
            recent_acceptance_rate=1.0,
            recent_avg_energy=0.0
        )

        return RLState(
            task=task_state,
            hw_types=hw_type_states,
            global_state=global_state
        )

    def _get_compatible_hw_types(self, request: AllocationRequest) -> List[int]:
        """
        Determine which HW types are compatible with the task.
        """
        compatible = []
        task = request.task

        for cell in request.cells:
            for hw_type in cell.hw_types:
                if self._is_compatible(task, hw_type):
                    if hw_type.hw_type_id not in compatible:
                        compatible.append(hw_type.hw_type_id)

        return compatible

    def get_agent_info(self) -> dict:
        """Get information about the RL agent."""
        info = self.agent.get_model_info()
        return info.model_dump()
