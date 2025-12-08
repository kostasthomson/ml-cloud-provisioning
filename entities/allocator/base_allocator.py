from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import logging

from config import fast_api_configuration
from entities import AllocationRequest, AllocationDecision, VMAllocation, CellStatus, HardwareType
from services import EnergyCalculator
from utils import AllocationLogger

logger = logging.getLogger(__name__)


class BaseAllocator(ABC):
    """
    Abstract base class for task allocators.
    Defines the common interface and shared functionality for all allocation strategies.
    """

    def __init__(self):
        """Initialize common components for all allocators."""
        fast_api_configuration.model_type = self.get_method_name()
        self.energy_calc = EnergyCalculator()
        self.allocation_logger = AllocationLogger()
        self.allocation_count = 0
        self.rejection_count = 0

    def allocate_task(self, request: AllocationRequest) -> AllocationDecision:
        """
        Main allocation decision logic with template method pattern.
        This method orchestrates the allocation process and is final.

        Args:
            request: Allocation request with system state and task requirements

        Returns:
            AllocationDecision with selected cell/server or rejection
        """
        self.allocation_count += 1
        decision = None

        try:
            allocation = self._perform_allocation(request)

            if allocation:
                vm_allocations, energy_cost, reason = allocation
                logger.info(
                    f"Task {request.task.task_id} allocated {len(vm_allocations)} VMs, "
                    f"Est. Energy: {energy_cost:.4f} kWh"
                )

                decision = AllocationDecision(
                    success=True,
                    num_vms_allocated=len(vm_allocations),
                    vm_allocations=vm_allocations,
                    estimated_energy_cost=energy_cost,
                    reason=reason,
                    allocation_method=self.get_method_name(),
                    timestamp=request.timestamp
                )
            else:
                self.rejection_count += 1
                logger.warning(f"Task {request.task.task_id} rejected - no suitable resources")
                decision = AllocationDecision(
                    success=False,
                    num_vms_allocated=0,
                    reason="No suitable resources available",
                    allocation_method=self.get_method_name(),
                    timestamp=request.timestamp
                )

        except Exception as e:
            logger.error(f"Error allocating task {request.task.task_id}: {str(e)}")
            decision = AllocationDecision(
                success=False,
                reason=f"Internal error: {str(e)}",
                allocation_method=self.get_method_name(),
                timestamp=request.timestamp
            )

        finally:
            self.allocation_logger.log_decision(request, decision)

        return decision

    @abstractmethod
    def _perform_allocation(
            self,
            request: AllocationRequest
    ) -> Optional[Tuple[List[VMAllocation], float, str]]:
        """
        Abstract method to be implemented by concrete allocators.
        Each allocator implements its own allocation strategy.

        Args:
            request: Allocation request

        Returns:
            Tuple of (vm_allocations, energy_cost, reason) or None if no allocation
        """
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the allocation method."""
        pass

    # Shared helper methods that all allocators can use
    def _is_compatible(self, task, hw_type: HardwareType) -> bool:
        """Check if hardware type is compatible with task implementation."""
        if task.implementation_id == 1:
            return True
        elif task.implementation_id == 2:
            return hw_type.accelerators > 0 and "GPU" in hw_type.hw_type_name.upper()
        elif task.implementation_id == 3:
            return hw_type.accelerators > 0 and "DFE" in hw_type.hw_type_name.upper()
        elif task.implementation_id == 4:
            return hw_type.accelerators > 0 and "MIC" in hw_type.hw_type_name.upper()
        return False

    def _has_sufficient_resources(
            self,
            task,
            hw_type: HardwareType,
            cell: CellStatus
    ) -> bool:
        """Check if sufficient resources are available."""
        hw_id = hw_type.hw_type_id
        if hw_id not in cell.available_resources:
            return False

        available = cell.available_resources[hw_id]
        total_cpus_needed = task.num_vms * task.vcpus_per_vm
        total_memory_needed = task.num_vms * task.memory_per_vm
        total_storage_needed = task.num_vms * task.storage_per_vm
        total_network_needed = task.num_vms * task.network_per_vm
        total_accelerators_needed = task.num_vms if task.requires_accelerator else 0

        checks = [
            available.get('cpu', 0) >= total_cpus_needed,
            available.get('memory', 0) >= total_memory_needed,
            available.get('storage', 0) >= total_storage_needed,
            available.get('network', 0) >= total_network_needed,
        ]

        if task.requires_accelerator:
            checks.append(available.get('accelerators', 0) >= total_accelerators_needed)

        return all(checks)

    def _estimate_energy_cost(
            self,
            task,
            hw_type: HardwareType,
            cell: CellStatus
    ) -> float:
        """Estimate energy cost for running task on this hardware."""
        duration = task.estimated_duration if task.estimated_duration else 3600.0
        cpu_utilization = 0.8

        energy = self.energy_calc.estimate_task_energy(
            task_vcpus=task.num_vms * task.vcpus_per_vm,
            task_duration=duration,
            cpu_utilization=cpu_utilization,
            utilization_bins=hw_type.cpu_utilization_bins,
            power_values=hw_type.cpu_power_consumption,
            has_accelerator=task.requires_accelerator,
            accelerator_utilization=task.accelerator_utilization,
            accelerator_idle_power=hw_type.accelerator_idle_power,
            accelerator_max_power=hw_type.accelerator_max_power
        )

        return energy

    def _allocate_vms_to_servers(
            self,
            task,
            cell: CellStatus,
            hw_type: HardwareType
    ) -> List[VMAllocation]:
        """
        Allocate VMs to specific servers using first-fit strategy.

        Args:
            task: Task requirements
            cell: Selected cell
            hw_type: Selected hardware type

        Returns:
            List of VM allocations with server assignments
        """
        vm_allocations = []
        hw_id = hw_type.hw_type_id

        available = cell.available_resources.get(hw_id, {})
        total_servers = hw_type.num_servers
        cpus_per_server = hw_type.num_cpus_per_server
        memory_per_server = hw_type.memory_per_server
        storage_per_server = hw_type.storage_per_server
        accelerators_per_server = hw_type.num_accelerators_per_server

        available_cpus_per_server = [cpus_per_server] * total_servers
        available_memory_per_server = [memory_per_server] * total_servers
        available_storage_per_server = [storage_per_server] * total_servers
        available_accelerators_per_server = [accelerators_per_server] * total_servers

        total_available_cpus = available.get('cpu', 0)
        total_available_memory = available.get('memory', 0)
        total_available_storage = available.get('storage', 0)
        total_available_accelerators = available.get('accelerators', 0)

        total_used_cpus = (total_servers * cpus_per_server) - total_available_cpus
        total_used_memory = (total_servers * memory_per_server) - total_available_memory
        total_used_storage = (total_servers * storage_per_server) - total_available_storage
        total_used_accelerators = (total_servers * accelerators_per_server) - total_available_accelerators

        avg_used_cpus = total_used_cpus / max(total_servers, 1)
        avg_used_memory = total_used_memory / max(total_servers, 1)
        avg_used_storage = total_used_storage / max(total_servers, 1)
        avg_used_accelerators = total_used_accelerators / max(total_servers, 1) if accelerators_per_server > 0 else 0

        for server_idx in range(total_servers):
            available_cpus_per_server[server_idx] = max(0, cpus_per_server - avg_used_cpus)
            available_memory_per_server[server_idx] = max(0, memory_per_server - avg_used_memory)
            available_storage_per_server[server_idx] = max(0, storage_per_server - avg_used_storage)
            if accelerators_per_server > 0:
                available_accelerators_per_server[server_idx] = max(0, accelerators_per_server - avg_used_accelerators)

        for vm_idx in range(task.num_vms):
            allocated = False

            for server_idx in range(total_servers):
                can_fit = (
                        available_cpus_per_server[server_idx] >= task.vcpus_per_vm and
                        available_memory_per_server[server_idx] >= task.memory_per_vm and
                        available_storage_per_server[server_idx] >= task.storage_per_vm
                )

                if task.requires_accelerator:
                    can_fit = can_fit and (available_accelerators_per_server[server_idx] >= 1)

                if can_fit:
                    vm_allocations.append(VMAllocation(
                        vm_index=vm_idx,
                        cell_id=cell.cell_id,
                        hw_type_id=hw_id,
                        server_index=server_idx
                    ))

                    available_cpus_per_server[server_idx] -= task.vcpus_per_vm
                    available_memory_per_server[server_idx] -= task.memory_per_vm
                    available_storage_per_server[server_idx] -= task.storage_per_vm
                    if task.requires_accelerator:
                        available_accelerators_per_server[server_idx] -= 1

                    allocated = True
                    break

            if not allocated:
                logger.warning(
                    f"Could not allocate VM {vm_idx} for task {task.task_id}. "
                    f"Only {len(vm_allocations)}/{task.num_vms} VMs allocated."
                )
                return []

        return vm_allocations

    def get_statistics(self) -> dict:
        """Get allocator statistics."""
        return {
            "total_allocations": self.allocation_count,
            "rejections": self.rejection_count,
            "success_rate": ((self.allocation_count - self.rejection_count) /
                             max(self.allocation_count, 1)) * 100
        }

    def get_logs(self) -> dict:
        return self.allocation_logger.get_summary()

    def save_logs(self) -> bool:
        return self.allocation_logger.save_to_file()

    def reset(self) -> None:
        self.allocation_count = 0
        self.rejection_count = 0
        self.allocation_logger.reset()
