from typing import Optional, Tuple, List

from config import fast_api_configuration
from entities import AllocationRequest, VMAllocation, HardwareType, CellStatus
from entities.allocator import BaseAllocator


class HeuristicAllocator(BaseAllocator):
    """
    Heuristic-based energy-aware allocation strategy.
    Optimizes for energy efficiency using predefined heuristics.
    """

    def get_method_name(self) -> str:
        return "heuristic_energy_aware"

    def _perform_allocation(
            self,
            request: AllocationRequest
    ) -> Optional[Tuple[List[VMAllocation], float, str]]:
        """
        Heuristic allocation optimizing for energy efficiency.

        Strategy:
        1. Filter cells/HW types that can accommodate ALL VMs
        2. Estimate energy cost for each candidate
        3. Select the one with the lowest energy cost
        """
        task = request.task
        candidates = []

        for cell in request.cells:
            for hw_type in cell.hw_types:
                if not self._is_compatible(task, hw_type):
                    continue

                if not self._has_sufficient_resources(task, hw_type, cell):
                    continue

                energy_cost = self._estimate_energy_cost(task, hw_type, cell)
                efficiency = self._calculate_efficiency_score(hw_type, cell)

                candidates.append({
                    'cell': cell,
                    'hw_type': hw_type,
                    'energy_cost': energy_cost,
                    'efficiency': efficiency,
                    'score': energy_cost * (2.0 - efficiency)
                })

        if not candidates:
            return None

        best = min(candidates, key=lambda x: x['score'])
        vm_allocations = self._allocate_vms_to_servers(
            task, best['cell'], best['hw_type']
        )

        if not vm_allocations:
            return None

        reason = (
            f"Allocated {len(vm_allocations)} VMs to {best['hw_type'].hw_type_name} "
            f"in Cell {best['cell'].cell_id} (Est: {best['energy_cost']:.4f} kWh)"
        )

        return vm_allocations, best['energy_cost'], reason

    def _calculate_efficiency_score(
            self,
            hw_type: HardwareType,
            cell: CellStatus
    ) -> float:
        """Calculate efficiency score for this hardware type in the cell."""
        hw_id = hw_type.hw_type_id
        if hw_id not in cell.available_resources:
            return 0.0

        available = cell.available_resources[hw_id]
        total_cpus = hw_type.num_servers * hw_type.num_cpus_per_server
        total_memory = hw_type.num_servers * hw_type.memory_per_server
        total_accelerators = hw_type.num_servers * hw_type.num_accelerators_per_server

        return self.energy_calc.calculate_server_efficiency(
            available_cpus=available.get('cpu', 0),
            total_cpus=total_cpus,
            available_memory=available.get('memory', 0),
            total_memory=total_memory,
            available_accelerators=available.get('accelerators', 0),
            total_accelerators=total_accelerators
        )
