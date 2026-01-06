import logging
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from entities.schemas import (
    ScoringAllocationRequest, ScoringAllocationResponse, ScoringWeights,
    HardwareTypeStatus, ScoringTaskImplementation, OngoingTask,
    HWScore, MetricBreakdown
)

logger = logging.getLogger(__name__)


@dataclass
class RawMetrics:
    energy_kwh: float
    exec_time_sec: float
    network_util: float
    ram_util: float
    storage_util: float


class ExecutionTimeEstimator:
    @staticmethod
    def estimate(
        task_instructions: float,
        task_vcpus: int,
        task_acc_rho: float,
        hw: HardwareTypeStatus,
        num_vms: int
    ) -> float:
        total_cpu_compute = hw.total_cpus * hw.compute_capability_per_cpu
        total_acc_compute = hw.total_accelerators * hw.accelerator_compute_capability if hw.total_accelerators > 0 else 0

        used_cpu_compute = 0.0
        used_acc_compute = 0.0
        events = []

        for ongoing in hw.ongoing_tasks:
            cpu_compute_used = ongoing.resources_used.vcpus * hw.compute_capability_per_cpu
            acc_compute_used = ongoing.accelerator_rho * hw.accelerator_compute_capability if ongoing.accelerator_rho > 0 else 0
            used_cpu_compute += cpu_compute_used
            used_acc_compute += acc_compute_used
            events.append((
                ongoing.estimated_remaining_time_sec,
                cpu_compute_used,
                acc_compute_used
            ))

        events.sort(key=lambda x: x[0])

        available_cpu_compute = max(0, total_cpu_compute - used_cpu_compute)
        available_acc_compute = max(0, total_acc_compute - used_acc_compute)

        task_cpu_need = task_vcpus * num_vms * hw.compute_capability_per_cpu
        task_acc_need = task_acc_rho * hw.accelerator_compute_capability * num_vms if task_acc_rho > 0 else 0

        remaining_instructions = task_instructions
        current_time = 0.0
        event_idx = 0
        max_iterations = 1000

        for _ in range(max_iterations):
            if remaining_instructions <= 0:
                break

            if task_acc_rho > 0 and available_acc_compute > 0:
                effective_rate = min(available_acc_compute, task_acc_need)
            else:
                effective_rate = min(available_cpu_compute, task_cpu_need)

            if effective_rate <= 1e-10:
                if event_idx < len(events):
                    wait_time = events[event_idx][0] - current_time
                    if wait_time > 0:
                        current_time = events[event_idx][0]
                    available_cpu_compute += events[event_idx][1]
                    available_acc_compute += events[event_idx][2]
                    event_idx += 1
                    continue
                else:
                    logger.warning(f"No compute capacity available for task")
                    return float('inf')

            time_to_complete = remaining_instructions / effective_rate

            if event_idx < len(events):
                time_to_next_event = events[event_idx][0] - current_time
                if time_to_next_event > 0 and time_to_next_event < time_to_complete:
                    remaining_instructions -= effective_rate * time_to_next_event
                    current_time = events[event_idx][0]
                    available_cpu_compute += events[event_idx][1]
                    available_acc_compute += events[event_idx][2]
                    event_idx += 1
                    continue

            current_time += time_to_complete
            remaining_instructions = 0

        return current_time


class EnergyEstimator:
    @staticmethod
    def estimate(
        exec_time_sec: float,
        task_vcpus: int,
        task_acc_rho: float,
        hw: HardwareTypeStatus,
        num_vms: int
    ) -> float:
        if exec_time_sec == float('inf') or exec_time_sec <= 0:
            return float('inf')

        current_cpu_util = 1.0 - (hw.available_cpus / max(hw.total_cpus, 1))
        task_cpu_util = (task_vcpus * num_vms) / max(hw.total_cpus, 1)
        new_cpu_util = min(1.0, current_cpu_util + task_cpu_util)

        cpu_power = hw.cpu_idle_power + new_cpu_util * (hw.cpu_max_power - hw.cpu_idle_power)

        acc_power = 0.0
        if task_acc_rho > 0 and hw.total_accelerators > 0:
            acc_power = hw.acc_idle_power + task_acc_rho * (hw.acc_max_power - hw.acc_idle_power)
            acc_power *= num_vms

        total_power_w = cpu_power + acc_power
        energy_kwh = (total_power_w * exec_time_sec) / 3_600_000.0

        return energy_kwh


class UtilizationEstimator:
    @staticmethod
    def estimate_network_util(
        task_network_per_vm: float,
        num_vms: int,
        hw: HardwareTypeStatus
    ) -> float:
        if hw.total_network <= 0:
            return 0.0
        current_util = 1.0 - (hw.available_network / hw.total_network)
        task_util = (task_network_per_vm * num_vms) / hw.total_network
        return min(1.0, current_util + task_util)

    @staticmethod
    def estimate_ram_util(
        task_memory_per_vm: float,
        num_vms: int,
        hw: HardwareTypeStatus
    ) -> float:
        if hw.total_memory <= 0:
            return 0.0
        current_util = 1.0 - (hw.available_memory / hw.total_memory)
        task_util = (task_memory_per_vm * num_vms) / hw.total_memory
        return min(1.0, current_util + task_util)

    @staticmethod
    def estimate_storage_util(
        task_storage_per_vm: float,
        num_vms: int,
        hw: HardwareTypeStatus
    ) -> float:
        if hw.total_storage <= 0:
            return 0.0
        current_util = 1.0 - (hw.available_storage / hw.total_storage)
        task_util = (task_storage_per_vm * num_vms) / hw.total_storage
        return min(1.0, current_util + task_util)


class ScoringAllocator:
    def __init__(self):
        self._global_weights = ScoringWeights()
        self._exec_estimator = ExecutionTimeEstimator()
        self._energy_estimator = EnergyEstimator()
        self._util_estimator = UtilizationEstimator()
        self.allocation_count = 0
        self.rejection_count = 0
        logger.info("ScoringAllocator initialized with default weights")

    @property
    def global_weights(self) -> ScoringWeights:
        return self._global_weights

    def set_global_weights(self, weights: ScoringWeights) -> None:
        self._global_weights = weights
        logger.info(f"Global weights updated: {weights.model_dump()}")

    def get_method_name(self) -> str:
        return "scoring_allocator"

    def _is_implementation_compatible(
        self,
        impl: ScoringTaskImplementation,
        hw: HardwareTypeStatus
    ) -> Tuple[bool, Optional[str]]:
        if impl.requires_accelerator:
            if hw.total_accelerators <= 0:
                return False, "HW has no accelerators"
            if hw.hw_type_id != impl.impl_id:
                return False, f"Impl {impl.impl_id} not compatible with HW type {hw.hw_type_id}"
        return True, None

    def _has_sufficient_resources(
        self,
        impl: ScoringTaskImplementation,
        hw: HardwareTypeStatus,
        num_vms: int
    ) -> Tuple[bool, Optional[str]]:
        total_cpus_needed = impl.vcpus_per_vm * num_vms
        total_memory_needed = impl.memory_per_vm * num_vms
        total_storage_needed = impl.storage_per_vm * num_vms
        total_network_needed = impl.network_per_vm * num_vms

        if hw.available_cpus < total_cpus_needed:
            return False, f"Insufficient CPUs: need {total_cpus_needed}, have {hw.available_cpus}"
        if hw.available_memory < total_memory_needed:
            return False, f"Insufficient memory: need {total_memory_needed}, have {hw.available_memory}"
        if hw.available_storage < total_storage_needed:
            return False, f"Insufficient storage: need {total_storage_needed}, have {hw.available_storage}"
        if hw.available_network < total_network_needed:
            return False, f"Insufficient network: need {total_network_needed}, have {hw.available_network}"

        if impl.requires_accelerator:
            if hw.available_accelerators < num_vms:
                return False, f"Insufficient accelerators: need {num_vms}, have {hw.available_accelerators}"

        return True, None

    def _calculate_raw_metrics(
        self,
        impl: ScoringTaskImplementation,
        hw: HardwareTypeStatus,
        num_vms: int
    ) -> RawMetrics:
        exec_time = self._exec_estimator.estimate(
            task_instructions=impl.instructions,
            task_vcpus=impl.vcpus_per_vm,
            task_acc_rho=impl.accelerator_rho,
            hw=hw,
            num_vms=num_vms
        )

        energy = self._energy_estimator.estimate(
            exec_time_sec=exec_time,
            task_vcpus=impl.vcpus_per_vm,
            task_acc_rho=impl.accelerator_rho,
            hw=hw,
            num_vms=num_vms
        )

        network_util = self._util_estimator.estimate_network_util(
            impl.network_per_vm, num_vms, hw
        )
        ram_util = self._util_estimator.estimate_ram_util(
            impl.memory_per_vm, num_vms, hw
        )
        storage_util = self._util_estimator.estimate_storage_util(
            impl.storage_per_vm, num_vms, hw
        )

        return RawMetrics(
            energy_kwh=energy,
            exec_time_sec=exec_time,
            network_util=network_util,
            ram_util=ram_util,
            storage_util=storage_util
        )

    def _normalize_metrics(
        self,
        raw_metrics: Dict[Tuple[int, int], RawMetrics]
    ) -> Dict[Tuple[int, int], Dict[str, float]]:
        if not raw_metrics:
            return {}

        metric_names = ['energy_kwh', 'exec_time_sec', 'network_util', 'ram_util', 'storage_util']
        min_vals = {m: float('inf') for m in metric_names}
        max_vals = {m: float('-inf') for m in metric_names}

        for raw in raw_metrics.values():
            for m in metric_names:
                val = getattr(raw, m)
                if val != float('inf'):
                    min_vals[m] = min(min_vals[m], val)
                    max_vals[m] = max(max_vals[m], val)

        normalized = {}
        for key, raw in raw_metrics.items():
            norm = {}
            for m in metric_names:
                val = getattr(raw, m)
                if val == float('inf'):
                    norm[m] = 1.0
                elif max_vals[m] == min_vals[m]:
                    norm[m] = 0.0
                else:
                    norm[m] = (val - min_vals[m]) / (max_vals[m] - min_vals[m])
            normalized[key] = norm

        return normalized

    def allocate(self, request: ScoringAllocationRequest) -> ScoringAllocationResponse:
        self.allocation_count += 1
        weights = request.weights or self._global_weights

        logger.info(
            f"Scoring allocation for task {request.task_id}: "
            f"{len(request.implementations)} implementations, "
            f"{len(request.hw_types)} HW types"
        )

        candidates = []
        raw_metrics: Dict[Tuple[int, int], RawMetrics] = {}
        infeasible_scores: List[HWScore] = []

        for impl in request.implementations:
            compatible_hw_found = False
            for hw in request.hw_types:
                is_compatible, compat_reason = self._is_implementation_compatible(impl, hw)
                if not is_compatible:
                    infeasible_scores.append(HWScore(
                        hw_type_id=hw.hw_type_id,
                        hw_type_name=hw.hw_type_name,
                        impl_id=impl.impl_id,
                        total_score=float('inf'),
                        feasible=False,
                        rejection_reason=compat_reason,
                        metrics={}
                    ))
                    continue

                compatible_hw_found = True
                has_resources, resource_reason = self._has_sufficient_resources(
                    impl, hw, request.num_vms
                )
                if not has_resources:
                    infeasible_scores.append(HWScore(
                        hw_type_id=hw.hw_type_id,
                        hw_type_name=hw.hw_type_name,
                        impl_id=impl.impl_id,
                        total_score=float('inf'),
                        feasible=False,
                        rejection_reason=resource_reason,
                        metrics={}
                    ))
                    continue

                metrics = self._calculate_raw_metrics(impl, hw, request.num_vms)
                key = (hw.hw_type_id, impl.impl_id)
                raw_metrics[key] = metrics
                candidates.append((impl, hw, metrics))

            if not compatible_hw_found and impl.impl_id == 1:
                for hw in request.hw_types:
                    if hw.hw_type_id == 1:
                        has_resources, resource_reason = self._has_sufficient_resources(
                            impl, hw, request.num_vms
                        )
                        if has_resources:
                            metrics = self._calculate_raw_metrics(impl, hw, request.num_vms)
                            key = (hw.hw_type_id, impl.impl_id)
                            if key not in raw_metrics:
                                raw_metrics[key] = metrics
                                candidates.append((impl, hw, metrics))

        if not candidates:
            self.rejection_count += 1
            logger.warning(f"No feasible candidates for task {request.task_id}")
            return ScoringAllocationResponse(
                success=False,
                all_scores=infeasible_scores,
                weights_used=weights,
                reason="No feasible (implementation, hardware) combinations found",
                timestamp=request.timestamp
            )

        normalized = self._normalize_metrics(raw_metrics)

        feasible_scores: List[HWScore] = []
        for impl, hw, raw in candidates:
            key = (hw.hw_type_id, impl.impl_id)
            norm = normalized[key]

            energy_weighted = weights.energy * norm['energy_kwh']
            exec_weighted = weights.exec_time * norm['exec_time_sec']
            network_weighted = weights.network * norm['network_util']
            ram_weighted = weights.ram * norm['ram_util']
            storage_weighted = weights.storage * norm['storage_util']

            total_score = energy_weighted + exec_weighted + network_weighted + ram_weighted + storage_weighted

            feasible_scores.append(HWScore(
                hw_type_id=hw.hw_type_id,
                hw_type_name=hw.hw_type_name,
                impl_id=impl.impl_id,
                total_score=total_score,
                feasible=True,
                metrics={
                    'energy': MetricBreakdown(
                        raw_value=raw.energy_kwh,
                        normalized=norm['energy_kwh'],
                        weighted=energy_weighted
                    ),
                    'exec_time': MetricBreakdown(
                        raw_value=raw.exec_time_sec,
                        normalized=norm['exec_time_sec'],
                        weighted=exec_weighted
                    ),
                    'network': MetricBreakdown(
                        raw_value=raw.network_util,
                        normalized=norm['network_util'],
                        weighted=network_weighted
                    ),
                    'ram': MetricBreakdown(
                        raw_value=raw.ram_util,
                        normalized=norm['ram_util'],
                        weighted=ram_weighted
                    ),
                    'storage': MetricBreakdown(
                        raw_value=raw.storage_util,
                        normalized=norm['storage_util'],
                        weighted=storage_weighted
                    )
                }
            ))

        feasible_scores.sort(key=lambda x: x.total_score)
        best = feasible_scores[0]

        best_raw = raw_metrics[(best.hw_type_id, best.impl_id)]

        all_scores = feasible_scores + infeasible_scores

        logger.info(
            f"Task {request.task_id} allocated to HW {best.hw_type_id} ({best.hw_type_name}) "
            f"with impl {best.impl_id}, score={best.total_score:.4f}, "
            f"exec_time={best_raw.exec_time_sec:.2f}s, energy={best_raw.energy_kwh:.4f}kWh"
        )

        return ScoringAllocationResponse(
            success=True,
            selected_hw_type_id=best.hw_type_id,
            selected_impl_id=best.impl_id,
            total_score=best.total_score,
            estimated_exec_time_sec=best_raw.exec_time_sec,
            estimated_energy_kwh=best_raw.energy_kwh,
            all_scores=all_scores,
            weights_used=weights,
            reason=f"Optimal score {best.total_score:.4f}: HW{best.hw_type_id} with impl{best.impl_id}",
            timestamp=request.timestamp
        )

    def get_statistics(self) -> dict:
        return {
            "total_allocations": self.allocation_count,
            "rejections": self.rejection_count,
            "success_rate": ((self.allocation_count - self.rejection_count) /
                             max(self.allocation_count, 1)) * 100,
            "global_weights": self._global_weights.model_dump()
        }

    def reset(self) -> None:
        self.allocation_count = 0
        self.rejection_count = 0
