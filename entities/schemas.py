"""
Pydantic entities for request/response validation.
Defines the data structures for communication between C++ simulator and ML service.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class HardwareType(BaseModel):
    """Hardware type specification for a server."""
    hw_type_id: int = Field(..., description="Hardware type ID")
    hw_type_name: str = Field(..., description="Hardware type name (CPU, CPU+GPU, etc.)")
    num_servers: int = Field(..., description="Total number of servers of this type")
    num_cpus_per_server: int = Field(..., description="Number of CPUs per server")
    memory_per_server: float = Field(..., description="Memory per server (GB)")
    storage_per_server: float = Field(..., description="Storage per server (TB)")
    compute_capability: float = Field(..., description="CPU compute capability")
    accelerators: int = Field(..., description="Has accelerators (0 or 1)")
    num_accelerators_per_server: int = Field(..., description="Number of accelerators per server")
    accelerator_compute_capability: float = Field(..., description="Accelerator compute capability")
    cpu_power_consumption: List[float] = Field(..., description="CPU power consumption profile")
    cpu_utilization_bins: List[float] = Field(..., description="CPU utilization bins")
    cpu_idle_power: float = Field(default=0.0, description="CPU idle power consumption")
    accelerator_idle_power: float = Field(default=0.0, description="Accelerator idle power")
    accelerator_max_power: float = Field(default=0.0, description="Accelerator max power")


class CellStatus(BaseModel):
    """Current status of a cell (data center)."""
    cell_id: int = Field(..., description="Cell ID")
    hw_types: List[HardwareType] = Field(..., description="Hardware types in this cell")
    available_resources: Dict[int, Dict[str, float]] = Field(
        ..., 
        description="Available resources per HW type {hw_type_id: {resource_name: value}}"
    )
    current_utilization: Dict[int, Dict[str, float]] = Field(
        ..., 
        description="Current utilization per HW type {hw_type_id: {resource_name: value}}"
    )


class TaskRequirements(BaseModel):
    """Resource requirements for an incoming task."""
    task_id: str = Field(..., description="Unique task identifier")
    application_id: int = Field(..., description="Application ID")
    implementation_id: int = Field(..., description="Implementation ID (1=CPU, 2=GPU, 3=DFE, 4=MIC)")
    num_vms: int = Field(..., description="Number of VMs required")
    vcpus_per_vm: int = Field(..., description="vCPUs per VM")
    memory_per_vm: float = Field(..., description="Memory per VM (GB)")
    storage_per_vm: float = Field(..., description="Storage per VM (TB)")
    network_per_vm: float = Field(..., description="Network bandwidth per VM")
    requires_accelerator: bool = Field(..., description="Whether accelerator is required")
    accelerator_utilization: float = Field(default=0.0, description="Accelerator utilization ratio (rho)")
    estimated_duration: Optional[float] = Field(None, description="Estimated task duration (seconds)")


class AllocationRequest(BaseModel):
    """Request for task allocation decision."""
    timestamp: float = Field(..., description="Current simulation timestamp")
    cells: List[CellStatus] = Field(..., description="Status of all cells")
    task: TaskRequirements = Field(..., description="Task to be allocated")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 100.0,
                "cells": [
                    {
                        "cell_id": 1,
                        "hw_types": [],
                        "available_resources": {1: {"cpu": 500000, "memory": 3200000}},
                        "current_utilization": {1: {"cpu": 0.1, "memory": 0.05}}
                    }
                ],
                "task": {
                    "task_id": "task_001",
                    "application_id": 1,
                    "implementation_id": 1,
                    "num_vms": 2,
                    "vcpus_per_vm": 4,
                    "memory_per_vm": 8.0,
                    "storage_per_vm": 0.02,
                    "network_per_vm": 0.0025,
                    "requires_accelerator": False,
                    "accelerator_utilization": 0.0
                }
            }
        }


class VMAllocation(BaseModel):
    """Allocation details for a single VM."""
    vm_index: int = Field(..., description="VM index (0 to num_vms-1)")
    cell_id: int = Field(..., description="Cell where VM should be deployed")
    hw_type_id: int = Field(..., description="Hardware type for this VM")
    server_index: int = Field(..., description="Server index within the hw_type")


class AllocationDecision(BaseModel):
    """Response containing the allocation decision for multi-VM tasks."""
    success: bool = Field(..., description="Whether allocation is possible")
    num_vms_allocated: int = Field(default=0, description="Number of VMs successfully allocated")
    vm_allocations: List[VMAllocation] = Field(
        default_factory=list,
        description="Allocation details for each VM"
    )
    estimated_energy_cost: Optional[float] = Field(None, description="Estimated energy cost (kWh)")
    reason: Optional[str] = Field(None, description="Explanation of decision")
    allocation_method: str = Field(..., description="Method used for allocation")
    timestamp: float = Field(..., description="Decision timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "num_vms_allocated": 2,
                "vm_allocations": [
                    {"vm_index": 0, "cell_id": 1, "hw_type_id": 1, "server_index": 0},
                    {"vm_index": 1, "cell_id": 1, "hw_type_id": 1, "server_index": 1}
                ],
                "estimated_energy_cost": 0.5,
                "reason": "Optimal energy efficiency",
                "allocation_method": "heuristic_energy_aware",
                "timestamp": 100.0
            }
        }


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model_type: str


class ImplementationOption(BaseModel):
    """A single implementation option for an application."""
    impl_id: int = Field(..., description="Implementation ID")
    impl_name: str = Field(..., description="Implementation name (e.g., CPU-only, GPU-accelerated)")
    num_vms: int = Field(..., description="Number of VMs required")
    vcpus_per_vm: int = Field(..., description="vCPUs per VM")
    memory_per_vm: float = Field(..., description="Memory per VM (GB)")
    storage_per_vm: float = Field(default=0.1, description="Storage per VM (TB)")
    network_per_vm: float = Field(default=0.01, description="Network bandwidth per VM")
    requires_accelerator: bool = Field(default=False, description="Whether accelerator is required")
    accelerator_utilization: float = Field(default=0.0, description="Accelerator utilization ratio (rho)")
    estimated_instructions: float = Field(default=1e9, description="Estimated instructions to execute")


class MultiImplAllocationRequest(BaseModel):
    """Request for multi-implementation allocation decision."""
    timestamp: float = Field(..., description="Current simulation timestamp")
    cells: List[CellStatus] = Field(..., description="Status of all cells")
    application_id: int = Field(..., description="Application ID")
    task_id: str = Field(..., description="Unique task identifier")
    implementations: List[ImplementationOption] = Field(..., description="Available implementations")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 100.0,
                "cells": [],
                "application_id": 1,
                "task_id": "task_001",
                "implementations": [
                    {
                        "impl_id": 1,
                        "impl_name": "CPU-only",
                        "num_vms": 16,
                        "vcpus_per_vm": 8,
                        "memory_per_vm": 32.0,
                        "requires_accelerator": False
                    },
                    {
                        "impl_id": 2,
                        "impl_name": "GPU-accelerated",
                        "num_vms": 4,
                        "vcpus_per_vm": 4,
                        "memory_per_vm": 16.0,
                        "requires_accelerator": True,
                        "accelerator_utilization": 0.9
                    }
                ]
            }
        }


class EnergyPrediction(BaseModel):
    """Energy prediction for a single (implementation, hardware) combination."""
    impl_id: int = Field(..., description="Implementation ID")
    impl_name: str = Field(..., description="Implementation name")
    cell_id: int = Field(..., description="Cell ID")
    hw_type_id: int = Field(..., description="Hardware type ID")
    hw_name: str = Field(..., description="Hardware type name")
    predicted_energy_wh: float = Field(..., description="Predicted energy in Wh")


class MultiImplAllocationDecision(BaseModel):
    """Response for multi-implementation allocation decision."""
    success: bool = Field(..., description="Whether allocation is possible")
    selected_impl_id: Optional[int] = Field(None, description="Selected implementation ID")
    selected_impl_name: Optional[str] = Field(None, description="Selected implementation name")
    num_vms_allocated: int = Field(default=0, description="Number of VMs allocated")
    vm_allocations: List[VMAllocation] = Field(default_factory=list, description="VM allocation details")
    estimated_energy_wh: Optional[float] = Field(None, description="Estimated energy (Wh)")
    all_predictions: List[EnergyPrediction] = Field(default_factory=list, description="All predictions")
    skipped_combinations: List[str] = Field(default_factory=list, description="Skipped combinations")
    reason: Optional[str] = Field(None, description="Explanation of decision")
    allocation_method: str = Field(..., description="Method used for allocation")
    timestamp: float = Field(..., description="Decision timestamp")


class ResourceUsage(BaseModel):
    """Resource consumption by a task."""
    vcpus: float = Field(..., description="vCPUs used")
    memory: float = Field(..., description="Memory used (GB)")
    storage: float = Field(default=0.0, description="Storage used (TB)")
    network: float = Field(default=0.0, description="Network bandwidth used")
    accelerators: int = Field(default=0, description="Accelerators used")


class OngoingTask(BaseModel):
    """Information about a currently running task on a hardware type."""
    task_id: str = Field(..., description="Task identifier")
    remaining_instructions: float = Field(..., description="Instructions left to execute")
    resources_used: ResourceUsage = Field(..., description="Resources consumed by this task")
    estimated_remaining_time_sec: float = Field(..., description="Estimated seconds until completion")
    accelerator_rho: float = Field(default=0.0, description="Accelerator utilization ratio")


class HardwareTypeStatus(BaseModel):
    """Extended hardware type status including ongoing tasks for scoring allocation."""
    hw_type_id: int = Field(..., description="Hardware type ID (1=CPU, 2=GPU, 3=DFE, 4=MIC)")
    hw_type_name: str = Field(..., description="Hardware type name")
    num_servers: int = Field(..., description="Total servers of this type")
    total_cpus: float = Field(..., description="Total CPUs across all servers")
    total_memory: float = Field(..., description="Total memory (GB)")
    total_storage: float = Field(..., description="Total storage (TB)")
    total_network: float = Field(..., description="Total network bandwidth")
    total_accelerators: int = Field(default=0, description="Total accelerators")
    available_cpus: float = Field(..., description="Available CPUs")
    available_memory: float = Field(..., description="Available memory (GB)")
    available_storage: float = Field(..., description="Available storage (TB)")
    available_network: float = Field(..., description="Available network bandwidth")
    available_accelerators: int = Field(default=0, description="Available accelerators")
    compute_capability_per_cpu: float = Field(..., description="MIPS per CPU")
    accelerator_compute_capability: float = Field(default=0.0, description="MIPS per accelerator")
    cpu_idle_power: float = Field(..., description="CPU idle power (W)")
    cpu_max_power: float = Field(..., description="CPU max power (W)")
    acc_idle_power: float = Field(default=0.0, description="Accelerator idle power (W)")
    acc_max_power: float = Field(default=0.0, description="Accelerator max power (W)")
    ongoing_tasks: List[OngoingTask] = Field(default_factory=list, description="Currently running tasks")


class ScoringWeights(BaseModel):
    """Configurable weights for multi-objective scoring (must sum to 1.0)."""
    energy: float = Field(default=0.25, ge=0.0, le=1.0, description="Weight for energy consumption")
    exec_time: float = Field(default=0.25, ge=0.0, le=1.0, description="Weight for execution time")
    network: float = Field(default=0.20, ge=0.0, le=1.0, description="Weight for network utilization")
    ram: float = Field(default=0.15, ge=0.0, le=1.0, description="Weight for RAM utilization")
    storage: float = Field(default=0.15, ge=0.0, le=1.0, description="Weight for storage utilization")

    @validator('storage')
    def weights_must_sum_to_one(cls, v, values):
        total = values.get('energy', 0) + values.get('exec_time', 0) + \
                values.get('network', 0) + values.get('ram', 0) + v
        if abs(total - 1.0) > 0.01:
            raise ValueError(f'Weights must sum to 1.0, got {total}')
        return v


class ScoringTaskImplementation(BaseModel):
    """Task implementation option for scoring-based allocation."""
    impl_id: int = Field(..., description="Implementation ID (maps to compatible HW type)")
    instructions: float = Field(..., description="Total work in instructions/FLOPS")
    vcpus_per_vm: int = Field(..., description="vCPUs required per VM")
    memory_per_vm: float = Field(..., description="Memory per VM (GB)")
    storage_per_vm: float = Field(default=0.1, description="Storage per VM (TB)")
    network_per_vm: float = Field(default=0.01, description="Network bandwidth per VM")
    requires_accelerator: bool = Field(default=False, description="Whether accelerator is required")
    accelerator_rho: float = Field(default=0.0, description="Accelerator utilization ratio (0-1)")


class ScoringAllocationRequest(BaseModel):
    """Request for multi-objective scoring-based allocation."""
    timestamp: float = Field(..., description="Current simulation timestamp")
    task_id: str = Field(..., description="Unique task identifier")
    num_vms: int = Field(..., description="Number of VMs to allocate")
    implementations: List[ScoringTaskImplementation] = Field(..., description="Available task implementations")
    hw_types: List[HardwareTypeStatus] = Field(..., description="Hardware types with current state")
    weights: Optional[ScoringWeights] = Field(None, description="Per-request weight override")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 100.0,
                "task_id": "task_001",
                "num_vms": 4,
                "implementations": [
                    {
                        "impl_id": 1,
                        "instructions": 1e10,
                        "vcpus_per_vm": 4,
                        "memory_per_vm": 16.0,
                        "requires_accelerator": False
                    },
                    {
                        "impl_id": 2,
                        "instructions": 1e10,
                        "vcpus_per_vm": 2,
                        "memory_per_vm": 8.0,
                        "requires_accelerator": True,
                        "accelerator_rho": 0.85
                    }
                ],
                "hw_types": [
                    {
                        "hw_type_id": 1,
                        "hw_type_name": "CPU",
                        "num_servers": 100,
                        "total_cpus": 2000,
                        "total_memory": 12800,
                        "total_storage": 100,
                        "total_network": 40,
                        "available_cpus": 1500,
                        "available_memory": 10000,
                        "available_storage": 80,
                        "available_network": 30,
                        "compute_capability_per_cpu": 4400,
                        "cpu_idle_power": 163,
                        "cpu_max_power": 220,
                        "ongoing_tasks": []
                    }
                ],
                "weights": {
                    "energy": 0.25,
                    "exec_time": 0.25,
                    "network": 0.20,
                    "ram": 0.15,
                    "storage": 0.15
                }
            }
        }


class MetricBreakdown(BaseModel):
    """Breakdown of a single metric's contribution to the score."""
    raw_value: float = Field(..., description="Raw metric value before normalization")
    normalized: float = Field(..., description="Normalized value in [0,1] range")
    weighted: float = Field(..., description="Weighted contribution to final score")


class HWScore(BaseModel):
    """Scoring result for a specific (implementation, hardware) combination."""
    hw_type_id: int = Field(..., description="Hardware type ID")
    hw_type_name: str = Field(..., description="Hardware type name")
    impl_id: int = Field(..., description="Implementation ID used")
    total_score: float = Field(..., description="Total weighted score (lower is better)")
    feasible: bool = Field(..., description="Whether this allocation is feasible")
    rejection_reason: Optional[str] = Field(None, description="Reason if not feasible")
    metrics: Dict[str, MetricBreakdown] = Field(default_factory=dict, description="Per-metric breakdown")


class ScoringAllocationResponse(BaseModel):
    """Response for scoring-based allocation with full breakdown."""
    success: bool = Field(..., description="Whether allocation is possible")
    selected_hw_type_id: Optional[int] = Field(None, description="Selected hardware type ID")
    selected_impl_id: Optional[int] = Field(None, description="Selected implementation ID")
    total_score: Optional[float] = Field(None, description="Score of selected option (lower is better)")
    estimated_exec_time_sec: Optional[float] = Field(None, description="Estimated execution time in seconds")
    estimated_energy_kwh: Optional[float] = Field(None, description="Estimated energy consumption in kWh")
    all_scores: List[HWScore] = Field(default_factory=list, description="Scores for all evaluated candidates")
    weights_used: ScoringWeights = Field(..., description="Weights that were applied")
    reason: Optional[str] = Field(None, description="Explanation of the decision")
    timestamp: float = Field(..., description="Decision timestamp")
