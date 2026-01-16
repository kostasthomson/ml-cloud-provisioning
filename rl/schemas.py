"""
Pydantic schemas for RL service API.

These schemas define the contract between the RL service and any external system
(simulator, real cloud environment, etc.). The service is system-agnostic.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import IntEnum


class ActionType(IntEnum):
    REJECT = -1           # Reject task (no suitable resources)
    # HW type IDs are dynamic - action = hw_type_id for allocation


class TaskState(BaseModel):
    """Incoming task characteristics."""
    task_id: str = Field(..., description="Unique task identifier")
    num_vms: int = Field(..., ge=1, description="Number of VMs requested")
    vcpus_per_vm: int = Field(..., ge=1, description="vCPUs per VM")
    memory_per_vm: float = Field(..., gt=0, description="Memory per VM (GB)")
    storage_per_vm: float = Field(default=0.1, ge=0, description="Storage per VM (TB)")
    network_per_vm: float = Field(default=0.01, ge=0, description="Network per VM")
    instructions: float = Field(..., gt=0, description="Total compute work (instructions/FLOPS)")
    compatible_hw_types: List[int] = Field(..., description="List of compatible HW type IDs [1,2,3,4]")
    requires_accelerator: bool = Field(default=False, description="Whether accelerator is required")
    accelerator_rho: float = Field(default=0.0, ge=0, le=1, description="Accelerator utilization ratio")
    deadline: Optional[float] = Field(None, description="Optional deadline in seconds")


class HWTypeState(BaseModel):
    """Current state of a hardware type."""
    hw_type_id: int = Field(..., ge=1, description="Hardware type ID (any positive integer)")
    utilization_cpu: float = Field(..., ge=0, le=1, description="CPU utilization ratio")
    utilization_memory: float = Field(..., ge=0, le=1, description="Memory utilization ratio")
    utilization_storage: float = Field(default=0.0, ge=0, le=1, description="Storage utilization")
    utilization_network: float = Field(default=0.0, ge=0, le=1, description="Network utilization")
    utilization_accelerator: float = Field(default=0.0, ge=0, le=1, description="Accelerator utilization")
    available_cpus: float = Field(..., ge=0, description="Available CPUs")
    available_memory: float = Field(..., ge=0, description="Available memory (GB)")
    available_storage: float = Field(default=0.0, ge=0, description="Available storage (TB)")
    available_network: float = Field(default=0.0, ge=0, description="Available network bandwidth")
    available_accelerators: int = Field(default=0, ge=0, description="Available accelerators")
    total_cpus: float = Field(..., gt=0, description="Total CPUs")
    total_memory: float = Field(..., gt=0, description="Total memory (GB)")
    total_storage: float = Field(default=1.0, gt=0, description="Total storage (TB)")
    total_network: float = Field(default=1.0, gt=0, description="Total network bandwidth")
    total_accelerators: int = Field(default=0, ge=0, description="Total accelerators")
    compute_capability: float = Field(..., gt=0, description="Compute capability (MIPS)")
    accelerator_compute_capability: float = Field(default=0.0, ge=0, description="Accelerator MIPS")
    power_idle: float = Field(default=163.0, ge=0, description="Idle power (W)")
    power_max: float = Field(default=220.0, ge=0, description="Max power (W)")
    acc_power_idle: float = Field(default=0.0, ge=0, description="Accelerator idle power (W)")
    acc_power_max: float = Field(default=0.0, ge=0, description="Accelerator max power (W)")
    num_running_tasks: int = Field(default=0, ge=0, description="Number of running tasks")
    avg_remaining_time: float = Field(default=0.0, ge=0, description="Avg remaining time of running tasks (s)")


class GlobalState(BaseModel):
    """Global system state."""
    timestamp: float = Field(..., ge=0, description="Current simulation/system timestamp")
    total_power_consumption: float = Field(default=0.0, ge=0, description="Current total power (W)")
    queue_length: int = Field(default=0, ge=0, description="Tasks waiting in queue")
    recent_acceptance_rate: float = Field(default=1.0, ge=0, le=1, description="Recent task acceptance rate")
    recent_avg_energy: float = Field(default=0.0, ge=0, description="Recent average energy per task (kWh)")


class RLState(BaseModel):
    """
    Complete state representation for RL agent.
    This is the observation the agent receives to make decisions.
    """
    task: TaskState = Field(..., description="Incoming task to allocate")
    hw_types: List[HWTypeState] = Field(..., min_length=1, description="Available HW type states (any number)")
    global_state: GlobalState = Field(..., description="Global system state")

    class Config:
        json_schema_extra = {
            "example": {
                "task": {
                    "task_id": "task_001",
                    "num_vms": 4,
                    "vcpus_per_vm": 8,
                    "memory_per_vm": 32.0,
                    "storage_per_vm": 0.1,
                    "network_per_vm": 0.01,
                    "instructions": 1e10,
                    "compatible_hw_types": [1, 2],
                    "requires_accelerator": False,
                    "accelerator_rho": 0.0
                },
                "hw_types": [
                    {
                        "hw_type_id": 1,
                        "utilization_cpu": 0.3,
                        "utilization_memory": 0.25,
                        "available_cpus": 1400,
                        "available_memory": 9600,
                        "total_cpus": 2000,
                        "total_memory": 12800,
                        "compute_capability": 4400,
                        "num_running_tasks": 5
                    }
                ],
                "global_state": {
                    "timestamp": 100.0,
                    "total_power_consumption": 50000,
                    "queue_length": 3
                }
            }
        }


class RLAction(BaseModel):
    """Action selected by the RL agent."""
    action: int = Field(..., description="Action: -1 for reject, or hw_type_id for allocation")
    hw_type_id: Optional[int] = Field(None, description="HW type ID if allocating, None if rejecting")
    action_name: str = Field(..., description="Human-readable action name")
    confidence: float = Field(default=1.0, ge=0, le=1, description="Action confidence/probability")
    action_probs: Optional[Dict[int, float]] = Field(None, description="Probabilities for each HW type and reject")

    @classmethod
    def from_hw_type(
        cls,
        hw_type_id: Optional[int],
        confidence: float = 1.0,
        probs: Optional[Dict[int, float]] = None
    ) -> "RLAction":
        if hw_type_id is None or hw_type_id == -1:
            return cls(
                action=-1,
                hw_type_id=None,
                action_name="REJECT",
                confidence=confidence,
                action_probs=probs
            )
        return cls(
            action=hw_type_id,
            hw_type_id=hw_type_id,
            action_name=f"ALLOCATE_HW_{hw_type_id}",
            confidence=confidence,
            action_probs=probs
        )


class TaskOutcome(BaseModel):
    """Outcome after executing an action (for reward computation)."""
    task_id: str = Field(..., description="Task identifier")
    action_taken: int = Field(..., description="Action: -1 for reject, or hw_type_id")
    hw_type_id: Optional[int] = Field(None, description="HW type used, None if rejected")
    accepted: bool = Field(..., description="Whether task was accepted")
    energy_consumed_kwh: float = Field(default=0.0, ge=0, description="Energy consumed (kWh)")
    execution_time_sec: float = Field(default=0.0, ge=0, description="Actual execution time (s)")
    deadline_met: Optional[bool] = Field(None, description="Whether deadline was met (if applicable)")
    sla_violation: bool = Field(default=False, description="Whether SLA was violated")


class RLExperience(BaseModel):
    """
    Single experience tuple for training.
    External systems can submit experiences to train the agent.
    """
    state: RLState = Field(..., description="State before action")
    action: int = Field(..., description="Action: -1 for reject, or hw_type_id")
    reward: float = Field(..., description="Reward received")
    next_state: RLState = Field(..., description="State after action")
    done: bool = Field(default=False, description="Whether episode ended")
    info: Optional[Dict[str, Any]] = Field(default=None, description="Additional info")


class RLPredictionRequest(BaseModel):
    """Request for action prediction (inference)."""
    state: RLState = Field(..., description="Current state")
    deterministic: bool = Field(default=True, description="Use deterministic policy (no exploration)")


class RLPredictionResponse(BaseModel):
    """Response with predicted action."""
    action: RLAction = Field(..., description="Selected action")
    state_value: Optional[float] = Field(None, description="Estimated state value (V(s))")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class RLTrainingConfig(BaseModel):
    """Configuration for training."""
    learning_rate: float = Field(default=3e-4, gt=0, description="Learning rate")
    gamma: float = Field(default=0.99, ge=0, le=1, description="Discount factor")
    batch_size: int = Field(default=64, gt=0, description="Training batch size")
    n_epochs: int = Field(default=10, gt=0, description="PPO epochs per update")
    clip_range: float = Field(default=0.2, gt=0, le=1, description="PPO clip range")
    total_timesteps: int = Field(default=100000, gt=0, description="Total training timesteps")
    max_steps_per_episode: int = Field(default=2048, gt=0, description="Max steps before truncating episode")
    reward_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Reward component weights: {energy, sla, rejection, efficiency}"
    )


class RLModelInfo(BaseModel):
    """Information about the current RL model."""
    model_type: str = Field(default="PPO", description="Algorithm type")
    is_trained: bool = Field(..., description="Whether model has been trained")
    training_timesteps: int = Field(default=0, ge=0, description="Total training steps")
    task_dim: int = Field(..., description="Task embedding dimension")
    hw_dim: int = Field(..., description="HW type embedding dimension")
    infrastructure_agnostic: bool = Field(default=True, description="Supports any number of HW types")
    model_path: Optional[str] = Field(None, description="Path to saved model")
    last_training_reward: Optional[float] = Field(None, description="Last training episode reward")
    training_config: Optional[RLTrainingConfig] = Field(None, description="Training configuration used")


class RLTrainingStatus(BaseModel):
    """Status of ongoing training."""
    is_training: bool = Field(..., description="Whether training is in progress")
    current_timestep: int = Field(default=0, description="Current training timestep")
    total_timesteps: int = Field(default=0, description="Target total timesteps")
    episodes_completed: int = Field(default=0, description="Episodes completed")
    avg_reward: float = Field(default=0.0, description="Average reward (recent)")
    avg_episode_length: float = Field(default=0.0, description="Average episode length")


class ExperienceBatch(BaseModel):
    """Batch of experiences for bulk training."""
    experiences: List[RLExperience] = Field(..., min_length=1, description="List of experiences")
