"""
REST API router for RL service.

Provides system-agnostic endpoints for:
- Inference (predict action given state)
- Training (from experiences or online)
- Model management (save, load, info)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import logging
import threading

from .schemas import (
    RLPredictionRequest,
    RLPredictionResponse,
    RLTrainingConfig,
    RLTrainingStatus,
    RLModelInfo,
    RLExperience,
    ExperienceBatch,
    TaskOutcome,
)
from .agent import RLAgent
from .trainer import PPOTrainer
from .environment import CloudProvisioningEnv
from .reward import RewardCalculator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rl", tags=["Reinforcement Learning"])

_agent: Optional[RLAgent] = None
_trainer: Optional[PPOTrainer] = None
_training_lock = threading.Lock()


def get_agent() -> RLAgent:
    """Get or create the global RL agent."""
    global _agent
    if _agent is None:
        _agent = RLAgent()
        logger.info("RL Agent initialized")
    return _agent


def get_trainer() -> PPOTrainer:
    """Get or create the global PPO trainer."""
    global _trainer
    if _trainer is None:
        _trainer = PPOTrainer(get_agent())
        logger.info("PPO Trainer initialized")
    return _trainer


@router.get("/health")
async def rl_health():
    """RL service health check."""
    agent = get_agent()
    return {
        "status": "healthy",
        "agent_initialized": agent is not None,
        "is_trained": agent.is_trained,
        "training_timesteps": agent.training_timesteps,
        "infrastructure_agnostic": True,
        "task_dim": agent.task_dim,
        "hw_dim": agent.hw_dim
    }


@router.post("/predict", response_model=RLPredictionResponse)
async def predict_action(request: RLPredictionRequest) -> RLPredictionResponse:
    """
    Predict action for given state.

    This is the main inference endpoint used by external systems.
    The model is infrastructure-agnostic and works with any number of HW types.

    Args:
        request: RLPredictionRequest with state and options

    Returns:
        RLPredictionResponse with action (hw_type_id or -1 for reject) and metadata
    """
    try:
        agent = get_agent()

        action, state_value, inference_time_ms = agent.predict(
            state=request.state,
            deterministic=request.deterministic
        )

        return RLPredictionResponse(
            action=action,
            state_value=state_value,
            inference_time_ms=inference_time_ms
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/model", response_model=RLModelInfo)
async def get_model_info() -> RLModelInfo:
    """Get information about the current RL model."""
    agent = get_agent()
    return agent.get_model_info()


@router.post("/model/save")
async def save_model(path: str = "models/rl/ppo/model.pth"):
    """Save current model to file."""
    try:
        agent = get_agent()
        agent.save(path)
        return {"status": "success", "path": path}
    except Exception as e:
        logger.error(f"Save error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/load")
async def load_model(path: str = "models/rl/ppo/model.pth"):
    """Load model from file."""
    try:
        agent = get_agent()
        agent.load(path)
        return {"status": "success", "path": path, "info": agent.get_model_info().model_dump()}
    except Exception as e:
        logger.error(f"Load error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experience")
async def submit_experience(experience: RLExperience):
    """
    Submit a single experience tuple for training.

    External systems submit (state, action, reward, next_state, done) tuples
    which are used for training updates.
    """
    try:
        trainer = get_trainer()

        if not hasattr(trainer, 'experience_buffer'):
            trainer.experience_buffer = []

        trainer.experience_buffer.append(experience)

        return {
            "status": "success",
            "buffer_size": len(trainer.experience_buffer)
        }

    except Exception as e:
        logger.error(f"Experience submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experience/batch")
async def submit_experience_batch(batch: ExperienceBatch):
    """Submit batch of experiences."""
    try:
        trainer = get_trainer()

        if not hasattr(trainer, 'experience_buffer'):
            trainer.experience_buffer = []

        trainer.experience_buffer.extend(batch.experiences)

        return {
            "status": "success",
            "added": len(batch.experiences),
            "buffer_size": len(trainer.experience_buffer)
        }

    except Exception as e:
        logger.error(f"Batch submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status", response_model=RLTrainingStatus)
async def get_training_status() -> RLTrainingStatus:
    """Get current training status."""
    trainer = get_trainer()
    return trainer.get_status()


@router.post("/training/start")
async def start_training(
    background_tasks: BackgroundTasks,
    config: Optional[RLTrainingConfig] = None
):
    """
    Start training from buffered experiences.

    Training runs in background and status can be queried via /training/status.
    """
    trainer = get_trainer()

    if trainer.is_training:
        raise HTTPException(status_code=400, detail="Training already in progress")

    if not hasattr(trainer, 'experience_buffer') or len(trainer.experience_buffer) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 100 experiences. Current: {getattr(trainer, 'experience_buffer', [])}"
        )

    if config:
        trainer.config = config

    def train_background():
        with _training_lock:
            try:
                experiences = trainer.experience_buffer
                trainer.experience_buffer = []
                trainer.train_from_experiences(experiences)
                logger.info("Background training completed")
            except Exception as e:
                logger.error(f"Training error: {str(e)}", exc_info=True)

    background_tasks.add_task(train_background)

    return {
        "status": "started",
        "experiences": len(trainer.experience_buffer),
        "config": (config or trainer.config).model_dump()
    }


@router.post("/training/simulate")
async def train_simulation(
    background_tasks: BackgroundTasks,
    total_timesteps: int = 10000,
    config: Optional[RLTrainingConfig] = None
):
    """
    Train using simulated environment (for testing/demonstration).

    Runs PPO training on a simulated environment with random states.
    """
    trainer = get_trainer()

    if trainer.is_training:
        raise HTTPException(status_code=400, detail="Training already in progress")

    if config:
        trainer.config = config

    def train_background():
        with _training_lock:
            try:
                env = CloudProvisioningEnv()
                trainer.train(env, total_timesteps=total_timesteps)
                logger.info(f"Simulation training completed: {total_timesteps} timesteps")
            except Exception as e:
                logger.error(f"Simulation training error: {str(e)}", exc_info=True)

    background_tasks.add_task(train_background)

    return {
        "status": "started",
        "total_timesteps": total_timesteps,
        "config": (config or trainer.config).model_dump()
    }


@router.post("/reward/compute")
async def compute_reward(outcome: TaskOutcome):
    """
    Compute reward for a task outcome.

    Utility endpoint for external systems to compute rewards
    using the same logic as the RL agent.
    """
    try:
        agent = get_agent()
        reward = agent.reward_calculator.compute_reward(outcome)
        return {
            "reward": reward,
            "config": agent.reward_calculator.get_config()
        }
    except Exception as e:
        logger.error(f"Reward computation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reward/config")
async def get_reward_config():
    """Get current reward configuration."""
    agent = get_agent()
    return agent.reward_calculator.get_config()


@router.put("/reward/config")
async def set_reward_config(
    energy_weight: float = 0.5,
    sla_weight: float = 0.3,
    rejection_penalty: float = 0.2,
    energy_baseline: float = 1.0
):
    """Update reward configuration."""
    try:
        agent = get_agent()
        agent.reward_calculator = RewardCalculator(
            energy_weight=energy_weight,
            sla_weight=sla_weight,
            rejection_penalty=rejection_penalty,
            energy_baseline=energy_baseline
        )
        return agent.reward_calculator.get_config()
    except Exception as e:
        logger.error(f"Reward config error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
