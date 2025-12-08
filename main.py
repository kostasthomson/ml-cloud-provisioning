"""
FastAPI application for smart task allocation.
Provides REST API endpoint for C++ simulator to request allocation decisions.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from config import fast_api_configuration
from entities import BaseAllocator, HeuristicAllocator, NNAllocator, AllocationRequest, AllocationDecision, \
    HealthCheckResponse
from models import NeuralNetwork, EnergyAwareNN
from utils import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global allocator instance
allocator: BaseAllocator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global allocator

    # Startup
    logger.info(f"Starting {fast_api_configuration.app_name} v{fast_api_configuration.app_version}")

    # Initialize allocator
    allocator = NNAllocator(NeuralNetwork.parent_directory)
    logger.info("Task allocator initialized")
    logger.info("Task allocator and allocation logger initialized")
    logger.info(f"Model type: {fast_api_configuration.model_type}")

    yield

    # Shutdown
    logger.info("Shutting down application")
    stats = allocator.get_statistics()
    logger.info(f"Final statistics: {stats}")

    allocator.save_logs()
    logger.info("Allocation decisions saved to file")


# Create FastAPI application
app = FastAPI(
    title=fast_api_configuration.app_name,
    version=fast_api_configuration.app_version,
    description="Smart task allocation service for cloud simulation using ML/DL techniques",
    lifespan=lifespan
)

# Configure CORS
if fast_api_configuration.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=fast_api_configuration.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Root endpoint - health check."""
    return HealthCheckResponse(
        status="healthy",
        version=fast_api_configuration.app_version,
        model_type=fast_api_configuration.model_type
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version=fast_api_configuration.app_version,
        model_type=fast_api_configuration.model_type
    )


@app.post("/allocate_task", response_model=AllocationDecision)
async def allocate_task(request: AllocationRequest) -> AllocationDecision:
    """
    Main allocation endpoint.

    Receives current system state and task requirements from C++ simulator,
    returns allocation decision (cell and server selection).

    Args:
        request: AllocationRequest containing cells status and task requirements

    Returns:
        AllocationDecision with selected placement or rejection

    Raises:
        HTTPException: If request processing fails
    """
    try:
        logger.info(
            f"Received allocation request for task {request.task.task_id} "
            f"at timestamp {request.timestamp}"
        )

        # Validate request
        if not request.cells:
            raise HTTPException(
                status_code=400,
                detail="Request must contain at least one cell"
            )

        # Get allocation decision
        decision = allocator.allocate_task(request)

        logger.info(
            f"Decision for task {request.task.task_id}: "
            f"{'SUCCESS' if decision.success else 'REJECTED'}"
        )

        return decision

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing allocation request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/statistics")
async def get_statistics():
    """
    Get allocation statistics.

    Returns current statistics about allocation decisions.
    """
    try:
        stats = allocator.get_statistics()
        log_summary = allocator.get_logs()

        return {
            "status": "success",
            "statistics": stats,
            "logged_decisions": log_summary
        }
    except Exception as e:
        logger.error(f"Error retrieving statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset_statistics")
async def reset_statistics():
    """Reset allocation statistics."""
    try:
        allocator.reset()
        logger.info("Statistics reset")
        return {"status": "success", "message": "Statistics reset"}
    except Exception as e:
        logger.error(f"Error resetting statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_logs")
async def save_logs():
    """Manually trigger saving of allocation logs to file."""
    try:
        saved_logs = allocator.save_logs()
        if not saved_logs:
            raise Exception("Failed to save logs to file")
        summary = allocator.get_logs()
        return {
            "status": "success",
            "model": fast_api_configuration.model_type,
            "message": "Allocation logs saved successfully",
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error saving logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=fast_api_configuration.api_host,
        port=fast_api_configuration.api_port,
        reload=True,  # Enable auto-reload during development
        log_level=fast_api_configuration.log_level.lower()
    )
