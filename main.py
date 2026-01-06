"""
FastAPI application for smart task allocation.
Provides REST API endpoint for C++ simulator to request allocation decisions.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from config import fast_api_configuration
from entities import BaseAllocator, HeuristicAllocator, NNAllocator, EnergyRegressionAllocator, \
    AllocationRequest, AllocationDecision, HealthCheckResponse, \
    MultiImplAllocationRequest, MultiImplAllocationDecision, EnergyPrediction, VMAllocation, \
    ScoringAllocator, ScoringAllocationRequest, ScoringAllocationResponse, ScoringWeights, \
    RLAllocator
from models import NeuralNetwork, EnergyAwareNN
from utils import setup_logging
from rl.api import router as rl_router

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global allocator instances
allocator: BaseAllocator = None
scoring_allocator: ScoringAllocator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global allocator, scoring_allocator

    # Startup
    logger.info(f"Starting {fast_api_configuration.app_name} v{fast_api_configuration.app_version}")

    # Initialize allocator based on configuration
    allocator_type = fast_api_configuration.allocator_type
    logger.info(f"Initializing allocator: {allocator_type}")

    if allocator_type == "heuristic":
        allocator = HeuristicAllocator()
    elif allocator_type == "nn":
        allocator = NNAllocator(NeuralNetwork.parent_directory)
    elif allocator_type == "energy_regression":
        allocator = EnergyRegressionAllocator()
    elif allocator_type == "rl":
        allocator = RLAllocator()
    else:
        logger.warning(f"Unknown allocator type: {allocator_type}, defaulting to rl")
        allocator = RLAllocator()

    # Initialize scoring allocator (always available)
    scoring_allocator = ScoringAllocator()
    logger.info("Scoring allocator initialized")

    logger.info("Task allocator initialized")
    logger.info("Task allocator and allocation logger initialized")
    logger.info(f"Model type: {fast_api_configuration.model_type}")

    yield

    # Shutdown
    logger.info("Shutting down application")
    stats = allocator.get_statistics()
    logger.info(f"Final statistics: {stats}")

    scoring_stats = scoring_allocator.get_statistics()
    logger.info(f"Scoring allocator statistics: {scoring_stats}")

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

# Include RL router
app.include_router(rl_router)


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


@app.post("/allocate_multi_impl", response_model=MultiImplAllocationDecision)
async def allocate_multi_impl(request: MultiImplAllocationRequest) -> MultiImplAllocationDecision:
    """
    Multi-implementation allocation endpoint.

    For applications with multiple implementations (CPU-only, GPU-accelerated, etc.),
    evaluates all (implementation, hardware) combinations and selects the one with
    minimum predicted energy consumption.

    Args:
        request: MultiImplAllocationRequest with cells, implementations

    Returns:
        MultiImplAllocationDecision with optimal selection and all predictions
    """
    try:
        logger.info(
            f"Multi-impl allocation request for app {request.application_id}, "
            f"task {request.task_id}, {len(request.implementations)} implementations"
        )

        if not request.cells:
            raise HTTPException(status_code=400, detail="Request must contain at least one cell")

        if not request.implementations:
            raise HTTPException(status_code=400, detail="Request must contain at least one implementation")

        if not isinstance(allocator, EnergyRegressionAllocator):
            raise HTTPException(
                status_code=400,
                detail="Multi-implementation allocation requires energy_regression allocator"
            )

        result = allocator.allocate_multi_impl(request.cells, request.implementations)

        if not result['success']:
            return MultiImplAllocationDecision(
                success=False,
                reason="No valid allocation found for any implementation",
                skipped_combinations=result['skipped'],
                allocation_method=allocator.get_method_name(),
                timestamp=request.timestamp
            )

        best = result['best']
        best_impl = best['impl']
        best_cell = best['cell']
        best_hw = best['hw']

        class TaskProxy:
            def __init__(self, impl):
                self.num_vms = impl.num_vms
                self.vcpus_per_vm = impl.vcpus_per_vm
                self.memory_per_vm = impl.memory_per_vm
                self.storage_per_vm = impl.storage_per_vm
                self.network_per_vm = impl.network_per_vm
                self.requires_accelerator = impl.requires_accelerator
                self.task_id = request.task_id

        vm_allocations = allocator._allocate_vms_to_servers(
            TaskProxy(best_impl), best_cell, best_hw
        )

        all_predictions = [
            EnergyPrediction(
                impl_id=p['impl'].impl_id,
                impl_name=p['impl'].impl_name,
                cell_id=p['cell'].cell_id,
                hw_type_id=p['hw'].hw_type_id,
                hw_name=p['hw'].hw_type_name,
                predicted_energy_wh=p['energy']
            )
            for p in result['all_predictions']
        ]

        logger.info(
            f"Multi-impl decision: {best_impl.impl_name} on "
            f"Cell{best_cell.cell_id}_HW{best_hw.hw_type_id}, "
            f"Energy={best['energy']:.2f} Wh"
        )

        return MultiImplAllocationDecision(
            success=True,
            selected_impl_id=best_impl.impl_id,
            selected_impl_name=best_impl.impl_name,
            num_vms_allocated=len(vm_allocations),
            vm_allocations=vm_allocations,
            estimated_energy_wh=best['energy'],
            all_predictions=all_predictions,
            skipped_combinations=result['skipped'],
            reason=f"Energy-optimal: {best_impl.impl_name} on {best_hw.hw_type_name}",
            allocation_method=allocator.get_method_name(),
            timestamp=request.timestamp
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing multi-impl request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/weights", response_model=ScoringWeights)
async def get_weights():
    """
    Get current global scoring weights.

    Returns the weights used for multi-objective scoring when no
    per-request weights are provided.
    """
    return scoring_allocator.global_weights


@app.put("/weights", response_model=ScoringWeights)
async def set_weights(weights: ScoringWeights):
    """
    Set global scoring weights.

    Updates the default weights used for multi-objective scoring.
    Weights must sum to 1.0.

    Args:
        weights: New scoring weights configuration

    Returns:
        Updated weights configuration
    """
    try:
        scoring_allocator.set_global_weights(weights)
        logger.info(f"Global weights updated: {weights.model_dump()}")
        return scoring_allocator.global_weights
    except Exception as e:
        logger.error(f"Error setting weights: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/allocate_scoring", response_model=ScoringAllocationResponse)
async def allocate_scoring(request: ScoringAllocationRequest) -> ScoringAllocationResponse:
    """
    Multi-objective scoring-based allocation endpoint.

    Evaluates all (implementation, hardware) combinations using weighted scoring
    across 5 metrics: energy, execution time, network utilization, RAM utilization,
    and storage utilization. Returns the combination with the lowest score.

    Features:
    - Time-aware execution estimation with partial allocation model
    - Considers ongoing tasks and when they will free resources
    - Per-request weight override or global weights
    - Full breakdown of all metrics for each candidate

    Args:
        request: ScoringAllocationRequest with task implementations, HW state, and optional weights

    Returns:
        ScoringAllocationResponse with selected HW/impl and full scoring breakdown

    Raises:
        HTTPException: If request processing fails
    """
    try:
        logger.info(
            f"Scoring allocation request for task {request.task_id} "
            f"at timestamp {request.timestamp}, "
            f"{len(request.implementations)} implementations, "
            f"{len(request.hw_types)} HW types"
        )

        if not request.hw_types:
            raise HTTPException(
                status_code=400,
                detail="Request must contain at least one hardware type"
            )

        if not request.implementations:
            raise HTTPException(
                status_code=400,
                detail="Request must contain at least one implementation"
            )

        decision = scoring_allocator.allocate(request)

        logger.info(
            f"Scoring decision for task {request.task_id}: "
            f"{'SUCCESS' if decision.success else 'REJECTED'}"
            f"{f', HW={decision.selected_hw_type_id}, impl={decision.selected_impl_id}' if decision.success else ''}"
        )

        return decision

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing scoring allocation request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/scoring_statistics")
async def get_scoring_statistics():
    """
    Get scoring allocator statistics.

    Returns statistics about scoring-based allocation decisions.
    """
    try:
        stats = scoring_allocator.get_statistics()
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving scoring statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=fast_api_configuration.api_host,
        port=fast_api_configuration.api_port,
        reload=True,
        log_level=fast_api_configuration.log_level.lower()
    )
