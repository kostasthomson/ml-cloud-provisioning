# Energy-Aware Smart Provisioning System

## Overview

This document describes the implementation of an ML-based energy-aware resource allocation system for cloud computing environments. The system predicts energy consumption for task-hardware combinations and selects the most energy-efficient allocation.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CloudLightning Simulator (C++)                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │
│  │   mlBroker  │───▶│ httpClient  │───▶│ POST /allocate_task     │  │
│  └─────────────┘    └─────────────┘    └───────────┬─────────────┘  │
└────────────────────────────────────────────────────┼────────────────┘
                                                     │ HTTP/JSON
                                                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Service (Python)                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 EnergyRegressionAllocator                    │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │    │
│  │  │ StandardScaler│  │ EnergyAwareNN │  │ Feature Builder │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. Energy Regression vs Hardware Classification

**Problem**: Initial approach used classification to predict optimal hardware type (1-4). This resulted in 100% accuracy from epoch 1, indicating data leakage.

**Root Cause**: Features like `acc_req`, `rho_acc`, and accelerator-related power parameters directly encoded the hardware type, making classification trivial.

**Solution**: Switched to energy regression model that predicts energy consumption (in Wh) for any given (task, hardware) combination.

**Rationale**:
- The model learns the relationship between task requirements, hardware characteristics, and energy consumption
- At inference time, we evaluate all valid hardware options and select the minimum energy
- This approach is hardware-type agnostic - works with any number of hardware configurations
- Avoids data leakage by treating hardware properties as input features, not encoded targets

### 2. Scoring Model Approach

Instead of directly predicting "which hardware to use", the model answers "how much energy would this task consume on this hardware?"

```
For each incoming task:
    For each available hardware type:
        predicted_energy = model.predict(task_features, hw_features)
    Select hardware with minimum predicted_energy
```

**Benefits**:
- Supports dynamic hardware configurations
- Enables multi-implementation optimization (same application, different implementations)
- Provides interpretable energy estimates for logging and analysis

### 3. Feature Engineering

**Task Features** (from allocation request):
| Feature | Description |
|---------|-------------|
| `num_vms` | Number of VMs requested |
| `cpu_req` | vCPUs per VM |
| `mem_req` | Memory per VM (GB) |
| `storage_req` | Storage per VM (TB) - optional |
| `network_req` | Network bandwidth per VM - optional |
| `acc_req` | Accelerator requirement (0/1) - optional |
| `rho_acc` | Accelerator utilization ratio - optional |

**Hardware State Features** (from system state):
| Feature | Description |
|---------|-------------|
| `util_cpu_before` | CPU utilization before allocation |
| `util_mem_before` | Memory utilization before allocation |
| `avail_cpu_before` | Available CPU cores |
| `avail_mem_before` | Available memory |
| `avail_storage_before` | Available storage |
| `avail_accelerators_before` | Available accelerators |
| `total_cpu` | Total CPU capacity |
| `total_mem` | Total memory capacity |
| `total_accelerators` | Total accelerator count |
| `cpu_idle_power` | CPU idle power consumption (W) |
| `cpu_max_power` | CPU max power consumption (W) |
| `acc_idle_power` | Accelerator idle power (W) |
| `acc_max_power` | Accelerator max power (W) |
| `compute_cap_per_cpu` | CPU compute capability |
| `compute_cap_acc` | Accelerator compute capability |

**Derived Features** (computed at runtime):
| Feature | Computation |
|---------|-------------|
| `total_cpu_req` | `num_vms × cpu_req` |
| `total_mem_req` | `num_vms × mem_req` |
| `power_range` | `cpu_max_power - cpu_idle_power` |
| `cpu_util_ratio` | `avail_cpu / total_cpu` |

**Dynamic Feature Detection**: The allocator automatically detects which features are available in the training data and adapts accordingly. This ensures compatibility across different training datasets.

### 4. Neural Network Architecture

```python
EnergyAwareNN(
    Input(N features)
    → Linear(N → 128) + BatchNorm + LeakyReLU(0.1) + Dropout(0.1)
    → Linear(128 → 128) + BatchNorm + LeakyReLU(0.1) + Dropout(0.1)
    → Linear(128 → 64) + BatchNorm + LeakyReLU(0.1) + Dropout(0.1)
    → Linear(64 → 32) + LeakyReLU(0.1)
    → Linear(32 → 1)  # Energy output
)
```

**Architecture Decisions**:

| Decision | Rationale |
|----------|-----------|
| LeakyReLU(0.1) instead of ReLU | Prevents dead neurons when predicting small energy values |
| Low Dropout (0.1) | Energy prediction is deterministic; high dropout caused instability |
| BatchNorm | Stabilizes training across varying feature scales |
| No output activation | Energy is unbounded positive; ReLU would work but unnecessary |

### 5. Target Scaling

**Problem**: Raw energy values in kWh were very small (~0.001-0.01), causing vanishing gradients and model collapse (all predictions → 0).

**Solution**: Scale energy target from kWh to Wh (multiply by 1000).

```python
y = df['energy_kwh'].values * 1000.0  # Convert to Wh
```

**Result**: Model now predicts in Wh range (0.1-10), providing better gradient signal.

### 6. Multi-Implementation Support

Applications may have multiple implementations with different resource requirements:

```json
{
  "implementations": [
    {"name": "CPU-only", "num_vms": 16, "cpu_req": 8, "requires_accelerator": false},
    {"name": "GPU-accelerated", "num_vms": 4, "cpu_req": 4, "requires_accelerator": true}
  ]
}
```

The allocator evaluates all (implementation × hardware) combinations:

```
Implementation    Hardware     Predicted Energy
─────────────────────────────────────────────────
CPU-only         CPU-only     0.45 Wh  ← SELECTED
CPU-only         CPU+GPU      0.87 Wh
GPU-accelerated  CPU+GPU      0.78 Wh
GPU-accelerated  CPU-only     SKIPPED (no accelerator)
```

### 7. Fallback Mechanism

The C++ simulator includes fallback logic:

1. Try Windows host IP (from WSL's `/etc/resolv.conf`)
2. If connection fails, fallback to localhost
3. If ML service unavailable, use traditional heuristic broker

```cpp
if (res != CURLE_OK && hostIP != "127.0.0.1") {
    fallbackToLocalhost();
    // Retry with localhost
}
if (response.empty()) {
    fallbackBroker->deploy(...);  // Use heuristic
}
```

## Training Pipeline

### Data Generation

1. C++ simulator runs with `improvedSosmBroker`
2. Each allocation decision is logged to CSV with full state
3. Multiple simulation runs with varying configurations generate diverse training data

### Data Preprocessing

```python
# Filter valid records
df = df[df['accepted'] == 1]      # Only accepted tasks
df = df[df['chosen_hw_type'] > 0] # Valid hardware selection
df = df[df['energy_kwh'] > 0]     # Positive energy consumption

# Scale target
y = df['energy_kwh'].values * 1000.0  # kWh → Wh

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss Function | MSE |
| Epochs | 100 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Train/Val/Test Split | 70% / 15% / 15% |

## API Endpoints

### POST /allocate_task

Single-implementation task allocation.

**Request**:
```json
{
  "timestamp": 100.0,
  "cells": [{
    "cell_id": 1,
    "hw_types": [...],
    "available_resources": {...},
    "current_utilization": {...}
  }],
  "task": {
    "task_id": "task_001",
    "num_vms": 4,
    "vcpus_per_vm": 8,
    "memory_per_vm": 32.0,
    "requires_accelerator": false
  }
}
```

**Response**:
```json
{
  "success": true,
  "num_vms_allocated": 4,
  "vm_allocations": [...],
  "estimated_energy_cost": 0.00045,
  "allocation_method": "energy_regression"
}
```

### POST /allocate_multi_impl

Multi-implementation optimization.

**Request**: Includes array of implementation options.

**Response**: Includes selected implementation, all predictions, and skipped combinations.

## Results

### Training Metrics (Example Run)

| Metric | Value |
|--------|-------|
| Training Samples | 565,844 |
| Input Features | 36 |
| Test RMSE | ~0.15 Wh |
| Test MAE | ~0.10 Wh |
| Test R² | ~0.85 |

### API Test Results

```
============================================================
SMART ALLOCATOR API INTEGRATION TEST
============================================================
Testing health endpoint...
  Status: 200
  Response: {'status': 'healthy', 'model_type': 'energy_regression'}

Testing /allocate_task endpoint...
  Status: 200
  Success: True
  Method: energy_regression
  VMs allocated: 4

Testing /allocate_multi_impl endpoint...
  Status: 200
  Success: True
  Selected impl: CPU-only (ID: 1)
  Energy: 0.45 Wh

  All predictions (3):
    CPU-only on Cell1_HW1: 0.45 Wh
    GPU-accelerated on Cell1_HW2: 0.78 Wh
    CPU-only on Cell1_HW2: 0.87 Wh

============================================================
Overall: ALL TESTS PASSED
```

## File Structure

```
ml-cloud-provisioning/
├── main.py                          # FastAPI entry point
├── config/
│   ├── fast_api_config.py           # API configuration
│   └── model_configuration.py       # Training hyperparameters
├── entities/
│   ├── schemas.py                   # Pydantic models
│   └── allocator/
│       ├── base_allocator.py        # Abstract base class
│       ├── energy_regression_allocator.py  # ML allocator
│       └── heuristic_allocator.py   # Fallback allocator
├── models/
│   └── energy_aware/nn/
│       ├── energy_aware_nn.py       # Neural network definition
│       └── model.pth                # Trained model weights
├── scripts/
│   ├── train_agent.py               # Training script
│   ├── smart_allocator.py           # Standalone agent demo
│   └── utility_functions/
│       ├── data_loading_preprocessing.py
│       └── training_functions.py
├── training/
│   └── training_data.csv            # Merged training data
└── tests/
    └── test_api_integration.py      # API tests
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOCATOR_ALLOCATOR_TYPE` | `energy_regression` | Allocator type: `heuristic`, `nn`, `energy_regression` |
| `ALLOCATOR_API_HOST` | `0.0.0.0` | API bind host |
| `ALLOCATOR_API_PORT` | `8000` | API port |
| `ALLOCATOR_LOG_LEVEL` | `INFO` | Logging level |

## Future Improvements

1. **Online Learning**: Update model with new allocation decisions during runtime
2. **Uncertainty Estimation**: Add confidence intervals to energy predictions
3. **Multi-Objective Optimization**: Balance energy, latency, and cost
4. **Federated Training**: Train across multiple data centers without centralizing data
5. **Hardware-Specific Models**: Ensemble of specialized models per hardware type
