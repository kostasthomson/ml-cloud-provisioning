import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(data_path):
    """Load and preprocess decision data"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    # Filter only accepted tasks (learn from successful allocations)
    df_accepted = df[df['accepted'] == 1].copy()

    # Reset DataFrame index after filtering
    df_accepted = df_accepted.reset_index(drop=True)

    print(f"Accepted tasks: {len(df_accepted)} ({100 * len(df_accepted) / len(df):.1f}%)")

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    # EXCLUDED from features (data leakage):
    #   - task_id: unique identifier, no predictive value
    #   - timestamp: temporal ordering, may leak system evolution patterns
    #   - source_broker: tells which broker made decision (obvious leakage)
    #   - chosen_hw_type: this is our LABEL (what we're predicting)
    #   - energy_kwh: computed AFTER allocation decision
    #   - processing_time_sec: observed AFTER allocation decision
    # =========================================================================

    # Task requirement features (3 base + 2 derived)
    task_features = [
        'num_vms',  # Number of VMs requested
        'cpu_req',  # CPU per VM
        'mem_req',  # Memory per VM
    ]

    # Derived features: total resource requirements
    df_accepted['total_cpu_req'] = df_accepted['num_vms'] * df_accepted['cpu_req']
    df_accepted['total_mem_req'] = df_accepted['num_vms'] * df_accepted['mem_req']

    task_features.extend(['total_cpu_req', 'total_mem_req'])

    # Cell state features (6) - system state BEFORE allocation
    state_features = [
        'util_cpu_before',  # CPU utilization (0-1)
        'util_mem_before',  # Memory utilization (0-1)
        'avail_cpu_before',  # Available CPU cores
        'avail_mem_before',  # Available memory (GB)
        'avail_storage_before',  # Available storage (GB)
        'avail_accelerators_before'  # Available accelerators (e.g., GPUs)
    ]

    # Derived state features: resource availability ratios
    # (helps model understand relative scarcity)
    df_accepted['cpu_to_mem_ratio'] = (
            df_accepted['avail_cpu_before'] / (df_accepted['avail_mem_before'] + 1e-6)
    )
    df_accepted['util_avg'] = (
                                      df_accepted['util_cpu_before'] + df_accepted['util_mem_before']
                              ) / 2.0

    state_features.extend(['cpu_to_mem_ratio', 'util_avg'])

    # Combined features (5 task + 8 state = 13 features)
    all_features = task_features + state_features

    print(f"\n{'=' * 60}")
    print(f"FEATURE SELECTION (NO LEAKAGE)")
    print(f"{'=' * 60}")
    print(f"Task features ({len(task_features)}): {task_features}")
    print(f"State features ({len(state_features)}): {state_features}")
    print(f"Total features: {len(all_features)}")
    print(f"{'=' * 60}\n")

    # Extract features and labels
    X = df_accepted[all_features].values
    y = df_accepted['chosen_hw_type'].values - 1  # Convert to 0-indexed (0-3)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution:")
    for hw_type in range(4):
        count = np.sum(y == hw_type)
        print(f"  HW Type {hw_type + 1}: {count} ({100 * count / len(y):.1f}%)")

    return X, y, df_accepted


def normalize_features(X_train, X_val, X_test):
    """Normalize features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
