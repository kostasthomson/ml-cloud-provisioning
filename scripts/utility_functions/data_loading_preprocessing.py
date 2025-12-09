import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(data_path):
    """Load and preprocess decision data"""

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    # Filter only accepted tasks (we learn from successful allocations)
    df_accepted = df[df['accepted'] == 1].copy()
    print(f"Accepted tasks: {len(df_accepted)} ({100 * len(df_accepted) / len(df):.1f}%)")

    # Feature engineering
    # Task features (5)
    task_features = [
        'num_vms',
        'cpu_req',
        'mem_req',
        'timestamp',  # Time context
    ]

    # Derived feature: total CPU request
    df_accepted['total_cpu_req'] = df_accepted['num_vms'] * df_accepted['cpu_req']
    task_features.append('total_cpu_req')

    # Cell state features (6)
    state_features = [
        'util_cpu_before',
        'util_mem_before',
        'avail_cpu_before',
        'avail_mem_before',
        'avail_storage_before',
        'avail_accelerators_before'
    ]

    # Combined features
    all_features = task_features + state_features

    # Extract features and labels
    X = df_accepted[all_features].values
    y = df_accepted['chosen_hw_type'].values - 1  # Convert to 0-indexed (0-3)

    print(f"\\nFeature matrix shape: {X.shape}")
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
