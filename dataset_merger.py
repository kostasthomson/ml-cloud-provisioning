import os

import pandas as pd
import json
import numpy as np


def load_system_state(json_file):
    """
    Parses CLSim JSON output into: { timestamp: feature_dict }
    Each timestamp maps to a dict with descriptive column names.
    """
    print(f"  Reading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)

    state_lookup = {}
    temp_by_time = {}  # {timestamp: {(cell_id, hw_type_id): features_dict}}

    # 1. Parse and group by timestamp and (cell, hw_type)
    for cell_data in data['CLSim outputs']:
        cell_id = cell_data['Cell']
        hw_type_id = cell_data['HW Type']

        for record in cell_data['Outputs']:
            t = record['Time Step']

            if t not in temp_by_time:
                temp_by_time[t] = {}

            # Feature extraction with DESCRIPTIVE NAMES
            total_cpu = record.get('Total Processors', 1)
            total_mem = record.get('Total Memory', 1)

            util_cpu = record['Utilized Processors'] / (total_cpu + 1e-9)
            util_mem = record['Utilized Memory'] / (total_mem + 1e-9)

            # Use descriptive keys
            features = {
                'util_cpu': util_cpu,
                'util_mem': util_mem,
                'avail_cpu': record['Available Processors'],
                'avail_mem': record['Available Memory'],
                'avail_storage': record['Available Storage'],
                'avail_accelerators': record['Available Accelerators']
            }

            temp_by_time[t][(cell_id, hw_type_id)] = features

    # 2. Flatten into DataFrame with descriptive column names
    for t, hw_dict in temp_by_time.items():
        sorted_keys = sorted(hw_dict.keys())  # (cell, hw_type) sorted

        # Build row with descriptive column names
        row_dict = {}
        for (cell_id, hw_type_id) in sorted_keys:
            features = hw_dict[(cell_id, hw_type_id)]

            # Create columns like: cell1_hw2_util_cpu, cell1_hw2_avail_mem, etc.
            for feat_name, feat_val in features.items():
                col_name = f"cell{cell_id}_hw{hw_type_id}_{feat_name}"
                row_dict[col_name] = feat_val

        state_lookup[t] = row_dict

    return state_lookup


def process_run_pair(csv_file, json_file, broker_name):
    """
    Merges Decisions (CSV) with System State (JSON).
    """
    print(f"Processing {broker_name}...")

    # 1. Load decision log
    df = pd.read_csv(csv_file)

    # 2. Load system state context
    state_lookup = load_system_state(json_file)

    # 3. Join on timestamp
    def get_state_dict(ts):
        # Handle float precision
        ts_rounded = round(ts, 2)
        if ts_rounded in state_lookup:
            return state_lookup[ts_rounded]
        elif ts in state_lookup:
            return state_lookup[ts]
        else:
            # Return empty dict if not found
            return {}

    # Convert list of dicts to DataFrame
    state_dicts = df['timestamp'].apply(get_state_dict).tolist()
    state_df = pd.DataFrame(state_dicts)

    # Fill missing values with 0 (in case of timestamp mismatch)
    state_df = state_df.fillna(0)

    # 4. Combine
    merged_df = pd.concat([df.reset_index(drop=True), state_df], axis=1)

    # 5. DROP TIMESTAMP
    merged_df = merged_df.drop(columns=['timestamp'])

    # Sanity check
    if state_df.shape[1] == 0:
        print(f"  ⚠️  WARNING: No system state features extracted for {broker_name}!")
    else:
        print(f"  ✓ Extracted {state_df.shape[1]} state features")

    return merged_df


# ========== MAIN ==========
training_dir = 'training'
runs = [(f'{d}/decisions.csv', f'{d}/outputCLSim.json', d.split('/')[1]) for d in
        list(map(lambda x: f'{training_dir}/{x}', ['traditional', 'sosm', 'improved sosm']))]

dfs = []
for csv, json_file, name in runs:
    try:
        d = process_run_pair(csv, json_file, name)
        dfs.append(d)
        print(f"  ✓ {name}: {d.shape[0]} rows")
    except FileNotFoundError as e:
        print(f"  ✗ Skipping {name}: {e}")
    except Exception as e:
        print(f"  ✗ Error processing {name}: {e}")

if dfs:
    final_dataset = pd.concat(dfs, ignore_index=True)

    print("-" * 50)
    print(f"Total Dataset Size: {final_dataset.shape}")

    # Show sample of column names
    state_cols = [c for c in final_dataset.columns if c.startswith('cell')]
    print(f"\nExample state columns (first 10):")
    for col in state_cols[:10]:
        print(f"  - {col}")

    final_dataset.to_csv("training/training_data_v1.csv", index=False)
    print("\n✓ Saved to training_data_v1.csv")
else:
    print("No data processed.")
