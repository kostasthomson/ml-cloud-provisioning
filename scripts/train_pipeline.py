import shutil
from pathlib import Path
from dataset_merger import CellDatasetMerger
from train_agent import main as train_main

SIMULATOR_OUTPUT = Path(__file__).parent.parent.parent / "cloudlightning-simulator" / "output" / "improved"
TRAINING_RUNS_DIR = Path(__file__).parent.parent / "training" / "simulation_runs"


def sync_from_simulator(config_name: str = "latest"):
    """Copy CSV logs from simulator output to training directory."""
    if not SIMULATOR_OUTPUT.exists():
        print(f"Simulator output not found: {SIMULATOR_OUTPUT}")
        return False

    csv_files = list(SIMULATOR_OUTPUT.glob("cell_*_decisions.csv"))
    if not csv_files:
        print("No CSV files found in simulator output")
        return False

    target_dir = TRAINING_RUNS_DIR / config_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying {len(csv_files)} CSV files to {target_dir}")
    for src in csv_files:
        shutil.copy2(src, target_dir / src.name)

    return True


def main(generate_data: bool = False, sync: bool = False, config_name: str = "latest"):
    if generate_data:
        from generate_training_data import generate_all_configs
        generate_all_configs()

    if sync:
        sync_from_simulator(config_name)

    merger = CellDatasetMerger("../training/simulation_runs")
    df = merger.process_all_configs("../training/training_data.csv")

    if df is not None:
        train_main("../training/training_data.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate new training data')
    parser.add_argument('--sync', action='store_true', help='Sync CSV logs from simulator output')
    parser.add_argument('--config-name', type=str, default='latest', help='Name for synced config (default: latest)')
    args = parser.parse_args()
    main(generate_data=args.generate, sync=args.sync, config_name=args.config_name)
