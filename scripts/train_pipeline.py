from dataset_merger import CellDatasetMerger
from train_agent import main as train_main


def main(generate_data: bool = False):
    if generate_data:
        from generate_training_data import generate_all_configs
        generate_all_configs()

    merger = CellDatasetMerger("../training/simulation_runs")
    df = merger.process_all_configs("../training/training_data.csv")

    if df is not None:
        train_main("../training/training_data.csv")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate new training data')
    args = parser.parse_args()
    main(generate_data=args.generate)
