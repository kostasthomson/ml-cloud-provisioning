from postporocess import main as post_process_main
from train_agent import main as train_main


def main():
    training_data_path = post_process_main()
    train_main(training_data_path)


if __name__ == "__main__":
    main()
