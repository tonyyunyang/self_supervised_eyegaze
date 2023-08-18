import json
import sys

from utils.load_data_from_file import *


def main():
    # Load the config from JSON file first
    with open("utils/config.json", "r") as file:
        config = json.load(file)
    print(config)

    if config["test_mode"] == "Mixed":
        data, labels = load_mixed_data(window_size=config["window_size"], overlap=config["overlap"])
        eyegaze_data = (prepare_mixed_data_loader
                        (data, labels, batch_size=config["batch_size"], max_len=config["window_size"]))
    elif config["test_mode"] == "One_out":
        train_data, train_labels, test_data, test_labels = load_one_out_data(window_size=150, overlap=0.8)
        eyegaze_data = (prepare_one_out_data_loader
                        (train_data, train_labels, test_data, test_labels,
                         batch_size=config["batch_size"], max_len=config["window_size"]))
    else:
        print("Either Mixed / One_out")
        sys.exit()


if __name__ == "__main__":
    main()
