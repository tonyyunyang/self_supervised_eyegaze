import json
import sys

from modules.limu_model import limu_model4pretrain
from modules.pretrain_hyperparameters import LIMU_Pretrain_Hyperparameters
from utils.load_data_from_file import load_mixed_data, prepare_mixed_data_loader, load_one_out_data, \
    prepare_one_out_data_loader, limu_prepare_mixed_data_loader, limu_prepare_one_out_data_loader
from utils.pretrain import pretrain_limu_model


def main():
    # Load the config from JSON file first
    with open("utils/config.json", "r") as file:
        config = json.load(file)
    print(config)

    if config["general"]["test_mode"] == "Mixed":
        data, labels, encoder = load_mixed_data(window_size=config["general"]["window_size"],
                                                overlap=config["general"]["overlap"])

        num_classes = len(encoder.classes_)
        feat_dim = data[0].shape[1]
        labels_dim = labels.shape
        print(f"The shape of data is {data.shape}, the feat_dim is {feat_dim}, the labels_dim is {labels_dim}")

        eyegaze_data_loader = (limu_prepare_mixed_data_loader
                               (config, data, labels, batch_size=config["general"]["batch_size"],
                                max_len=config["general"]["window_size"]))

    elif config["general"]["test_mode"] == "One_out":
        train_data, train_labels, test_data, test_labels, encoder = (load_one_out_data
                                                                     (window_size=config["general"]["window_size"],
                                                                      overlap=config["general"]["overlap"]))

        num_classes = len(encoder.classes_)
        feat_dim = train_data[0].shape[1]
        print(f"The number of classes is {num_classes}, the feat_dim is {feat_dim}")

        eyegaze_data_loader = (limu_prepare_one_out_data_loader
                               (config, train_data, train_labels, test_data, test_labels,
                                batch_size=config["general"]["batch_size"],
                                max_len=config["general"]["window_size"]))
    else:
        print("Either Mixed / One_out")
        sys.exit()

    hyperparameters = LIMU_Pretrain_Hyperparameters(config)
    model = limu_model4pretrain(config, feat_dim)
    loss = hyperparameters.loss
    optimizer = hyperparameters.optimizer(model.parameters(), hyperparameters.lr, weight_decay=hyperparameters.weight_decay)

    pretrain_limu_model(model, loss, optimizer, eyegaze_data_loader[0], config)

    # for i in eyegaze_data_loader[0]:
    #     print(i)


if __name__ == "__main__":
    main()
