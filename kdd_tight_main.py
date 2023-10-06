import json
import sys

from modules.finetune_hyperparameters import KDD_Finetune_Hyperparameters
from modules.kdd_model import kdd_model4pretrain, kdd_model4finetune
from modules.pretrain_hyperparameters import KDD_Pretrain_Hyperparameters
from utils.finetune import finetune_kdd_model, eval_finetune_kdd_model
from utils.load_data_from_file import load_mixed_data, prepare_mixed_data_loader, load_one_out_data, \
    prepare_one_out_data_loader, load_one_out_data_with_difference, load_tight_one_out_data, \
    prepare_tight_one_out_data_loader
from utils.pretrain import pretrain_kdd_model


def main():
    # TIGHT means that, when running this mode, we will be using a very small overlap for both pretrain and finetune
    # In this case, in order to distinguish between samples. During finetune, only the first 10% of the data are used for finetune, and the next 90% of the data are used for testing.
    # PLEASE DO NOT USE MIXED TEST MODE IN THIS FILE, THE DIFFERENCE LOADER IS NOT IMPLEMENTED FOR MIX, AND IT WILL ALSO NOT BE.
    # Load the config from JSON file first
    with open("utils/config.json", "r") as file:
        config = json.load(file)
    print(config)

    # config["general"]["pretrain_model"] = "results/kdd_model/One_out/linear/pretrain/window_size_30sec/freeze_False_epoch_500_lr_0.0001_d_hidden_64_d_ff_256_n_heads_8_n_layer_1_pos_encode_learnable_activation_gelu_norm_BatchNorm"

    config["general"]["test_set"] = "Reading" # Reading or Desktop

    config["general"]["window_size"] = 150
    config["general"]["overlap"] = 0.899
    config["general"]["batch_size"] = 128
    config["kdd_pretrain"]["epoch"] = 700
    config["kdd_finetune"]["epoch"] = 600

    config["kdd_model"]["d_hidden"] = 64
    config["kdd_model"]["d_ff"] = 256
    config["kdd_model"]["n_heads"] = 8
    config["kdd_model"]["n_layers"] = 3


    # First load the data into dataloader according to chosen test_mode: Mixed or One_out
    if config["general"]["test_mode"] == "Mixed":
        data, labels, encoder = load_mixed_data(window_size=config["general"]["window_size"],
                                                overlap=config["general"]["overlap"],
                                                data_set=config["general"]["test_set"])

        num_classes = len(encoder.classes_)
        feat_dim = data[0].shape[1]
        labels_dim = labels.shape
        config["general"]["feat_dim"] = feat_dim
        print(f"The number of classes is {num_classes}, the feat_dim is {feat_dim}, the labels_dim is {labels_dim}")

        eyegaze_data_loader = (prepare_mixed_data_loader
                               (data, labels, batch_size=config["general"]["batch_size"],
                                max_len=config["general"]["window_size"]))

    elif config["general"]["test_mode"] == "One_out":
        train_data, train_labels, test_train_data, test_train_labels, test_test_data, test_test_labels, encoder = (
            load_tight_one_out_data
            (window_size=config["general"]["window_size"],
             overlap=config["general"]["overlap"],
             data_set=config["general"]["test_set"]))

        num_classes = len(encoder.classes_)
        feat_dim = train_data[0].shape[1]
        config["general"]["feat_dim"] = feat_dim
        print(f"The number of classes is {num_classes}, the feat_dim is {feat_dim}")

        eyegaze_data_loader = (prepare_tight_one_out_data_loader
                               (train_data, train_labels, test_train_data, test_train_labels, test_test_data,
                                test_test_labels,
                                batch_size=config["general"]["batch_size"],
                                max_len=config["general"]["window_size"]))
    else:
        print("Either Mixed / One_out")
        sys.exit()

    # ==================================================================================================================
    # If the pretrain_model path is not provided, start with pretraining the model
    if config["general"]["pretrain_model"] is None:
        hyperparameters = KDD_Pretrain_Hyperparameters(config)
        model = kdd_model4pretrain(config, feat_dim)
        loss = hyperparameters.loss
        optimizer = hyperparameters.optimizer(model.parameters(), hyperparameters.lr,
                                              weight_decay=hyperparameters.weight_decay)

        pretrain_kdd_model(model, loss, optimizer, eyegaze_data_loader[0], config)

    # If the pretrain_model path is provided, meaning that there is already a pretrained model, then directly finetune
    # After pretrain, finetune will be performed automatically, because the pretrain_model will be filled
    hyperparameters = KDD_Finetune_Hyperparameters(config)
    model = kdd_model4finetune(config, feat_dim, num_classes)
    loss = hyperparameters.loss
    optimizer = hyperparameters.optimizer(model.parameters(), hyperparameters.lr,
                                          weight_decay=hyperparameters.weight_decay)

    # eyegaze_data_loader[1] is the training set, and eyegaze_data_loader[2] is the validation set
    finetune_kdd_model(model, loss, optimizer, eyegaze_data_loader[1], eyegaze_data_loader[2], config)

    eval_finetune_kdd_model(model, eyegaze_data_loader[3], config, encoder)


if __name__ == "__main__":
    main()
