import json
import sys

from modules.finetune_hyperparameters import KDD_Finetune_Hyperparameters, KDD_Fully_Supervised_Finetune_Hyperparameters
from modules.kdd_model import kdd_model4fullysupervise_finetune, kdd_model4fullysupervise_pretrain, kdd_model4pretrain, kdd_model4finetune
from modules.pretrain_hyperparameters import KDD_Fully_Supervised_Pretrain_Hyperparameters, KDD_Pretrain_Hyperparameters
from utils.finetune import finetune_kdd_model, eval_finetune_kdd_model, fully_supervise_train_kdd_model_finetune, fully_supervise_train_kdd_model_pretrain
from utils.load_data_from_file import load_mixed_data, prepare_fully_supervised_one_out_data_loader, prepare_mixed_data_loader, load_one_out_data, \
    prepare_one_out_data_loader
from utils.pretrain import pretrain_kdd_model


def main():
    # Load the config from JSON file first
    with open("utils/config.json", "r") as file:
        config = json.load(file)
    print(config)

    config["general"]["pretrain_model"] = "results/Desktop/kdd_model/One_out/convolution/fully_supervised_pretrain/window_size_5sec/feat_dim_2/kernelsize_20_stride_10_dilation_1_padding_0/freeze_False_epoch_1200_lr_0.001_d_hidden_8_d_ff_16_n_heads_4_n_layer_8_pos_encode_learnable_activation_gelu_norm_BatchNorm"
   
    config["general"]["test_set"] = "Desktop" # Reading or Desktop or CosSin

    config["general"]["window_size"] = 150
    config["general"]["overlap"] = 0.8
    config["general"]["batch_size"] = 128
    config["kdd_pretrain"]["epoch"] = 1200
    config["kdd_finetune"]["epoch"] = 6000

    config["kdd_model"]["d_hidden"] = 8
    config["kdd_model"]["d_ff"] = 16
    config["kdd_model"]["n_heads"] = 4
    config["kdd_model"]["n_layers"] = 8
    
    config["kdd_model"]["projection"] = "convolution"
    # config["general"]["freeze"] = True
    
    # First load the data into dataloader according to chosen test_mode: Mixed or One_out
    if config["general"]["test_mode"] == "Mixed":
        data, labels, encoder = load_mixed_data(window_size=config["general"]["window_size"],
                                                overlap=config["general"]["overlap"])

        num_classes = len(encoder.classes_)
        feat_dim = data[0].shape[1]
        labels_dim = labels.shape
        config["general"]["feat_dim"] = feat_dim
        print(f"The number of classes is {num_classes}, the feat_dim is {feat_dim}, the labels_dim is {labels_dim}")

        eyegaze_data_loader = (prepare_mixed_data_loader
                               (data, labels, batch_size=config["general"]["batch_size"],
                                max_len=config["general"]["window_size"]))

    elif config["general"]["test_mode"] == "One_out":
        train_data, train_labels, test_data, test_labels, encoder = (load_one_out_data
                                                                     (window_size=config["general"]["window_size"],
                                                                      overlap=config["general"]["overlap"],
                                                                      data_set=config["general"]["test_set"]))

        num_classes = len(encoder.classes_)
        feat_dim = train_data[0].shape[1]
        config["general"]["feat_dim"] = feat_dim
        print(f"The number of classes is {num_classes}, the feat_dim is {feat_dim}")

        eyegaze_data_loader = (prepare_fully_supervised_one_out_data_loader
                               (train_data, train_labels, test_data, test_labels,
                                batch_size=config["general"]["batch_size"],
                                max_len=config["general"]["window_size"]))
    else:
        print("Either Mixed / One_out")
        sys.exit()

    # ==================================================================================================================
    # If the pretrain_model path is not provided, start with pretraining the model
    if config["general"]["pretrain_model"] is None:
        hyperparameters = KDD_Fully_Supervised_Pretrain_Hyperparameters(config)
        model = kdd_model4fullysupervise_pretrain(config, feat_dim, num_classes)
        loss = hyperparameters.loss
        optimizer = hyperparameters.optimizer(model.parameters(), hyperparameters.lr,
                                              weight_decay=hyperparameters.weight_decay)

        fully_supervise_train_kdd_model_pretrain(model, loss, optimizer, eyegaze_data_loader[0], eyegaze_data_loader[1], config)

    # If the pretrain_model path is provided, meaning that there is already a pretrained model, then directly finetune
    # After pretrain, finetune will be performed automatically, because the pretrain_model will be filled
    hyperparameters = KDD_Fully_Supervised_Finetune_Hyperparameters(config)
    model = kdd_model4fullysupervise_finetune(config, feat_dim, num_classes)
    loss = hyperparameters.loss
    optimizer = hyperparameters.optimizer(model.parameters(), hyperparameters.lr,
                                          weight_decay=hyperparameters.weight_decay)

    # eyegaze_data_loader[1] is the training set, and eyegaze_data_loader[2] is the validation set
    fully_supervise_train_kdd_model_finetune(model, loss, optimizer, eyegaze_data_loader[2], eyegaze_data_loader[3], config)

    eval_finetune_kdd_model(model, eyegaze_data_loader[4], config, encoder)


if __name__ == "__main__":
    main()
