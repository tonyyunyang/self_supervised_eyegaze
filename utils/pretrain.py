import os.path
import sys

import torch


def pretrain_kdd_model(model, loss, optimizer, pretrain_data, config):
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please activate CUDA for GPU acceleration.")
        print("This is a computationally expensive training process and requires GPU acceleration.")
        sys.exit()
    print("CUDA ACTIVATED")

    path = os.path.join("results", f"kdd_model")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['general']['test_mode']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['kdd_model']['projection']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"pretrain")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"freeze_{config['general']['freeze']}_epoch_{config['pretrain']['epoch']}_"
                              f"lr_{format(config['pretrain']['lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"d_hidden_{config['kdd_model']['d_hidden']}_d_ff_{config['kdd_model']['d_ff']}_"
                              f"n_heads_{config['kdd_model']['n_heads']}_n_layer_{config['kdd_model']['n_layers']}_"
                              f"pos_encode_{config['kdd_model']['pos_encoding']}_"
                              f"activation_{config['kdd_model']['activation']}_norm_{config['kdd_model']['norm']}")

    config["general"]["pretrain_model"] = path
