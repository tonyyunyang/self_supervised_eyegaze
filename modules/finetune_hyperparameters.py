import sys

import torch

from modules.loss import NoFussCrossEntropyLoss
from modules.optimizer import RAdam


class KDD_Finetune_Hyperparameters:
    def __init__(self, config):
        self.epoch = config["kdd_finetune"]["epoch"]
        self.batch = config["general"]["batch_size"]
        self.lr = config["kdd_finetune"]["lr"]

        optimizer = config["kdd_finetune"]["optimizer"]
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam
        elif optimizer == "RAdam":
            self.optimizer = RAdam
        else:
            print(f"Optimizer either Adam or RAdam")
            sys.exit()

        self.loss = NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch element

        weight_decay = config["kdd_finetune"]["weight_decay"]
        if weight_decay is None:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay