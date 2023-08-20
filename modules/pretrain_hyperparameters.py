import sys

from modules.optimizer import *
from modules.loss import *


class Pretrain_Hyperparameters:
    def __init__(self, config):
        self.epoch = config["pretrain"]["epoch"]
        self.batch = config["general"]["batch_size"]
        self.lr = config["pretrain"]["lr"]

        optimizer = config["pretrain"]["optimizer"]
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam
        elif optimizer == "RAdam":
            self.optimizer = RAdam
        else:
            print(f"Optimizer either Adam or RAdam")
            sys.exit()

        task = config["pretrain"]["task"]
        if (task == "imputation") or (task == "transduction"):
            self.loss = MaskedMSELoss(reduction='none')  # outputs loss for each batch element
        elif task == "classification":
            self.loss = NoFussCrossEntropyLoss(reduction='none')  # outputs loss for each batch sample
        elif task == "regression":
            self.loss = nn.MSELoss(reduction='none')  # outputs loss for each batch sample
        else:
            raise ValueError("Loss module for task '{}' does not exist".format(task))

        weight_decay = config["pretrain"]["weight_decay"]
        if weight_decay is None:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
