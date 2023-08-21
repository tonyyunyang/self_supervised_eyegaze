import sys

from modules.optimizer import *
from modules.loss import *


class KDD_Pretrain_Hyperparameters:
    def __init__(self, config):
        self.epoch = config["kdd_pretrain"]["epoch"]
        self.batch = config["general"]["batch_size"]
        self.lr = config["kdd_pretrain"]["lr"]

        optimizer = config["kdd_pretrain"]["optimizer"]
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam
        elif optimizer == "RAdam":
            self.optimizer = RAdam
        else:
            print(f"Optimizer either Adam or RAdam")
            sys.exit()

        self.loss = MaskedMSELoss(reduction='none')  # outputs loss for each batch element

        weight_decay = config["kdd_pretrain"]["weight_decay"]
        if weight_decay is None:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay


class LIMU_Pretrain_Hyperparameters:
    def __init__(self, config):
        self.epoch = config["limu_pretrain"]["epoch"]
        self.batch = config["general"]["batch_size"]
        self.lr = config["limu_pretrain"]["lr"]

        optimizer = config["limu_pretrain"]["optimizer"]
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam
        elif optimizer == "RAdam":
            self.optimizer = RAdam
        else:
            print(f"Optimizer either Adam or RAdam")
            sys.exit()

        # self.loss = nn.MSELoss(reduction='none')  # outputs loss for each batch sample
        self.loss = nn.MSELoss(reduction='none')

        weight_decay = config["limu_pretrain"]["weight_decay"]
        if weight_decay is None:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
