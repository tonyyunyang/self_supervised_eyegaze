import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from modules.dataset import ImputationDataset, ClassiregressionDataset, collate_unsuperv, collate_superv
from torch.utils.data import DataLoader


def load_mixed_data(window_size, overlap):
    directory = "data/DesktopActivity/ALL"
    step_size = int(window_size * (1 - overlap))
    data = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            label = filename.split("_")[1].split(".")[0]
            df = pd.read_csv(os.path.join(directory, filename), header=None)

            for i in range(0, len(df) - window_size + 1, step_size):
                window = df.iloc[i:i + window_size]
                data.append(window)
                labels.append(label)

    return np.array(data), np.array(labels)


def load_one_out_data(window_size, overlap):
    directory = "data/DesktopActivity/ALL"
    step_size = int(window_size * (1 - overlap))

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            label = filename.split("_")[1].split(".")[0]
            df = pd.read_csv(os.path.join(directory, filename), header=None)

            for i in range(0, len(df) - window_size + 1, step_size):
                window = df.iloc[i:i + window_size]

                if filename.startswith("P08"):
                    test_data.append(window)
                    test_labels.append(label)
                else:
                    train_data.append(window)
                    train_labels.append(label)

    return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)


def prepare_mixed_data_loader(data, labels, batch_size, max_len):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    print_encoded_classes(encoder)

    (remaining_data, finetune_data, remaining_labels, finetune_labels) = (train_test_split
                                                                          (data, encoded_labels, test_size=0.1,
                                                                           random_state=42, shuffle=True))
    (pretrain_data, test_data, pretrain_labels, test_labels) = (train_test_split
                                                                (remaining_data, remaining_labels, test_size=0.15))
    (finetune_train_data, finetune_val_data, finetune_train_labels, finetune_val_labels) = (train_test_split
                                                                                            (finetune_data,
                                                                                             finetune_labels,
                                                                                             test_size=0.3,
                                                                                             random_state=42,
                                                                                             shuffle=True))
    pretrain_indicies = np.arange(len(pretrain_data))
    finetune_train_indicies = np.arange(len(finetune_train_data))
    finetune_val_indicies = np.arange(len(finetune_val_data))
    test_indicies = np.arange(len(test_data))

    pretrain_imputation_dataset = (ImputationDataset
                                   (pretrain_data, pretrain_indicies, mean_mask_length=3, masking_ratio=0.15))
    finetune_train_classification_dataset = (ClassiregressionDataset
                                             (finetune_train_data, finetune_train_labels, finetune_train_indicies))
    finetune_val_classification_dataset = (ClassiregressionDataset
                                           (finetune_val_data, finetune_val_labels, finetune_val_indicies))
    test_classification_dataset = ClassiregressionDataset(test_data, test_labels, test_indicies)

    pretrain_loader = (DataLoader
                       (dataset=pretrain_imputation_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                        pin_memory=True, collate_fn=lambda x: collate_unsuperv(x, max_len=max_len)))
    finetune_train_loader = (DataLoader
                             (dataset=finetune_train_classification_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    finetune_val_loader = (DataLoader
                           (dataset=finetune_val_classification_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    test_loader = DataLoader(dataset=test_classification_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                             pin_memory=True, collate_fn=lambda x: collate_superv(x, max_len=max_len))

    return (pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader), encoder


def prepare_one_out_data_loader(train_data, train_labels, test_data, test_labels, batch_size, max_len):
    encoder = LabelEncoder()
    encoded_train_labels = encoder.fit_transform(train_labels)
    encoded_test_labels = encoder.fit_transform(test_labels)
    print_encoded_classes(encoder)

    (pretrain_data, finetune_data, pretrain_labels, finetune_labels) = (train_test_split
                                                                        (train_data, encoded_train_labels,
                                                                         test_size=0.1, random_state=42, shuffle=True))
    (finetune_train_data, finetune_val_data, finetune_train_labels, finetune_val_labels) = (train_test_split
                                                                                            (finetune_data,
                                                                                             finetune_labels,
                                                                                             test_size=0.3,
                                                                                             random_state=42,
                                                                                             shuffle=True))
    pretrain_indicies = np.arange(len(pretrain_data))
    finetune_train_indicies = np.arange(len(finetune_train_data))
    finetune_val_indicies = np.arange(len(finetune_val_data))
    test_indicies = np.arange(len(test_data))

    pretrain_imputation_dataset = (ImputationDataset
                                   (pretrain_data, pretrain_indicies, mean_mask_length=3, masking_ratio=0.15))
    finetune_train_classification_dataset = (ClassiregressionDataset
                                             (finetune_train_data, finetune_train_labels, finetune_train_indicies))
    finetune_val_classification_dataset = (ClassiregressionDataset
                                           (finetune_val_data, finetune_val_labels, finetune_val_indicies))
    test_classification_dataset = ClassiregressionDataset(test_data, encoded_test_labels, test_indicies)

    pretrain_loader = (DataLoader
                       (dataset=pretrain_imputation_dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                        pin_memory=True, collate_fn=lambda x: collate_unsuperv(x, max_len=max_len)))
    finetune_train_loader = (DataLoader
                             (dataset=finetune_train_classification_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=8, pin_memory=True, collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    finetune_val_loader = (DataLoader
                           (dataset=finetune_val_classification_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=8, pin_memory=True, collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    test_loader = DataLoader(dataset=test_classification_dataset, batch_size=batch_size, shuffle=False, num_workers=8,
                             pin_memory=True, collate_fn=lambda x: collate_superv(x, max_len=max_len))

    return (pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader), encoder


def print_encoded_classes(encoder):
    for original_label, encoded_label in zip(encoder.classes_, range(len(encoder.classes_))):
        print(f"Class: {original_label} -> Encoded Value: {encoded_label}")
