import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from modules.dataset import ImputationDataset, ClassiregressionDataset, collate_unsuperv, collate_superv, \
    LIBERTDataset4Pretrain, Preprocess4Mask
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
                window = df.iloc[i:i + window_size].values  # Convert window to numpy array
                data.append(window)  # Append the entire (150, 2) array as a single item
                labels.append(label)

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    print_encoded_classes(encoder)

    return np.array(data), np.array(encoded_labels), encoder


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

    encoder = LabelEncoder()
    encoded_train_labels = encoder.fit_transform(train_labels)
    encoded_test_labels = encoder.fit_transform(test_labels)
    print_encoded_classes(encoder)

    return np.array(train_data), np.array(encoded_train_labels), np.array(test_data), np.array(encoded_test_labels), encoder


def prepare_mixed_data_loader(data, labels, batch_size, max_len):
    # Get the range of indices for the data
    indices = list(range(len(data)))

    # Split the indices
    # Use only 10% of data for finetune
    remaining_indices, finetune_indices = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    # Split the rest of the data into pretrain and testing
    pretrain_indices, test_indices = train_test_split(remaining_indices, test_size=0.15, random_state=42, shuffle=True)
    # Split the finetune data into training and validation
    finetune_train_indices, finetune_val_indices = train_test_split(finetune_indices, test_size=0.3, random_state=42,
                                                                    shuffle=True)

    pretrain_imputation_dataset = ImputationDataset(data, pretrain_indices, mean_mask_length=3, masking_ratio=0.15)
    finetune_train_classification_dataset = ClassiregressionDataset(data, labels, finetune_train_indices)
    finetune_val_classification_dataset = ClassiregressionDataset(data, labels, finetune_val_indices)
    test_classification_dataset = ClassiregressionDataset(data, labels, test_indices)

    pretrain_loader = (DataLoader
                       (dataset=pretrain_imputation_dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda x: collate_unsuperv(x, max_len=max_len)))
    finetune_train_loader = (DataLoader
                             (dataset=finetune_train_classification_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    finetune_val_loader = (DataLoader
                           (dataset=finetune_val_classification_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    test_loader = DataLoader(dataset=test_classification_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: collate_superv(x, max_len=max_len))

    return pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader


def limu_prepare_mixed_data_loader(config, data, labels, batch_size, max_len):
    # Get the range of indices for the data
    indices = list(range(len(data)))

    # Split the indices
    # Use only 10% of data for finetune
    remaining_indices, finetune_indices = train_test_split(indices, test_size=0.1, random_state=42, shuffle=True)
    # Split the rest of the data into pretrain and testing
    pretrain_indices, test_indices = train_test_split(remaining_indices, test_size=0.15, random_state=42, shuffle=True)
    # Split the finetune data into training and validation
    finetune_train_indices, finetune_val_indices = train_test_split(finetune_indices, test_size=0.3, random_state=42,
                                                                    shuffle=True)

    pretrain_imputation_dataset = LIBERTDataset4Pretrain(data, pretrain_indices, pipeline=[Preprocess4Mask(config)])
    finetune_train_classification_dataset = ClassiregressionDataset(data, labels, finetune_train_indices)
    finetune_val_classification_dataset = ClassiregressionDataset(data, labels, finetune_val_indices)
    test_classification_dataset = ClassiregressionDataset(data, labels, test_indices)

    pretrain_loader = DataLoader(dataset=pretrain_imputation_dataset, batch_size=batch_size, shuffle=True)
    finetune_train_loader = (DataLoader
                             (dataset=finetune_train_classification_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    finetune_val_loader = (DataLoader
                           (dataset=finetune_val_classification_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    test_loader = DataLoader(dataset=test_classification_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: collate_superv(x, max_len=max_len))

    return pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader


def prepare_one_out_data_loader(train_data, train_labels, test_data, test_labels, batch_size, max_len):
    # Get the range of the indices for pretrain, fine tune and testing data
    pretrain_indices = np.arange(len(train_data))
    finetune_test_indices = list(range(len(test_data)))

    # Split the indices for finetune and testing
    test_indices, finetune_indices = train_test_split(finetune_test_indices, test_size=0.1, random_state=42, shuffle=True)
    # Split the finetune data into training and validation
    finetune_train_indices, finetune_val_indices = train_test_split(finetune_indices, test_size=0.3, random_state=42, shuffle=True)

    print(f"Pretrain samples amount: {len(pretrain_indices)}")
    print(f"Finetune training samples amount: {len(finetune_train_indices)}")
    print(f"Finetune validation samples amount: {len(finetune_val_indices)}")
    print(f"Final testing samples amount: {len(test_indices)}")

    pretrain_imputation_dataset = ImputationDataset(train_data, pretrain_indices, mean_mask_length=3, masking_ratio=0.15)
    finetune_train_classification_dataset = ClassiregressionDataset(test_data, test_labels, finetune_train_indices)
    finetune_val_classification_dataset = ClassiregressionDataset(test_data, test_labels, finetune_val_indices)
    test_classification_dataset = ClassiregressionDataset(test_data, test_labels, test_indices)

    pretrain_loader = (DataLoader
                       (dataset=pretrain_imputation_dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda x: collate_unsuperv(x, max_len=max_len)))
    finetune_train_loader = (DataLoader
                             (dataset=finetune_train_classification_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    finetune_val_loader = (DataLoader
                           (dataset=finetune_val_classification_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    test_loader = DataLoader(dataset=test_classification_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: collate_superv(x, max_len=max_len))

    return pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader


def limu_prepare_one_out_data_loader(config, train_data, train_labels, test_data, test_labels, batch_size, max_len):
    # Get the range of the indices for pretrain, fine tune and testing data
    pretrain_indices = np.arange(len(train_data))
    finetune_test_indices = list(range(len(test_data)))

    # Split the indices for finetune and testing
    test_indices, finetune_indices = train_test_split(finetune_test_indices, test_size=0.1, random_state=42, shuffle=True)
    # Split the finetune data into training and validation
    finetune_train_indices, finetune_val_indices = train_test_split(finetune_indices, test_size=0.3, random_state=42, shuffle=True)

    print(f"Pretrain samples amount: {len(pretrain_indices)}")
    print(f"Finetune training samples amount: {len(finetune_train_indices)}")
    print(f"Finetune validation samples amount: {len(finetune_val_indices)}")
    print(f"Final testing samples amount: {len(test_indices)}")

    pretrain_imputation_dataset = LIBERTDataset4Pretrain(train_data, pretrain_indices, pipeline=[Preprocess4Mask(config)])
    finetune_train_classification_dataset = ClassiregressionDataset(test_data, test_labels, finetune_train_indices)
    finetune_val_classification_dataset = ClassiregressionDataset(test_data, test_labels, finetune_val_indices)
    test_classification_dataset = ClassiregressionDataset(test_data, test_labels, test_indices)

    pretrain_loader = DataLoader(dataset=pretrain_imputation_dataset, batch_size=batch_size, shuffle=True)
    finetune_train_loader = (DataLoader
                             (dataset=finetune_train_classification_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    finetune_val_loader = (DataLoader
                           (dataset=finetune_val_classification_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: collate_superv(x, max_len=max_len)))
    test_loader = DataLoader(dataset=test_classification_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda x: collate_superv(x, max_len=max_len))

    return pretrain_loader, finetune_train_loader, finetune_val_loader, test_loader


def print_encoded_classes(encoder):
    for original_label, encoded_label in zip(encoder.classes_, range(len(encoder.classes_))):
        print(f"Class: {original_label} -> Encoded Value: {encoded_label}")
