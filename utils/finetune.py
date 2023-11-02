import os
import sys
import time

import torch
# Set the flag of deterministic to true to reproduce the results of ConvTranspose1D
# torch.backends.cudnn.deterministic = True

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
from modules.loss import l2_reg_loss


def finetune_kdd_model(model, loss, optimizer, train_set, val_set, config):
    # Check if CUDA is available
    # if not torch.cuda.is_available():
    #     print("CUDA is not available. Please activate CUDA for GPU acceleration.")
    #     print("This is a computationally expensive training process and requires GPU acceleration.")
    #     sys.exit()
    # print("CUDA ACTIVATED")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=============================================================\n"
          f"=====================Training via {device}===================\n"
          f"=============================================================")

    path = os.path.join("results", f"{config['general']['test_set']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"kdd_model")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['general']['test_mode']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['kdd_model']['projection']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"finetune")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"window_size_{int(config['general']['window_size'] / 30)}sec")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"feat_dim_{int(config['general']['feat_dim'])}")
    os.makedirs(path, exist_ok=True)

    if config['kdd_model']['projection'] == 'convolution':
        if int(config['general']['window_size'] / 30) == 5:
            path = os.path.join(path, f"kernelsize_{int(config['conv1d_5sec']['first']['kernel_size'])}_"
                                      f"stride_{int(config['conv1d_5sec']['first']['stride'])}_"
                                      f"dilation_{int(config['conv1d_5sec']['first']['dilation'])}_"
                                      f"padding_{int(config['conv1d_5sec']['first']['padding'])}")
            os.makedirs(path, exist_ok=True)
        elif int(config['general']['window_size'] / 30) == 30:
            path = os.path.join(path, f"kernelsize_{int(config['conv1d_30sec']['first']['kernel_size'])}_"
                                      f"stride_{int(config['conv1d_30sec']['first']['stride'])}_"
                                      f"dilation_{int(config['conv1d_30sec']['first']['dilation'])}_"
                                      f"padding_{int(config['conv1d_30sec']['first']['padding'])}")
            os.makedirs(path, exist_ok=True)
        else:
            sys.exit("Please create the corresponding folder for the time interval first")

    path = os.path.join(path, f"freeze_{config['general']['freeze']}_epoch_{config['kdd_finetune']['epoch']}_"
                              f"lr_{format(config['kdd_finetune']['lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"d_hidden_{config['kdd_model']['d_hidden']}_d_ff_{config['kdd_model']['d_ff']}_"
                              f"n_heads_{config['kdd_model']['n_heads']}_n_layer_{config['kdd_model']['n_layers']}_"
                              f"pos_encode_{config['kdd_model']['pos_encoding']}_"
                              f"activation_{config['kdd_model']['activation']}_norm_{config['kdd_model']['norm']}")
    os.makedirs(path, exist_ok=True)

    config["general"]["finetune_model"] = path

    model = model.to(device)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_f1_score_list = []

    best_val_acc = 0  # Initialize variable to keep track of best validation accuracy

    for epoch in range(1, config["kdd_finetune"]["epoch"] + 1):
        epoch_start_time = time.time()

        train_loss, val_loss, val_acc, val_f1 = finetune_kdd_epoch(model, loss, optimizer, train_set, val_set, config,
                                                                   device, epoch, l2_reg=False)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)
        val_f1_score_list.append(val_f1)

        epoch_runtime = time.time() - epoch_start_time

        print(
            f"Epoch {epoch}/{config['kdd_finetune']['epoch']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}, Time: {epoch_runtime}")

        # Save the best model based on validation accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(config["general"]["finetune_model"], "best_model.pth")
            )

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["general"]["finetune_model"], "continue_model.pth")
    )

    kdd_finetune_save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config)


def fully_supervise_train_kdd_model_pretrain(model, loss, optimizer, train_set, val_set, config):
    # Check if CUDA is available
    # if not torch.cuda.is_available():
    #     print("CUDA is not available. Please activate CUDA for GPU acceleration.")
    #     print("This is a computationally expensive training process and requires GPU acceleration.")
    #     sys.exit()
    # print("CUDA ACTIVATED")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=============================================================\n"
          f"=====================Training via {device}===================\n"
          f"=============================================================")

    path = os.path.join("results", f"{config['general']['test_set']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"kdd_model")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['general']['test_mode']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['kdd_model']['projection']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"fully_supervised_pretrain")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"window_size_{int(config['general']['window_size'] / 30)}sec")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"feat_dim_{int(config['general']['feat_dim'])}")
    os.makedirs(path, exist_ok=True)
    
    if config['kdd_model']['projection'] == 'convolution':
        if int(config['general']['window_size'] / 30) == 5:
            path = os.path.join(path, f"kernelsize_{int(config['conv1d_5sec']['first']['kernel_size'])}_"
                                      f"stride_{int(config['conv1d_5sec']['first']['stride'])}_"
                                      f"dilation_{int(config['conv1d_5sec']['first']['dilation'])}_"
                                      f"padding_{int(config['conv1d_5sec']['first']['padding'])}")
            os.makedirs(path, exist_ok=True)
        elif int(config['general']['window_size'] / 30) == 30:
            path = os.path.join(path, f"kernelsize_{int(config['conv1d_30sec']['first']['kernel_size'])}_"
                                      f"stride_{int(config['conv1d_30sec']['first']['stride'])}_"
                                      f"dilation_{int(config['conv1d_30sec']['first']['dilation'])}_"
                                      f"padding_{int(config['conv1d_30sec']['first']['padding'])}")
            os.makedirs(path, exist_ok=True)
        else:
            sys.exit("Please create the corresponding folder for the time interval first")

    path = os.path.join(path, f"freeze_{config['general']['freeze']}_epoch_{config['kdd_pretrain']['epoch']}_"
                              f"lr_{format(config['kdd_pretrain']['lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"d_hidden_{config['kdd_model']['d_hidden']}_d_ff_{config['kdd_model']['d_ff']}_"
                              f"n_heads_{config['kdd_model']['n_heads']}_n_layer_{config['kdd_model']['n_layers']}_"
                              f"pos_encode_{config['kdd_model']['pos_encoding']}_"
                              f"activation_{config['kdd_model']['activation']}_norm_{config['kdd_model']['norm']}")
    os.makedirs(path, exist_ok=True)

    config["general"]["pretrain_model"] = path

    model = model.to(device)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_f1_score_list = []

    best_val_acc = 0  # Initialize variable to keep track of best validation accuracy

    for epoch in range(1, config["kdd_pretrain"]["epoch"] + 1):
        epoch_start_time = time.time()

        train_loss, val_loss, val_acc, val_f1 = finetune_kdd_epoch(model, loss, optimizer, train_set, val_set, config,
                                                                   device, epoch, l2_reg=False)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)
        val_f1_score_list.append(val_f1)

        epoch_runtime = time.time() - epoch_start_time

        print(
            f"Epoch {epoch}/{config['kdd_pretrain']['epoch']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}, Time: {epoch_runtime}")

        # Save the best model based on validation accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(config["general"]["pretrain_model"], "best_model.pth")
            )

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["general"]["pretrain_model"], "continue_model.pth")
    )

    kdd_fully_supervised_pretrain_save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config)


def fully_supervise_train_kdd_model_finetune(model, loss, optimizer, train_set, val_set, config):
    # Check if CUDA is available
    # if not torch.cuda.is_available():
    #     print("CUDA is not available. Please activate CUDA for GPU acceleration.")
    #     print("This is a computationally expensive training process and requires GPU acceleration.")
    #     sys.exit()
    # print("CUDA ACTIVATED")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=============================================================\n"
          f"=====================Training via {device}===================\n"
          f"=============================================================")

    path = os.path.join("results", f"{config['general']['test_set']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"kdd_model")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['general']['test_mode']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['kdd_model']['projection']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"fully_supervised_finetune")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"window_size_{int(config['general']['window_size'] / 30)}sec")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"feat_dim_{int(config['general']['feat_dim'])}")
    os.makedirs(path, exist_ok=True)
    
    if config['kdd_model']['projection'] == 'convolution':
        if int(config['general']['window_size'] / 30) == 5:
            path = os.path.join(path, f"kernelsize_{int(config['conv1d_5sec']['first']['kernel_size'])}_"
                                      f"stride_{int(config['conv1d_5sec']['first']['stride'])}_"
                                      f"dilation_{int(config['conv1d_5sec']['first']['dilation'])}_"
                                      f"padding_{int(config['conv1d_5sec']['first']['padding'])}")
            os.makedirs(path, exist_ok=True)
        elif int(config['general']['window_size'] / 30) == 30:
            path = os.path.join(path, f"kernelsize_{int(config['conv1d_30sec']['first']['kernel_size'])}_"
                                      f"stride_{int(config['conv1d_30sec']['first']['stride'])}_"
                                      f"dilation_{int(config['conv1d_30sec']['first']['dilation'])}_"
                                      f"padding_{int(config['conv1d_30sec']['first']['padding'])}")
            os.makedirs(path, exist_ok=True)
        else:
            sys.exit("Please create the corresponding folder for the time interval first")

    path = os.path.join(path, f"freeze_{config['general']['freeze']}_epoch_{config['kdd_finetune']['epoch']}_"
                              f"lr_{format(config['kdd_finetune']['lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"d_hidden_{config['kdd_model']['d_hidden']}_d_ff_{config['kdd_model']['d_ff']}_"
                              f"n_heads_{config['kdd_model']['n_heads']}_n_layer_{config['kdd_model']['n_layers']}_"
                              f"pos_encode_{config['kdd_model']['pos_encoding']}_"
                              f"activation_{config['kdd_model']['activation']}_norm_{config['kdd_model']['norm']}")
    os.makedirs(path, exist_ok=True)

    config["general"]["finetune_model"] = path

    model = model.to(device)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_f1_score_list = []

    best_val_acc = 0  # Initialize variable to keep track of best validation accuracy

    for epoch in range(1, config["kdd_finetune"]["epoch"] + 1):
        epoch_start_time = time.time()

        train_loss, val_loss, val_acc, val_f1 = finetune_kdd_epoch(model, loss, optimizer, train_set, val_set, config,
                                                                   device, epoch, l2_reg=False)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)
        val_f1_score_list.append(val_f1)

        epoch_runtime = time.time() - epoch_start_time

        print(
            f"Epoch {epoch}/{config['kdd_finetune']['epoch']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}, Time: {epoch_runtime}")

        # Save the best model based on validation accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(config["general"]["finetune_model"], "best_model.pth")
            )

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["general"]["finetune_model"], "continue_model.pth")
    )

    kdd_finetune_save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config)


def finetune_kdd_epoch(model, loss, optimizer, train_set, val_set, config, device, epoch, l2_reg=False):
    # Train the model first
    model = model.train()
    train_loss = 0
    total_train_samples = 0

    for i, batch in enumerate(train_set):
        X, targets, padding_masks, IDs = batch
        targets = targets.to(device)
        # padding_masks = padding_masks.to(device)
        predictions = model(X.to(device))
        compute_loss = loss(predictions, targets)
        batch_loss = torch.sum(compute_loss)
        mean_loss = batch_loss / len(compute_loss)

        if l2_reg:
            total_loss = mean_loss + l2_reg * l2_reg_loss(model)
        else:
            total_loss = mean_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        total_loss.backward()

        # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()

        with torch.no_grad():
            total_train_samples += len(compute_loss)
            train_loss += batch_loss.item()

    train_loss = train_loss / total_train_samples

    # Evaluate the model
    model = model.eval()
    val_loss = 0
    total_val_samples = 0

    # Initialize lists to store true and predicted labels
    true_labels = []
    pred_labels = []

    for i, batch in enumerate(val_set):
        X, targets, padding_masks, IDs = batch
        targets = targets.to(device)
        # padding_masks = padding_masks.to(device)
        predictions = model(X.to(device))

        compute_loss = loss(predictions, targets)
        batch_loss = torch.sum(compute_loss).cpu().item()
        mean_loss = batch_loss / len(compute_loss)

        # Collect true and predicted labels
        true_labels.extend(targets.cpu().numpy())
        pred_labels.extend(torch.argmax(predictions, dim=1).cpu().numpy())

        total_val_samples += len(compute_loss)
        val_loss += batch_loss

    val_loss = val_loss / total_val_samples

    # Compute validation accuracy and F1 score
    val_acc = accuracy_score(true_labels, pred_labels)
    val_f1 = f1_score(true_labels, pred_labels, average='weighted')  # Use 'weighted' if you have imbalanced classes

    return train_loss, val_loss, val_acc, val_f1


def kdd_finetune_save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config):
    # Plot and save the metrics
    epochs = range(1, config["kdd_finetune"]["epoch"] + 1)
    plt.figure(figsize=(12, 8))

    plt.plot(epochs, train_loss_list, label='Train Loss', linestyle='--')
    plt.plot(epochs, val_loss_list, label='Validation Loss', linestyle='--')
    plt.plot(epochs, val_accuracy_list, label='Validation Accuracy', linestyle='-')
    plt.plot(epochs, val_f1_score_list, label='Validation F1 Score', linestyle='-')

    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config["general"]["finetune_model"], "all_metrics_plot.png"))


def kdd_fully_supervised_pretrain_save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config):
    # Plot and save the metrics
    epochs = range(1, config["kdd_pretrain"]["epoch"] + 1)
    plt.figure(figsize=(12, 8))

    plt.plot(epochs, train_loss_list, label='Train Loss', linestyle='--')
    plt.plot(epochs, val_loss_list, label='Validation Loss', linestyle='--')
    plt.plot(epochs, val_accuracy_list, label='Validation Accuracy', linestyle='-')
    plt.plot(epochs, val_f1_score_list, label='Validation F1 Score', linestyle='-')

    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config["general"]["pretrain_model"], "all_metrics_plot.png"))


def eval_finetune_kdd_model(model, test_set, config, encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(config["general"]["finetune_model"], "best_model.pth")))
    model = model.to(device)
    model.eval()

    # Evaluate the model on the test set
    true_labels = []
    pred_labels = []

    for i, batch in enumerate(test_set):
        X, targets, padding_masks, IDs = batch
        targets = targets.to(device)
        predictions = model(X.to(device))

        true_labels.extend(targets.cpu().numpy())
        pred_labels.extend(torch.argmax(predictions, dim=1).cpu().numpy())

    # Compute validation accuracy and F1 score
    test_acc = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average='weighted')  # Use 'weighted' if you have imbalanced classes

    # Decode the labels
    true_labels_decoded = encoder.inverse_transform(true_labels)
    pred_labels_decoded = encoder.inverse_transform(pred_labels)

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels_decoded, pred_labels_decoded)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Save the confusion matrix plot
    plt.savefig(os.path.join(config["general"]["finetune_model"], f"f1_{format(test_f1, '.5f').rstrip('0').rstrip('.')}_"
                                                                  f"acc_{format(test_acc, '.5f').rstrip('0').rstrip('.')}_"
                                                                  f"confusion_matrix.png"))


def finetune_limu_model(model, loss, optimizer, train_set, val_set, config):
    # Check if CUDA is available
    # if not torch.cuda.is_available():
    #     print("CUDA is not available. Please activate CUDA for GPU acceleration.")
    #     print("This is a computationally expensive training process and requires GPU acceleration.")
    #     sys.exit()
    # print("CUDA ACTIVATED")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=============================================================\n"
          f"=====================Training via {device}===================\n"
          f"=============================================================")

    path = os.path.join("results", f"{config['general']['test_set']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"limu_model")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['general']['test_mode']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"finetune")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"window_size_{int(config['general']['window_size'] / 30)}sec")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"feat_dim_{int(config['general']['feat_dim'])}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"epoch_{config['limu_finetune']['epoch']}_"
                              f"lr_{format(config['limu_finetune']['lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"d_hidden_{config['limu_model']['d_hidden']}_d_ff_{config['limu_model']['d_ff']}_"
                              f"n_heads_{config['limu_model']['n_heads']}_n_layer_{config['limu_model']['n_layers']}_"
                              f"embNorm_{config['limu_model']['emb_norm']}")
    os.makedirs(path, exist_ok=True)

    config["general"]["finetune_model"] = path

    model = model.to(device)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_f1_score_list = []

    best_val_acc = 0  # Initialize variable to keep track of best validation accuracy

    for epoch in range(1, config["limu_finetune"]["epoch"] + 1):
        epoch_start_time = time.time()

        train_loss, val_loss, val_acc, val_f1 = finetune_limu_epoch(model, loss, optimizer, train_set, val_set, config,
                                                                   device, epoch, l2_reg=False)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)
        val_f1_score_list.append(val_f1)

        epoch_runtime = time.time() - epoch_start_time

        print(
            f"Epoch {epoch}/{config['limu_finetune']['epoch']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},"
            f" Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}, Time: {epoch_runtime}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(config["general"]["finetune_model"], "best_model.pth")
            )

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["general"]["finetune_model"], "continue_model.pth")
    )

    limu_finetune_save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config)


def full_supervise_train_limu_model(model, loss, optimizer, train_set, val_set, config):
    # Check if CUDA is available
    # if not torch.cuda.is_available():
    #     print("CUDA is not available. Please activate CUDA for GPU acceleration.")
    #     print("This is a computationally expensive training process and requires GPU acceleration.")
    #     sys.exit()
    # print("CUDA ACTIVATED")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=============================================================\n"
          f"=====================Training via {device}===================\n"
          f"=============================================================")

    path = os.path.join("results", f"{config['general']['test_set']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"limu_model")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['general']['test_mode']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"fully_supervised")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"window_size_{int(config['general']['window_size'] / 30)}sec")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"feat_dim_{int(config['general']['feat_dim'])}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"epoch_{config['limu_finetune']['epoch']}_"
                              f"lr_{format(config['limu_finetune']['lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"d_hidden_{config['limu_model']['d_hidden']}_d_ff_{config['limu_model']['d_ff']}_"
                              f"n_heads_{config['limu_model']['n_heads']}_n_layer_{config['limu_model']['n_layers']}_"
                              f"embNorm_{config['limu_model']['emb_norm']}")
    os.makedirs(path, exist_ok=True)

    config["general"]["finetune_model"] = path

    model = model.to(device)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    val_f1_score_list = []

    best_val_acc = 0  # Initialize variable to keep track of best validation accuracy

    for epoch in range(1, config["limu_finetune"]["epoch"] + 1):
        epoch_start_time = time.time()

        train_loss, val_loss, val_acc, val_f1 = finetune_limu_epoch(model, loss, optimizer, train_set, val_set, config,
                                                                   device, epoch, l2_reg=False)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)
        val_f1_score_list.append(val_f1)

        epoch_runtime = time.time() - epoch_start_time

        print(
            f"Epoch {epoch}/{config['limu_finetune']['epoch']}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},"
            f" Accuracy: {val_acc:.4f}, F1 Score: {val_f1:.4f}, Time: {epoch_runtime}")

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(), os.path.join(config["general"]["finetune_model"], "best_model.pth")
            )

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["general"]["finetune_model"], "continue_model.pth")
    )

    limu_finetune_save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config)


def finetune_limu_epoch(model, loss, optimizer, train_set, val_set, config, device, epoch, l2_reg=False):
    # Train the model first
    model = model.train()

    train_loss_sum = 0.0
    for i, batch in enumerate(train_set):
        X, targets, padding_masks, IDs = batch
        targets = targets.to(device)
        X = X.to(device)
        # padding_masks = padding_masks.to(device)

        optimizer.zero_grad()

        predictions = model(X, True)
        compute_loss = loss(predictions, targets)

        compute_loss = compute_loss.mean()
        compute_loss.backward()
        optimizer.step()

        train_loss_sum += compute_loss.item()

    train_loss = train_loss_sum / len(train_set)

    # Evaluate the model
    model = model.eval()
    val_loss_sum = 0

    # Initialize lists to store true and predicted labels
    true_labels = []
    pred_labels = []

    for i, batch in enumerate(val_set):
        X, targets, padding_masks, IDs = batch
        targets = targets.to(device)
        X = X.to(device)
        # padding_masks = padding_masks.to(device)

        predictions = model(X, False)

        compute_loss = loss(predictions, targets)

        compute_loss = compute_loss.mean()

        # Collect true and predicted labels
        true_labels.extend(targets.cpu().numpy())
        pred_labels.extend(torch.argmax(predictions, dim=1).cpu().numpy())

        val_loss_sum += compute_loss.item()

    val_loss = val_loss_sum / len(val_set)

    # Compute validation accuracy and F1 score
    val_acc = accuracy_score(true_labels, pred_labels)
    val_f1 = f1_score(true_labels, pred_labels, average='weighted')  # Use 'weighted' if you have imbalanced classes

    return train_loss, val_loss, val_acc, val_f1


def limu_finetune_save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config):
    # Plot and save the metrics
    epochs = range(1, config["limu_finetune"]["epoch"] + 1)
    plt.figure(figsize=(12, 8))

    plt.plot(epochs, train_loss_list, label='Train Loss', linestyle='--')
    plt.plot(epochs, val_loss_list, label='Validation Loss', linestyle='--')
    plt.plot(epochs, val_accuracy_list, label='Validation Accuracy', linestyle='-')
    plt.plot(epochs, val_f1_score_list, label='Validation F1 Score', linestyle='-')

    plt.title('Training and Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config["general"]["finetune_model"], "all_metrics_plot.png"))

def eval_finetune_limu_model(model, test_set, config, encoder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best model
    model.load_state_dict(torch.load(os.path.join(config["general"]["finetune_model"], "best_model.pth")))
    model = model.to(device)
    model.eval()

    # Evaluate the model on the test set
    true_labels = []
    pred_labels = []

    for i, batch in enumerate(test_set):
        X, targets, padding_masks, IDs = batch
        targets = targets.to(device)
        X = X.to(device)

        predictions = model(X, False)

        true_labels.extend(targets.cpu().numpy())
        pred_labels.extend(torch.argmax(predictions, dim=1).cpu().numpy())

    # Compute validation accuracy and F1 score
    test_acc = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average='weighted')  # Use 'weighted' if you have imbalanced classes

    # Decode the labels
    true_labels_decoded = encoder.inverse_transform(true_labels)
    pred_labels_decoded = encoder.inverse_transform(pred_labels)

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels_decoded, pred_labels_decoded)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')

    # Save the confusion matrix plot
    plt.savefig(os.path.join(config["general"]["finetune_model"], f"f1_{format(test_f1, '.5f').rstrip('0').rstrip('.')}_"
                                                                  f"acc_{format(test_acc, '.5f').rstrip('0').rstrip('.')}_"
                                                                  f"confusion_matrix.png"))
