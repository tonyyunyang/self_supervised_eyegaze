import os
import time

import torch

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

    path = os.path.join("results", f"kdd_model")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['general']['test_mode']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"{config['kdd_model']['projection']}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"finetune")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"freeze_{config['general']['freeze']}_epoch_{config['kdd_finetune']['epoch']}_"
                              f"lr_{format(config['kdd_finetune']['lr'], '.10f').rstrip('0').rstrip('.')}_"
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

    for epoch in range(1, config["kdd_finetune"]["epoch"] + 1):
        epoch_start_time = time.time()

        train_loss, val_loss, val_acc, val_f1 = finetune_kdd_epoch(model, loss, optimizer, train_set, val_set, config, device, epoch, l2_reg=False)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)
        val_f1_score_list.append(val_f1)

        epoch_runtime = time.time() - epoch_start_time

        print(f"Epoch {epoch}/{config['kdd_finetune']['epoch']}, Train Loss: {train_loss:.4f}, Time: {epoch_runtime}")


    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["general"]["pretrain_model"], "continue_model.pth")
    )

    # kdd_finetune_save_metrics(train_loss_list, val_loss_list, val_accuracy_list, val_f1_score_list, config)


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

    for i, batch in enumerate(val_set):
        X, targets, padding_masks, IDs = batch
        targets = targets.to(device)
        # padding_masks = padding_masks.to(device)
        predictions = model(X.to(device))

        compute_loss = loss(predictions, targets)
        batch_loss = torch.sum(compute_loss).cpu().item()
        mean_loss = batch_loss / len(compute_loss)

        total_val_samples += len(compute_loss)
        val_loss += batch_loss

    val_loss = val_loss / total_val_samples