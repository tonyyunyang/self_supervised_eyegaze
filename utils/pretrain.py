import os.path
import sys
import time

from tqdm import tqdm

import torch

from modules.loss import l2_reg_loss

from matplotlib import pyplot as plt

import numpy as np


def pretrain_kdd_model(model, loss, optimizer, pretrain_data, config):
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

    path = os.path.join(path, f"pretrain")
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

    for epoch in range(1, config["kdd_pretrain"]["epoch"] + 1):
        epoch_start_time = time.time()

        train_loss = pretrain_kdd_epoch(model, loss, optimizer, pretrain_data, config, device, epoch, l2_reg=False)
        train_loss_list.append(train_loss)

        epoch_runtime = time.time() - epoch_start_time

        print(f"Epoch {epoch}/{config['kdd_pretrain']['epoch']}, Train Loss: {train_loss:.4f}, Time: {epoch_runtime}")

        # Difficulty scheduling
        if config["kdd_pretrain"]['harden'] and check_progress(epoch):
            print("==========================Updating difficulty=========================")
            pretrain_data.dataset.update()

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["general"]["pretrain_model"], "continue_model.pth")
    )

    kdd_pretrain_save_metrics(train_loss_list, config)


def pretrain_kdd_epoch(model, loss, optimizer, pretrain_data, config, device, epoch, l2_reg=False):
    model = model.train()
    train_loss = 0
    active_elements = 0
    for i, batch in enumerate(pretrain_data):
        X, targets, target_masks, padding_masks, IDs = batch
        targets = targets.to(device)
        target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
        padding_masks = padding_masks.to(device)  # 0s: ignore

        predictions = model(X.to(device))  # (batch_size, padded_length, feat_dim)

        # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
        target_masks = target_masks * padding_masks.unsqueeze(-1)

        compute_loss = loss(predictions, targets,
                            target_masks)  # (num_active,) individual loss (square error per element) for each active value in batch
        batch_loss = torch.sum(compute_loss)
        mean_loss = batch_loss / len(compute_loss)  # mean loss (over active elements) used for optimization
        # print(f"The mean-loss is {mean_loss}, The batch-loss is {batch_loss}, the number of examples is {len(compute_loss)}")

        if l2_reg:
            total_loss = mean_loss + config["kdd_pretrain"]["weight_decay"] * l2_reg_loss(model)
        else:
            total_loss = mean_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        total_loss.backward()

        # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()

        with torch.no_grad():
            active_elements += len(compute_loss)
            train_loss += batch_loss.item()  # add total loss of batch

        if epoch == config["kdd_pretrain"]["epoch"]:
            np.savetxt(f"{config['general']['pretrain_model']}/pred.txt", predictions.cpu().detach().numpy().reshape(-1))
            np.savetxt(f"{config['general']['pretrain_model']}/true.txt", targets.cpu().detach().numpy().reshape(-1))
            np.savetxt(f"{config['general']['pretrain_model']}/mask.txt", target_masks.cpu().detach().numpy().astype(int).reshape(-1))

    train_loss = train_loss / active_elements  # average loss per element for whole epoch
    return train_loss


def pretrain_kdd_model_dual_loss(model, loss, optimizer, pretrain_data, config):
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

    path = os.path.join(path, f"pretrain")
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

    for epoch in range(1, config["kdd_pretrain"]["epoch"] + 1):
        epoch_start_time = time.time()

        train_loss = pretrain_kdd_epoch_dual_loss(model, loss, optimizer, pretrain_data, config, device, epoch, l2_reg=False)
        train_loss_list.append(train_loss)

        epoch_runtime = time.time() - epoch_start_time

        print(f"Epoch {epoch}/{config['kdd_pretrain']['epoch']}, Train Loss: {train_loss:.4f}, Time: {epoch_runtime}")

        # Difficulty scheduling
        if config["kdd_pretrain"]['harden'] and check_progress(epoch):
            print("==========================Updating difficulty=========================")
            pretrain_data.dataset.update()

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["general"]["pretrain_model"], "continue_model.pth")
    )

    kdd_pretrain_save_metrics(train_loss_list, config)


def pretrain_kdd_epoch_dual_loss(model, loss, optimizer, pretrain_data, config, device, epoch, l2_reg=False):
    model = model.train()
    train_loss = 0
    active_elements = 0
    weight_reconstruction = 0.5  # Weight for the reconstruction loss
    weight_encoder_pass = 1 - weight_reconstruction  # Weight for the encoder pass loss
    for i, batch in enumerate(pretrain_data):
        X, UnmaskX, targets, target_masks, padding_masks, IDs = batch
        targets = targets.to(device)
        target_masks = target_masks.to(device)  # 1s: mask and predict, 0s: unaffected input (ignore)
        padding_masks = padding_masks.to(device)  # 0s: ignore

        reconstruction, reconstruction_pass_encoder, original_pass_encoder = model(X.to(device), UnmaskX.to(device))  # (batch_size, padded_length, feat_dim)

        # Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
        target_masks = target_masks * padding_masks.unsqueeze(-1)

        # Loss between reconstruction and targets
        compute_loss_reconstruction = loss(reconstruction, targets, target_masks)

        # Create a mask of True values for the encoder pass loss
        encoder_pass_mask = torch.ones_like(reconstruction_pass_encoder, dtype=torch.bool).to(device)

        # Loss between reconstruction_pass_encoder and original_pass_encoder
        compute_loss_encoder_pass = loss(reconstruction_pass_encoder, original_pass_encoder, encoder_pass_mask)

        # Compute the mean of each individual loss tensor
        batch_loss_reconstruction = torch.sum(compute_loss_reconstruction)
        batch_loss_encoder_pass = torch.sum(compute_loss_encoder_pass)
        mean_loss_reconstruction = batch_loss_reconstruction / len(compute_loss_reconstruction)
        mean_loss_encoder_pass = batch_loss_encoder_pass / len(compute_loss_encoder_pass)

        # mean_loss_reconstruction = torch.mean(compute_loss_reconstruction)
        # mean_loss_encoder_pass = torch.mean(compute_loss_encoder_pass)

        # Combine the mean losses using the given weights
        combined_loss = weight_reconstruction * mean_loss_reconstruction + weight_encoder_pass * mean_loss_encoder_pass

        mean_loss = combined_loss

        if l2_reg:
            total_loss = mean_loss + config["kdd_pretrain"]["weight_decay"] * l2_reg_loss(model)
        else:
            total_loss = mean_loss

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        total_loss.backward()

        # torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()

        with torch.no_grad():
            active_elements += len(compute_loss_reconstruction) + len(compute_loss_encoder_pass)
            train_loss += batch_loss_reconstruction.item() + batch_loss_encoder_pass.item()  # add total loss of batch

        if epoch == config["kdd_pretrain"]["epoch"]:
            np.savetxt(f"{config['general']['pretrain_model']}/pred.txt", reconstruction.cpu().detach().numpy().reshape(-1))
            np.savetxt(f"{config['general']['pretrain_model']}/true.txt", targets.cpu().detach().numpy().reshape(-1))
            np.savetxt(f"{config['general']['pretrain_model']}/mask.txt", target_masks.cpu().detach().numpy().astype(int).reshape(-1))

    train_loss = train_loss / active_elements  # average loss per element for whole epoch
    return train_loss


def check_progress(epoch):
    numbers = generate_numbers(0, 700, 5)
    if epoch in numbers:
        return True
    else:
        return False


# if epoch in [100, 140, 160, 220, 280, 340]: #[100, 140, 160, 220, 280, 340]
#     return True
# else:
#     return False


def kdd_pretrain_save_metrics(train_loss_list, config):
    # After the training loop, plot the loss list
    numbers = generate_numbers(0, 700, 5)

    plt.figure(figsize=(10, 5))
    # Exclude the first 50 values and generate x-axis values starting from certain value i
    i = 50
    x_values = np.arange(i, len(train_loss_list))
    plt.plot(x_values, train_loss_list[i:], label="Training Loss")

    # Highlight the points
    if config["kdd_pretrain"]["harden"]:
        for num in numbers:
            if num < len(train_loss_list):  # Check if the number is within the range of the list
                plt.scatter(num, train_loss_list[num], color='red')  # Highlight the point on the training loss plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(
        os.path.join(config["general"]["pretrain_model"], "loss_plot.png")
    )  # Save the figure before showing it


def generate_numbers(start, end, total_steps):
    step_size = (end - start) / (total_steps - 1)
    return [round(start + step_size * i) for i in range(1, total_steps + 1)]


def pretrain_limu_model(model, loss, optimizer, pretrain_data, config):
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

    path = os.path.join(path, f"pretrain")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"window_size_{int(config['general']['window_size'] / 30)}sec")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"feat_dim_{int(config['general']['feat_dim'])}")
    os.makedirs(path, exist_ok=True)

    path = os.path.join(path, f"epoch_{config['limu_pretrain']['epoch']}_"
                              f"lr_{format(config['limu_pretrain']['lr'], '.10f').rstrip('0').rstrip('.')}_"
                              f"d_hidden_{config['limu_model']['d_hidden']}_d_ff_{config['limu_model']['d_ff']}_"
                              f"n_heads_{config['limu_model']['n_heads']}_n_layer_{config['limu_model']['n_layers']}_"
                              f"embNorm_{config['limu_model']['emb_norm']}")
    os.makedirs(path, exist_ok=True)

    config["general"]["pretrain_model"] = path

    model = model.to(device)

    train_loss_list = []

    for epoch in range(1, config["limu_pretrain"]["epoch"] + 1):
        epoch_start_time = time.time()

        train_loss = pretrain_limu_epoch(model, loss, optimizer, pretrain_data, config, device, epoch, l2_reg=False)
        train_loss_list.append(train_loss)

        epoch_runtime = time.time() - epoch_start_time

        print(f"Epoch {epoch}/{config['limu_pretrain']['epoch']}, Train Loss: {train_loss:.4f}, Time: {epoch_runtime}")

    # Save the model for continuing the training
    torch.save(
        model.state_dict(), os.path.join(config["general"]["pretrain_model"], "continue_model.pth")
    )

    limu_pretrain_save_metrics(train_loss_list, config)


def pretrain_limu_epoch(model, loss_fn, optimizer, pretrain_data, config, device, epoch, l2_reg=False):
    loss_sum = 0.0
    model = model.train()
    for i, batch in enumerate(pretrain_data):
        mask_seqs, masked_pos, seqs, origin_seq = batch
        seqs = seqs.to(device)

        optimizer.zero_grad()
        seq_recon = model(mask_seqs.to(device), masked_pos.to(device))
        loss = loss_fn(seq_recon, seqs)

        loss = loss.mean()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if epoch == config["limu_pretrain"]["epoch"]:
            # Reshape the tensors to 2D and save
            np.savetxt(f"{config['general']['pretrain_model']}/pred.txt",
                       seq_recon.cpu().detach().numpy().reshape(-1, seq_recon.shape[-1]))
            np.savetxt(f"{config['general']['pretrain_model']}/true.txt",
                       seqs.cpu().detach().numpy().reshape(-1, seqs.shape[-1]))
            np.savetxt(f"{config['general']['pretrain_model']}/origin_seq.txt",
                       origin_seq.cpu().detach().numpy().reshape(-1, origin_seq.shape[-1]))
            np.savetxt(f"{config['general']['pretrain_model']}/mask_pos.txt",
                       masked_pos.cpu().detach().numpy().reshape(-1, masked_pos.shape[-1]))
            np.savetxt(f"{config['general']['pretrain_model']}/mask_seq.txt",
                       mask_seqs.cpu().detach().numpy().reshape(-1, mask_seqs.shape[-1]))
            

    training_loss = loss_sum / len(pretrain_data)
    return training_loss


def limu_pretrain_save_metrics(train_loss_list, config):
    # After the training loop, plot the loss list
    numbers = generate_numbers(0, 700, 5)

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_list, label="Training Loss")

    # Highlight the points
    if config["limu_pretrain"]["harden"]:
        for num in numbers:
            if num < len(train_loss_list):  # Check if the number is within the range of the list
                plt.scatter(num, train_loss_list[num], color='red')  # Highlight the point on the training loss plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(
        os.path.join(config["general"]["pretrain_model"], "loss_plot.png")
    )  # Save the figure before showing it
