import os
import sys
from copy import deepcopy
from typing import Optional, Any
import math

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from numpy import floor


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def kdd_model4pretrain(config, feat_dim):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoder(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_10sec"],
    )

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4pretrain_convstack(config, feat_dim):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderStack(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_10sec_stack"],
    )

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4pretrain_test(config, feat_dim):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderTest(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_20sec"],
    )

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4pretrain_stack_test(config, feat_dim):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderStackTest(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_10sec_stack"],
    )

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4pretrain_dual_loss(config, feat_dim):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderDualLoss(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
    )

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4finetune(config, feat_dim, num_classes):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderClassiregressor(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        num_classes,
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_10sec"]
    )

    model_path = os.path.join(config["general"]["pretrain_model"], "continue_model.pth")

    state_dict = torch.load(model_path)

    for key in list(
            state_dict.keys()):  # need to convert keys to list to avoid RuntimeError due to changing size during iteration
        if key.startswith('output_layer'):
            state_dict.pop(key)
            print(f"Popped layer {key}")
    model.load_state_dict(state_dict, strict=False)

    print('Loaded model from {}'.format(model_path))

    if config["general"]["freeze"]:
        for name, param in model.named_parameters():
            print(f"{name}")
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4finetune_convstack(config, feat_dim, num_classes):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderClassiregressorStack(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        num_classes,
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_10sec_stack"]
    )

    model_path = os.path.join(config["general"]["pretrain_model"], "continue_model.pth")

    state_dict = torch.load(model_path)

    for key in list(
            state_dict.keys()):  # need to convert keys to list to avoid RuntimeError due to changing size during iteration
        if key.startswith('output_layer'):
            state_dict.pop(key)
            print(f"Popped layer {key}")
    model.load_state_dict(state_dict, strict=False)

    print('Loaded model from {}'.format(model_path))

    if config["general"]["freeze"]:
        for name, param in model.named_parameters():
            print(f"{name}")
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4finetune_test(config, feat_dim, num_classes):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderClassiregressorTest(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        num_classes,
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_20sec"],
    )

    model_path = os.path.join(config["general"]["pretrain_model"], "best_model.pth")

    state_dict = torch.load(model_path)

    for key in list(
            state_dict.keys()):  # need to convert keys to list to avoid RuntimeError due to changing size during iteration
        if key.startswith('output_layer'):
            state_dict.pop(key)
            print(f"Popped layer {key}")
    model.load_state_dict(state_dict, strict=False)

    print('Loaded model from {}'.format(model_path))

    if config["general"]["freeze"]:
        for name, param in model.named_parameters():
            print(f"{name}")
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4finetune_stack_test(config, feat_dim, num_classes):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderClassiregressorStackTest(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        num_classes,
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_10sec_stack"],
    )

    model_path = os.path.join(config["general"]["pretrain_model"], "continue_model.pth")

    state_dict = torch.load(model_path)

    for key in list(
            state_dict.keys()):  # need to convert keys to list to avoid RuntimeError due to changing size during iteration
        if key.startswith('output_layer'):
            state_dict.pop(key)
            print(f"Popped layer {key}")
    model.load_state_dict(state_dict, strict=False)

    print('Loaded model from {}'.format(model_path))

    if config["general"]["freeze"]:
        for name, param in model.named_parameters():
            print(f"{name}")
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4fullysupervise_pretrain(config, feat_dim, num_classes):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderClassiregressorTest(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        num_classes,
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_5sec"],
    )

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4fullysupervise_finetune(config, feat_dim, num_classes):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderClassiregressorTest(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        num_classes,
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_5sec"],
    )
    
    model_path = os.path.join(config["general"]["pretrain_model"], "best_model.pth")

    state_dict = torch.load(model_path)
    
    model.load_state_dict(state_dict, strict=False)

    print('Loaded model from {}'.format(model_path))

    if config["general"]["freeze"]:
        for name, param in model.named_parameters():
            print(f"{name}")
            if name.startswith('output_layer'):
                param.requires_grad = True
            else:
                param.requires_grad = False

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def kdd_model4fullysupervise_random_initialize(config, feat_dim, num_classes):
    max_seq_len = config["general"]["window_size"]

    model = TSTransformerEncoderClassiregressorTest(
        feat_dim,
        max_seq_len,
        config["kdd_model"]["d_hidden"],
        config["kdd_model"]["n_heads"],
        config["kdd_model"]["n_layers"],
        config["kdd_model"]["d_ff"],
        num_classes,
        dropout=config["kdd_model"]["dropout"],
        pos_encoding=config["kdd_model"]["pos_encoding"],
        activation=config["kdd_model"]["activation"],
        norm=config["kdd_model"]["norm"],
        embedding=config["kdd_model"]["projection"],
        freeze=config["general"]["freeze"],
        conv_config=config["conv1d_10sec"],
    )

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: optional

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src


class TSTransformerEncoder(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', embedding='convolution', freeze=False, conv_config=None):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        # Below are configurations for the convolution layer
        kernel_size = conv_config["first"]["kernel_size"]
        stride = conv_config["first"]["stride"]
        dilation = conv_config["first"]["dilation"]
        padding = conv_config["first"]["padding"]

        if self.embedding == "linear":
            self.project_inp = nn.Linear(feat_dim, d_model)
        elif self.embedding == "convolution":
            # Calculate the output sequence size after the 1D Conv layer
            proj_conv_seq_length = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

            self.project_inp = nn.Conv1d(feat_dim, d_model, kernel_size, stride, padding, dilation)
            self.max_len = proj_conv_seq_length
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        if self.embedding == "linear":
            self.output_layer = nn.Linear(d_model, feat_dim)
        elif self.embedding == "convolution":
            # The second to the last 0 is output_padding, we always set it to 0
            recon_conv_seq_length = (proj_conv_seq_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 0 + 1
            # The build_out_layer stacks linear after the convTranspose1D to ensure the squence length is the same as the original
            self.output_layer = self.build_output_layer(recon_conv_seq_length, conv_config, d_model, feat_dim, max_len)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        
    def build_output_layer(self, recon_conv_seq_length, conv_config, d_model, feat_dim, max_len):
        class ConvTransposeLinear(nn.Module):
            def __init__(self):
                super(ConvTransposeLinear, self).__init__()
                kernel_size = conv_config["first"]["kernel_size"]
                stride = conv_config["first"]["stride"]
                padding = conv_config["first"]["padding"]
                dilation = conv_config["first"]["dilation"]

                # ConvTranspose1d layer
                self.conv_transpose = nn.ConvTranspose1d(d_model, feat_dim, kernel_size=kernel_size, stride=stride,
                                                        padding=padding, dilation=dilation)
                
                # Linear layer
                self.linear = nn.Linear(recon_conv_seq_length * feat_dim, max_len * feat_dim)

            def forward(self, x):
                x = self.conv_transpose(x)
                x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
                x = self.linear(x)
                return x.view(-1, feat_dim, max_len)  # Reshape to the original sequence shape

        return ConvTransposeLinear()

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        if self.embedding == "linear":
            # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
            inp = X.permute(1, 0, 2)
            # inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
            inp = self.project_inp(inp)
        elif self.embedding == "convolution":
            inp = X.permute(0, 2, 1)  # permute to (batch_size, feat_dim, seq_length)
            inp = self.project_inp(inp)
            inp = inp.permute(2, 0, 1)  # permute back to (seq_length, batch_size, d_model)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        # inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).

        if self.embedding == "linear":
            output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        elif self.embedding == "convolution":
            # Change this line to permute the output to the right shape before passing it to the output layer
            output = output.permute(0, 2, 1)  # (batch_size, d_model, seq_length)
            output = self.output_layer(output)  # (batch_size, feat_dim, seq_length)
            # Permute the output back to the original shape
            output = output.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        return output
    
    
class TSTransformerEncoderStack(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', embedding='convolution', freeze=False, conv_config=None):
        super(TSTransformerEncoderStack, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        if self.embedding == "linear":
            self.project_inp = nn.Linear(feat_dim, d_model)
        elif self.embedding == "convolution":
            self.project_inp = self.build_stack_conv(conv_config, feat_dim, d_model)
        else:
            raise ValueError("Embedding must be either 'linear' or 'convolution'")

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        if self.embedding == "linear":
            self.output_layer = nn.Linear(d_model, feat_dim)
        elif self.embedding == "convolution":
            self.output_layer = self.build_output_layer(self.max_len, conv_config, d_model, feat_dim, max_len)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        
    def build_stack_conv(self, conv_config, in_channels, out_channels):
        layers = []

        for layer_name, config in conv_config.items():
            kernel_size = config["kernel_size"]
            stride = config["stride"]
            dilation = config["dilation"]
            padding = config["padding"]

            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
            layers.append(conv_layer)

            # Update in_channels for the next layer
            in_channels = out_channels

            # Update self.max_len for the next layer
            self.max_len = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
            print(f"The {layer_name} produces sequence length of {self.max_len}")

        return nn.Sequential(*layers)
        
    def build_output_layer(self, recon_conv_seq_length, conv_config, d_model, feat_dim, max_len):
        class ConvTransposeLinear(nn.Module):
            def __init__(self, d_model, feat_dim, recon_conv_seq_length, conv_config, max_len):
                super(ConvTransposeLinear, self).__init__()
                conv_transpose_layers = []

                # Iterate over the conv_config in reverse order
                for layer_name, config in reversed(conv_config.items()):
                    kernel_size = config["kernel_size"]
                    stride = config["stride"]
                    dilation = config["dilation"]
                    padding = config["padding"]

                    # ConvTranspose1d layer with reversed configurations
                    conv_transpose = nn.ConvTranspose1d(d_model, feat_dim, kernel_size=kernel_size, stride=stride,
                                                        padding=padding, dilation=dilation)
                    conv_transpose_layers.append(conv_transpose)

                    # Update the dimensions for recon_conv_seq_length for subsequent layers
                    recon_conv_seq_length = (recon_conv_seq_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

                    # Update the d_model to feat_dim after the first transposed conv layer
                    d_model = feat_dim
                    
                    print(f"The {layer_name} produces sequence length of {recon_conv_seq_length}")

                self.conv_transpose = nn.Sequential(*conv_transpose_layers)
                
                # Linear layer to adjust the final shape
                self.linear = nn.Linear(recon_conv_seq_length * feat_dim, max_len * feat_dim)

            def forward(self, x):
                x = self.conv_transpose(x)
                x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
                x = self.linear(x)
                return x.view(-1, feat_dim, max_len)  # Reshape to the original sequence shape

        return ConvTransposeLinear(d_model, feat_dim, recon_conv_seq_length, conv_config, max_len)

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        if self.embedding == "linear":
            # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
            inp = X.permute(1, 0, 2)
            # inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
            inp = self.project_inp(inp)
        elif self.embedding == "convolution":
            inp = X.permute(0, 2, 1)  # permute to (batch_size, feat_dim, seq_length)
            inp = self.project_inp(inp)
            inp = inp.permute(2, 0, 1)  # permute back to (seq_length, batch_size, d_model)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        # inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).

        if self.embedding == "linear":
            output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        elif self.embedding == "convolution":
            # Change this line to permute the output to the right shape before passing it to the output layer
            output = output.permute(0, 2, 1)  # (batch_size, d_model, seq_length)
            output = self.output_layer(output)  # (batch_size, feat_dim, seq_length)
            # Permute the output back to the original shape
            output = output.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        return output
    

class TSTransformerEncoderTest(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', embedding='convolution', freeze=False, conv_config=None):
        super(TSTransformerEncoderTest, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        # Below are configurations for the convolution layer
        kernel_size = conv_config["first"]["kernel_size"]
        stride = conv_config["first"]["stride"]
        dilation = conv_config["first"]["dilation"]
        padding = conv_config["first"]["padding"]

        if self.embedding == "linear":
            self.project_inp = nn.Linear(feat_dim, d_model)
        elif self.embedding == "convolution":
            # Calculate the output sequence size after the 1D Conv layer
            proj_conv_seq_length = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

            self.project_inp = nn.Conv1d(feat_dim, d_model, kernel_size, stride, padding, dilation)
            self.max_len = proj_conv_seq_length
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        if self.embedding == "linear":
            self.output_layer = nn.Linear(d_model, feat_dim)
        elif self.embedding == "convolution":
            # The second to the last 0 is output_padding, we always set it to 0
            recon_conv_seq_length = (proj_conv_seq_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 0 + 1
            # The build_out_layer stacks linear after the convTranspose1D to ensure the squence length is the same as the original
            self.output_layer = self.build_output_layer(recon_conv_seq_length, conv_config, d_model, feat_dim, max_len)
            # self.output_layer = nn.ConvTranspose1d(d_model, feat_dim, kernel_size=kernel_size, stride=stride,
            #                                        padding=padding, dilation=dilation)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        
    def build_output_layer(self, recon_conv_seq_length, conv_config, d_model, feat_dim, max_len):
        class ConvTransposeLinear(nn.Module):
            def __init__(self):
                super(ConvTransposeLinear, self).__init__()
                kernel_size = conv_config["first"]["kernel_size"]
                stride = conv_config["first"]["stride"]
                padding = conv_config["first"]["padding"]
                dilation = conv_config["first"]["dilation"]

                # ConvTranspose1d layer
                self.conv_transpose = nn.ConvTranspose1d(d_model, feat_dim, kernel_size=kernel_size, stride=stride,
                                                        padding=padding, dilation=dilation)
                
                # Linear layer
                self.linear = nn.Linear(recon_conv_seq_length * feat_dim, max_len * feat_dim)

            def forward(self, x):
                x = self.conv_transpose(x)
                x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
                x = self.linear(x)
                return x.view(-1, feat_dim, max_len)  # Reshape to the original sequence shape

        return ConvTransposeLinear()

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        if self.embedding == "linear":
            # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
            inp = X.permute(1, 0, 2)
            # inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
            inp = self.project_inp(inp)
        elif self.embedding == "convolution":
            inp = X.permute(0, 2, 1)  # permute to (batch_size, feat_dim, seq_length)
            # inp = self.project_inp(inp)
            inp = self.project_inp(inp) * math.sqrt(self.d_model)
            inp = inp.permute(2, 0, 1)  # permute back to (seq_length, batch_size, d_model)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).

        if self.embedding == "linear":
            output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        elif self.embedding == "convolution":
            # Change this line to permute the output to the right shape before passing it to the output layer
            output = output.permute(0, 2, 1)  # (batch_size, d_model, seq_length)
            output = self.output_layer(output)  # (batch_size, feat_dim, seq_length)
            # Permute the output back to the original shape
            output = output.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        return output
    
    
class TSTransformerEncoderStackTest(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', embedding='convolution', freeze=False, conv_config=None):
        super(TSTransformerEncoderStackTest, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        # Below are configurations for the convolution layer
        # kernel_size = conv_config["first"]["kernel_size"]
        # stride = conv_config["first"]["stride"]
        # dilation = conv_config["first"]["dilation"]
        # padding = conv_config["first"]["padding"]

        if self.embedding == "linear":
            self.project_inp = nn.Linear(feat_dim, d_model)
        elif self.embedding == "convolution":
            # Calculate the output sequence size after the 1D Conv layer
            # proj_conv_seq_length = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

            # self.project_inp = nn.Conv1d(feat_dim, d_model, kernel_size, stride, padding, dilation)
            # self.max_len = proj_conv_seq_length
            self.project_inp = self.build_stack_conv(conv_config, feat_dim, d_model)
        else:
            raise ValueError("Embedding must be either 'linear' or 'convolution'")

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        if self.embedding == "linear":
            self.output_layer = nn.Linear(d_model, feat_dim)
        elif self.embedding == "convolution":
            # The second to the last 0 is output_padding, we always set it to 0
            # recon_conv_seq_length = (proj_conv_seq_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 0 + 1
            # The build_out_layer stacks linear after the convTranspose1D to ensure the squence length is the same as the original
            self.output_layer = self.build_output_layer(self.max_len, conv_config, d_model, feat_dim, max_len)
            # self.output_layer = nn.ConvTranspose1d(d_model, feat_dim, kernel_size=kernel_size, stride=stride,
            #                                        padding=padding, dilation=dilation)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        
    def build_stack_conv(self, conv_config, in_channels, out_channels):
        layers = []

        for layer_name, config in conv_config.items():
            kernel_size = config["kernel_size"]
            stride = config["stride"]
            dilation = config["dilation"]
            padding = config["padding"]

            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
            layers.append(conv_layer)

            # Update in_channels for the next layer
            in_channels = out_channels

            # Update self.max_len for the next layer
            self.max_len = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
            print(f"The {layer_name} produces sequence length of {self.max_len}")

        return nn.Sequential(*layers)
        
    def build_output_layer(self, recon_conv_seq_length, conv_config, d_model, feat_dim, max_len):
        class ConvTransposeLinear(nn.Module):
            def __init__(self, d_model, feat_dim, recon_conv_seq_length, conv_config, max_len):
                super(ConvTransposeLinear, self).__init__()
                conv_transpose_layers = []

                # Iterate over the conv_config in reverse order
                for layer_name, config in reversed(conv_config.items()):
                    kernel_size = config["kernel_size"]
                    stride = config["stride"]
                    dilation = config["dilation"]
                    padding = config["padding"]

                    # ConvTranspose1d layer with reversed configurations
                    conv_transpose = nn.ConvTranspose1d(d_model, feat_dim, kernel_size=kernel_size, stride=stride,
                                                        padding=padding, dilation=dilation)
                    conv_transpose_layers.append(conv_transpose)

                    # Update the dimensions for recon_conv_seq_length for subsequent layers
                    recon_conv_seq_length = (recon_conv_seq_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1

                    # Update the d_model to feat_dim after the first transposed conv layer
                    d_model = feat_dim
                    
                    print(f"The {layer_name} produces sequence length of {recon_conv_seq_length}")

                self.conv_transpose = nn.Sequential(*conv_transpose_layers)
                
                # Linear layer to adjust the final shape
                self.linear = nn.Linear(recon_conv_seq_length * feat_dim, max_len * feat_dim)

            def forward(self, x):
                x = self.conv_transpose(x)
                x = x.view(x.size(0), -1)  # Flatten the output for the linear layer
                x = self.linear(x)
                return x.view(-1, feat_dim, max_len)  # Reshape to the original sequence shape

        return ConvTransposeLinear(d_model, feat_dim, recon_conv_seq_length, conv_config, max_len)

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """

        if self.embedding == "linear":
            # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
            inp = X.permute(1, 0, 2)
            # inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
            inp = self.project_inp(inp)
        elif self.embedding == "convolution":
            inp = X.permute(0, 2, 1)  # permute to (batch_size, feat_dim, seq_length)
            inp = self.project_inp(inp)
            inp = inp.permute(2, 0, 1)  # permute back to (seq_length, batch_size, d_model)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        # inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).

        if self.embedding == "linear":
            output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
        elif self.embedding == "convolution":
            # Change this line to permute the output to the right shape before passing it to the output layer
            output = output.permute(0, 2, 1)  # (batch_size, d_model, seq_length)
            output = self.output_layer(output)  # (batch_size, feat_dim, seq_length)
            # Permute the output back to the original shape
            output = output.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        return output


class TSTransformerEncoderDualLoss(nn.Module):

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', embedding='convolution', freeze=False):
        super(TSTransformerEncoderDualLoss, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        # Below are configurations for the convolution layer
        kernel_size = 10
        stride = 2
        dilation = 1
        padding = 0

        if self.embedding == "linear":
            self.project_inp = nn.Linear(feat_dim, d_model)
        elif self.embedding == "convolution":
            # Calculate the output sequence size after the 1D Conv layer
            conv_seq_length = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

            self.project_inp = nn.Conv1d(feat_dim, d_model, kernel_size, stride, padding, dilation)
            self.max_len = conv_seq_length
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        if self.embedding == "linear":
            self.output_layer = nn.Linear(d_model, feat_dim)
        elif self.embedding == "convolution":
            self.output_layer = nn.ConvTranspose1d(d_model, feat_dim, kernel_size=kernel_size, stride=stride,
                                                   padding=padding, dilation=dilation)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, UnmaskX):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        # the function below is for passing through the entire enocde process again
        def encoder_process(inp):
            if self.embedding == "linear":
                inp = inp.permute(1, 0, 2)
                inp = self.project_inp(inp) * math.sqrt(self.d_model)
            elif self.embedding == "convolution":
                inp = inp.permute(0, 2, 1)
                inp = self.project_inp(inp)
                inp = inp.permute(2, 0, 1)
            else:
                print(f"Either linear / convolution")
                sys.exit()

            inp = self.pos_enc(inp)
            out = self.transformer_encoder(inp)
            out = self.act(out)
            out = out.permute(1, 0, 2)
            out = self.dropout1(out)
            return out

        if self.embedding == "linear":
            # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
            inp = X.permute(1, 0, 2)
            inp = self.project_inp(inp) * math.sqrt(
                self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        elif self.embedding == "convolution":
            inp = X.permute(0, 2, 1)  # permute to (batch_size, feat_dim, seq_length)
            inp = self.project_inp(inp)
            inp = inp.permute(2, 0, 1)  # permute back to (seq_length, batch_size, d_model)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).

        if self.embedding == "linear":
            output = self.output_layer(output)  # (batch_size, seq_length, feat_dim)
            # output is the reconstruction which has the same size as the input X
        elif self.embedding == "convolution":
            # Change this line to permute the output to the right shape before passing it to the output layer
            output = output.permute(0, 2, 1)  # (batch_size, d_model, seq_length)
            output = self.output_layer(output)  # (batch_size, feat_dim, seq_length)
            # Permute the output back to the original shape
            output = output.permute(0, 2, 1)  # (batch_size, seq_length, feat_dim)
            # output is the reconstruction which has the same size as the input X
        else:
            print(f"Either linear / convolution")
            sys.exit()

        """
        output2: The result passing the reconstructed sequence through the encoder again.
        output3: The result passing the original unmask sequence through the encoder.
        """
        # print(f"The shape of output is {output.shape}")
        # Compute the second output by processing the original output
        output2 = encoder_process(output)
        # print(f"The shape of output2 is {output2.shape}")

        # Compute the third output by processing UnmaskX
        output3 = encoder_process(UnmaskX)
        # print(f"The shape of output3 is {output3.shape}")


        return output, output2, output3


class TSTransformerEncoderClassiregressor(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', embedding='convolution',
                 freeze=False, conv_config=None):
        super(TSTransformerEncoderClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        # Below are configurations for the convolution layer
        kernel_size = conv_config["first"]["kernel_size"]
        stride = conv_config["first"]["stride"]
        dilation = conv_config["first"]["dilation"]
        padding = conv_config["first"]["padding"]

        if self.embedding == "linear":
            self.project_inp = nn.Linear(feat_dim, d_model)
        elif self.embedding == "convolution":
            # Calculate the output sequence size after the 1D Conv layer
            conv_seq_length = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

            self.project_inp = nn.Conv1d(feat_dim, d_model, kernel_size, stride, padding, dilation)
            self.max_len = conv_seq_length
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, self.max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        if self.embedding == "linear":
            # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
            inp = X.permute(1, 0, 2)
            inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        elif self.embedding == "convolution":
            inp = X.permute(0, 2, 1)  # permute to (batch_size, feat_dim, seq_length)
            inp = self.project_inp(inp)
            inp = inp.permute(2, 0, 1)  # permute back to (seq_length, batch_size, d_model)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        # inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        # output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output
    
    
class TSTransformerEncoderClassiregressorStack(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', embedding='convolution',
                 freeze=False, conv_config=None):
        super(TSTransformerEncoderClassiregressorStack, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        if self.embedding == "linear":
            self.project_inp = nn.Linear(feat_dim, d_model)
        elif self.embedding == "convolution":
            self.project_inp = self.build_stack_conv(conv_config, feat_dim, d_model)
        else:
            raise ValueError("Embedding must be either 'linear' or 'convolution'")

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, self.max_len, num_classes)
        
    def build_stack_conv(self, conv_config, in_channels, out_channels):
        layers = []

        for layer_name, config in conv_config.items():
            kernel_size = config["kernel_size"]
            stride = config["stride"]
            dilation = config["dilation"]
            padding = config["padding"]

            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
            layers.append(conv_layer)

            # Update in_channels for the next layer
            in_channels = out_channels

            # Update self.max_len for the next layer
            self.max_len = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
            print(f"The {layer_name} produces sequence length of {self.max_len}")

        return nn.Sequential(*layers)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        if self.embedding == "linear":
            # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
            inp = X.permute(1, 0, 2)
            inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        elif self.embedding == "convolution":
            inp = X.permute(0, 2, 1)  # permute to (batch_size, feat_dim, seq_length)
            inp = self.project_inp(inp)
            inp = inp.permute(2, 0, 1)  # permute back to (seq_length, batch_size, d_model)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        # inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        # output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output


class TSTransformerEncoderClassiregressorTest(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', embedding='convolution',
                 freeze=False, conv_config=None):
        super(TSTransformerEncoderClassiregressorTest, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        # Below are configurations for the convolution layer
        kernel_size = conv_config["first"]["kernel_size"]
        stride = conv_config["first"]["stride"]
        dilation = conv_config["first"]["dilation"]
        padding = conv_config["first"]["padding"]

        if self.embedding == "linear":
            self.project_inp = nn.Linear(feat_dim, d_model)
        elif self.embedding == "convolution":
            # Calculate the output sequence size after the 1D Conv layer
            conv_seq_length = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

            self.project_inp = nn.Conv1d(feat_dim, d_model, kernel_size, stride, padding, dilation)
            self.max_len = conv_seq_length
        else:
            print(f"Either linear / convolution")
            sys.exit()

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, self.max_len, num_classes)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        if self.embedding == "linear":
            # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
            inp = X.permute(1, 0, 2)
            inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        elif self.embedding == "convolution":
            inp = X.permute(0, 2, 1)  # permute to (batch_size, feat_dim, seq_length)
            # inp = self.project_inp(inp)
            inp = self.project_inp(inp) * math.sqrt(self.d_model)
            inp = inp.permute(2, 0, 1)  # permute back to (seq_length, batch_size, d_model)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        # output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output
    
    
class TSTransformerEncoderClassiregressorStackTest(nn.Module):
    """
    Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
    softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
    """

    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, num_classes,
                 dropout=0.1, pos_encoding='fixed', activation='gelu', norm='BatchNorm', embedding='convolution',
                 freeze=False, conv_config=None):
        super(TSTransformerEncoderClassiregressorStackTest, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.embedding = embedding

        # Below are configurations for the convolution layer
        # kernel_size = conv_config["first"]["kernel_size"]
        # stride = conv_config["first"]["stride"]
        # dilation = conv_config["first"]["dilation"]
        # padding = conv_config["first"]["padding"]

        if self.embedding == "linear":
            self.project_inp = nn.Linear(feat_dim, d_model)
        elif self.embedding == "convolution":
            # Calculate the output sequence size after the 1D Conv layer
            # conv_seq_length = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))

            # self.project_inp = nn.Conv1d(feat_dim, d_model, kernel_size, stride, padding, dilation)
            # self.max_len = conv_seq_length
            self.project_inp = self.build_stack_conv(conv_config, feat_dim, d_model)
        else:
            raise ValueError("Embedding must be either 'linear' or 'convolution'")

        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=self.max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, self.max_len, num_classes)

    def build_stack_conv(self, conv_config, in_channels, out_channels):
        layers = []

        for layer_name, config in conv_config.items():
            kernel_size = config["kernel_size"]
            stride = config["stride"]
            dilation = config["dilation"]
            padding = config["padding"]

            conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation)
            layers.append(conv_layer)

            # Update in_channels for the next layer
            in_channels = out_channels

            # Update self.max_len for the next layer
            self.max_len = int(floor((self.max_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1))
            print(f"The {layer_name} produces sequence length of {self.max_len}")

        return nn.Sequential(*layers)

    def build_output_module(self, d_model, max_len, num_classes):
        output_layer = nn.Linear(d_model * max_len, num_classes)
        # no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
        # add F.log_softmax and use NLLoss
        return output_layer

    def forward(self, X):
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """

        if self.embedding == "linear":
            # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
            inp = X.permute(1, 0, 2)
            inp = self.project_inp(inp) * math.sqrt(self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        elif self.embedding == "convolution":
            inp = X.permute(0, 2, 1)  # permute to (batch_size, feat_dim, seq_length)
            inp = self.project_inp(inp)
            inp = inp.permute(2, 0, 1)  # permute back to (seq_length, batch_size, d_model)
        else:
            print(f"Either linear / convolution")
            sys.exit()

        # inp = self.pos_enc(inp)  # add positional encoding
        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(inp)  # (seq_length, batch_size, d_model)
        output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
        output = self.dropout1(output)

        # Output
        # output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)

        return output