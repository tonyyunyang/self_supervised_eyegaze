import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def limu_model4pretrain(config, feat_dim):
    model = LIMUBertModel4Pretrain(config, feat_dim)

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def limu_model4finetune(config, feat_dim, classifier, frozen_bert):
    model = LIMUBertModel4Finetune(config, feat_dim, classifier=classifier, frozen_bert=frozen_bert)

    print("Model:\n{}".format(model))
    print("Total number of parameters: {}".format(count_parameters(model)))
    print("Trainable parameters: {}".format(count_parameters(model, trainable=True)))

    return model


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = x.size(-1) // -np.prod(shape)
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class LayerNorm(nn.Module):
    "A layernorm module in the TF style (epsilon inside the square root)."

    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg["limu_model"]["d_hidden"]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(cfg["limu_model"]["d_hidden"]), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        # return x


class Embeddings(nn.Module):

    def __init__(self, cfg, feat_dim, pos_embed=None):
        super().__init__()

        # factorized embedding
        self.lin = nn.Linear(feat_dim, cfg["limu_model"]["d_hidden"])
        if pos_embed is None:
            self.pos_embed = nn.Embedding(cfg["general"]["window_size"],
                                          cfg["limu_model"]["d_hidden"])  # position embedding
        else:
            self.pos_embed = pos_embed

        self.norm = LayerNorm(cfg)
        self.emb_norm = cfg["limu_model"]["emb_norm"]

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand(x.size(0), seq_len)  # (S,) -> (B, S)

        # factorized embedding
        e = self.lin(x)
        if self.emb_norm:
            e = self.norm(e)
        e = e + self.pos_embed(pos)
        return self.norm(e)


class MultiProjection(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        return q, k, v


class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """

    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg["limu_model"]["d_hidden"], cfg["limu_model"]["d_hidden"])
        self.proj_k = nn.Linear(cfg["limu_model"]["d_hidden"], cfg["limu_model"]["d_hidden"])
        self.proj_v = nn.Linear(cfg["limu_model"]["d_hidden"], cfg["limu_model"]["d_hidden"])
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = cfg["limu_model"]["n_heads"]

    def forward(self, x):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        # scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """

    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["limu_model"]["d_hidden"], cfg["limu_model"]["d_ff"])
        self.fc2 = nn.Linear(cfg["limu_model"]["d_ff"], cfg["limu_model"]["d_hidden"])
        # self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Transformer_Original(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.embed = Embeddings(cfg, feat_dim)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg["limu_model"]["n_layers"]
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg["limu_model"]["d_hidden"], cfg["limu_model"]["d_hidden"])
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        h = self.embed(x)

        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class Transformer_Corrected(nn.Module):
    """ Transformer with Self-Attentive Blocks"""

    def __init__(self, cfg, feat_dim):
        super().__init__()
        self.embed = Embeddings(cfg, feat_dim)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # To used parameter-sharing strategies
        self.n_layers = cfg["limu_model"]["n_layers"]
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg["limu_model"]["d_hidden"], cfg["limu_model"]["d_hidden"])
        self.norm = LayerNorm(cfg)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x):
        h = self.embed(x)

        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.norm(self.attn(h) + h)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class LIMUBertModel4Pretrain(nn.Module):
    def __init__(self, cfg, feat_dim, output_embed=False):
        super().__init__()
        self.transformer = Transformer_Original(cfg, feat_dim)  # encoder
        self.fc = nn.Linear(cfg["limu_model"]["d_hidden"], cfg["limu_model"]["d_hidden"])
        self.linear = nn.Linear(cfg["limu_model"]["d_hidden"], cfg["limu_model"]["d_hidden"])
        self.activ = gelu
        self.norm = LayerNorm(cfg)
        self.decoder = nn.Linear(cfg["limu_model"]["d_hidden"], feat_dim)
        self.output_embed = output_embed

    def forward(self, input_seqs, masked_pos=None):

        h_masked = self.transformer(input_seqs)
        if self.output_embed:
            return h_masked
        if masked_pos is not None:
            masked_pos = masked_pos[:, :, None].expand(-1, -1, h_masked.size(-1))
            h_masked = torch.gather(h_masked, 1, masked_pos)
        h_masked = self.activ(self.linear(h_masked))
        h_masked = self.norm(h_masked)
        logits_lm = self.decoder(h_masked)
        return logits_lm


class LIMUBertModel4Finetune(nn.Module):
    def __init__(self, cfg, feat_dim, classifier=None, frozen_bert=False):
        super().__init__()
        self.transformer = Transformer_Original(cfg, feat_dim)  # encoder
        if frozen_bert:
            for p in self.transformer.parameters():
                p.requires_grad = False
        self.classifier = classifier

    def forward(self, input_seqs, training=False):  # training
        h = self.transformer(input_seqs)
        h = self.classifier(h, training)
        return h

    def load_self(self, model_file, map_location=None):
        state_dict = self.state_dict()
        model_dicts = torch.load(model_file, map_location=map_location).items()
        for k, v in model_dicts:
            if k in state_dict:
                state_dict.update({k: v})
        self.load_state_dict(state_dict)
