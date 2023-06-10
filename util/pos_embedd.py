import numpy as np
import torch.nn as nn

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
import torch
from typing import Tuple
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim
        self.dim_fix = nn.Linear(2 * embed_dim, embed_dim)

    @staticmethod
    def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, grid_size, grid_size])
        pos_embed = PositionalEncoding.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token:
            pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
        return pos_embed

    @staticmethod
    def encode_relative_position(grid, pos, grid_size):
        # i1, j1, h1, w1, i2, j2, h2, w2
        # h2 / h1
        batch_size = pos.shape[0]

        grid = np.expand_dims(grid, axis=0)

        h = pos[:, 6] / pos[:, 2]
        # w2 / w1
        w = pos[:, 7] / pos[:, 3]
        h = h.reshape(batch_size, 1, 1, 1)
        w = w.reshape(batch_size, 1, 1, 1)

        # Nh * (i2 - i1) / h1
        b_h = grid_size * (pos[:, 4] - pos[:, 0]) / pos[:, 2]
        # Nw * (j2 - j1) / w1
        b_w = grid_size * (pos[:, 5] - pos[:, 1]) / pos[:, 3]
        b_h = b_h.reshape(batch_size, 1, 1, 1)
        b_w = b_w.reshape(batch_size, 1, 1, 1)

        rp_h = np.multiply(grid[:, 0], h) + b_h
        rp_w = np.multiply(grid[:, 1], w) + b_w

        rp = np.concatenate([rp_h, rp_w], axis=1)

        return rp

    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0

        batch_size, _, h, w = grid.shape

        # use half of dimensions to encode grid_h
        emb_h = PositionalEncoding.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 0])  # (N, H*W, D/2)
        emb_w = PositionalEncoding.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 1])  # (N, H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1) # (N, H*W, D)
        emb = np.reshape(emb, (batch_size, h*w, embed_dim))
        return emb

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega   # (D/2,)

        pos = pos.reshape(-1)   # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)   # (M, D/2)
        emb_cos = np.cos(out)   # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def calculate_grid(self, pos, grid_size) -> Tuple[Tensor, Tensor]:
        batch_size = pos.shape[0]

        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid1 = grid.reshape([2, grid_size, grid_size])
        grid2 = grid1.copy()

        grid1 = np.repeat(grid1[np.newaxis, :], batch_size, axis=0)

        grid2 = PositionalEncoding.encode_relative_position(grid2, pos, grid_size)

        return torch.tensor(grid1), torch.tensor(grid2)

    def forward(self, pos, grid_size) -> Tuple[Tensor, Tensor]:
        batch_size = pos.shape[0]
        grid1, grid2 = self.calculate_grid(pos, grid_size)

        pos_embed1 = PositionalEncoding.get_2d_sincos_pos_embed_from_grid(self.embed_dim, grid1)
        pos_embed2 = PositionalEncoding.get_2d_sincos_pos_embed_from_grid(self.embed_dim, grid2)

        pos_embed1 = pos_embed1.view(batch_size, -1)
        pos_embed2 = pos_embed2.view(batch_size, -1)

        return pos_embed1, pos_embed2



