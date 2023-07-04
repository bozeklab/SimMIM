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
import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim
        self.dim_fix = nn.Linear(2 * embed_dim, embed_dim)

    @staticmethod
    def encode_relative_position(grid, random_crop, grid_size):
        # i1, j1, h1, w1, i2, j2, h2, w2
        batch_size = random_crop.shape[0]
        grid = grid.to(random_crop.device)

        grid = grid.unsqueeze(dim=0)

        # h2 / h1
        h = random_crop[:, 6] / random_crop[:, 2]
        # w2 / w1
        w = random_crop[:, 7] / random_crop[:, 3]
        h = h.reshape(batch_size, 1, 1, 1)
        w = w.reshape(batch_size, 1, 1, 1)

        # Nh * (i2 - i1) / h1
        b_h = grid_size * (random_crop[:, 4] - random_crop[:, 0]) / random_crop[:, 2]
        # Nw * (j2 - j1) / w1
        b_w = grid_size * (random_crop[:, 5] - random_crop[:, 1]) / random_crop[:, 3]
        b_h = b_h.reshape(batch_size, 1, 1, 1)
        b_w = b_w.reshape(batch_size, 1, 1, 1)

        rp_h = torch.multiply(grid[:, 0], h) + b_h
        rp_w = torch.multiply(grid[:, 1], w) + b_w

        rp = torch.cat([rp_h, rp_w], dim=1)
        return rp

    @staticmethod
    def encode_scale_variation(random_crop):
        batch_size = random_crop.shape[0]
        # h2 / h1
        h = random_crop[:, 6] / random_crop[:, 2]
        # w2 / w1
        w = random_crop[:, 7] / random_crop[:, 3]
        h = h.reshape(batch_size, 1, 1, 1)
        w = w.reshape(batch_size, 1, 1, 1)

        rp = torch.cat([h, w], dim=1)
        rp = 10. * torch.log(rp)
        return rp

    @staticmethod
    def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
        assert embed_dim % 2 == 0
        batch_size, _, h, w = grid.shape

        # use half of dimensions to encode grid_h
        emb_h = PositionalEmbedding.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 0])  # (N, H*W, D/2)
        emb_w = PositionalEmbedding.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[:, 1])  # (N, H*W, D/2)

        emb = torch.cat([emb_h, emb_w], dim=1)    # (N, H*W, D)
        emb = emb.reshape((batch_size, h*w, embed_dim))
        return emb

    @staticmethod
    def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=float).to(pos.device)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega   # (D/2,)

        pos = pos.reshape(-1)   # (M,)
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = torch.sin(out)   # (M, D/2)
        emb_cos = torch.cos(out)   # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb

    def calculate_grid(self, random_crop, grid_size) -> Tuple[Tensor, Tensor]:
        batch_size = random_crop.shape[0]

        grid_h = torch.arange(grid_size).float()

        grid_w = torch.arange(grid_size).float()
        grid = torch.meshgrid(grid_w, grid_h)  # here w goes first
        grid = torch.stack((grid[0].t(), grid[1].t()), dim=0)

        grid1 = grid.reshape([2, grid_size, grid_size])
        grid2 = grid1.clone()

        grid1 = grid1.unsqueeze(0).repeat(batch_size, 1, 1)
        grid2 = PositionalEmbedding.encode_relative_position(grid2, random_crop, grid_size)

        return grid1, grid2

    def forward(self, random_crop, grid_size, cls_token=False) -> Tuple[Tensor, Tensor]:
        grid1, grid2 = self.calculate_grid(random_crop, grid_size)

        pos_embed1 = PositionalEmbedding.get_2d_sincos_pos_embed_from_grid(self.embed_dim, grid1)
        pos_embed2 = PositionalEmbedding.get_2d_sincos_pos_embed_from_grid(self.embed_dim, grid2)

        pos_embed1 = pos_embed1.float()
        pos_embed2 = pos_embed2.float()

        scale_variation = self.encode_scale_variation(random_crop)
        scale_variation = PositionalEmbedding.get_2d_sincos_pos_embed_from_grid(self.embed_dim, scale_variation)
        scale_variation = scale_variation.float()

        scale_variation = scale_variation.repeat(1, pos_embed2.shape[1], 1)

        pos_embed2 = self.dim_fix(torch.concat([pos_embed2, scale_variation], dim=-1))

        if cls_token:
            pos_embed1 = torch.concat([torch.zeros([pos_embed1.shape[0], 1, self.embed_dim]), pos_embed1], dim=1)
            pos_embed2 = torch.concat([torch.zeros([pos_embed2.shape[0], 1, self.embed_dim]), pos_embed2], dim=1)

        return pos_embed1, pos_embed2



