import torch.nn as nn

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
import torch
from typing import Tuple

from timm.models.layers import trunc_normal_
from torch import Tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, grid_size, embed_dim, device='cuda'):
        super().__init__()

        self.embed_dim = embed_dim
        self.dim_fix = nn.Linear(2 * embed_dim, embed_dim)

        grid_h = torch.arange(grid_size).float().to(device)
        grid_w = torch.arange(grid_size).float().to(device)

        grid = torch.meshgrid(grid_w, grid_h)  # here w goes first
        grid = torch.stack((grid[0].t(), grid[1].t()), dim=0)

        self.grid1 = grid.reshape([2, grid_size, grid_size])

        self.apply(self._init_weights)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self._trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            self._trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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

    def forward(self, random_crop, grid_size, cls_token=False) -> Tuple[Tensor, Tensor]:
        batch_size = random_crop.shape[0]

        grid2 = self.grid1
        grid1 = self.grid1.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        grid2 = PositionalEmbedding.encode_relative_position(grid2, random_crop, grid_size)

        print(grid2)

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



