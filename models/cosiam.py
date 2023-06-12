import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .vision_transformer import VisionTransformer


class VisionTransformerForCOSiam(VisionTransformer):
    def __init__(self, num_projection_layers, last_projection_bn, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.num_projection_layers = num_projection_layers
        self.last_projection_bn = last_projection_bn
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def _build_mlp(self):
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)

        mlp = []
        for l in range(self.num_projection_layers):
            dim1 = self.embed_dim if l == 0 else mlp_hidden_dim
            dim2 = self.embed_dim if l == self.num_projection_layers - 1 else mlp_hidden_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < self.num_projection_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif self.last_projection_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def forward(self, x):
        x = self.patch_embed(x)

        B, L, _ = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class COSiam(nn.Module):
    """Momentum encoder implementation stolen from MoCo v3"""
    def __init__(self, encoder, decoder):
        super().__init__()

        self.base_encoder = encoder
        self.momentum_encoder = encoder
        self.decoder = decoder

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def forward(self, x1, x2, m, mask):
        """
         Input:
            x1: first views of images
            x2: second views of images
            m: momentum of the target encoder
         Output:
            loss
        """
        assert mask is not None
        B, L, _ = x1.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x1 = x1 * (1 - w) + mask_token * w

        y_a = self.base_encoder(x1)
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)    # update the momentum encoder
            x_b = self.momentum_encoder(x2)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = set()
        for field_name in ['base_encoder', 'momentum_encoder', 'decoder']:
            field_value = getattr(self, field_name)
            no_weight_decay.update({f'{field_name}.' + i for i in field_value.no_weight_decay()}) \
                if hasattr(self.field_value, 'no_weight_decay') else {}
        return no_weight_decay

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        no_weight_decay = set()
        for field_name in ['base_encoder', 'momentum_encoder', 'decoder']:
            field_value = getattr(self, field_name)
            no_weight_decay.update({f'{field_name}.' + i for i in field_value.no_weight_decay()}) \
                if hasattr(self.field_value, 'no_weight_decay_keywords') else {}
        return no_weight_decay
