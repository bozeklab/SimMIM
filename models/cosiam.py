from functools import partial

import torch
import copy
import torch.nn as nn
from timm.models.layers import trunc_normal_

from util.pos_embedding import PositionalEmbedding
from .vision_transformer import VisionTransformer


class VisionTransformerEncoder(VisionTransformer):
    def __init__(self, num_projection_layers=None, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        mlp_ratio = kwargs['mlp_ratio']

        if num_projection_layers is not None:
            self.projector = self._build_mlp(num_projection_layers=num_projection_layers, mlp_ratio=mlp_ratio)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def _build_mlp(self, mlp_ratio, num_projection_layers, last_projection_bn=True):
        mlp_hidden_dim = int(self.embed_dim * mlp_ratio)

        mlp = []
        for l in range(num_projection_layers):
            dim1 = self.embed_dim if l == 0 else mlp_hidden_dim
            dim2 = self.embed_dim if l == num_projection_layers - 1 else mlp_hidden_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_projection_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_projection_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def forward(self, x, mask=None):
        x = self.patch_embed(x)

        B, L, _ = x.shape

        if mask is not None:
            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        if self.projector:
            x = x.flatten(0, 1)
            x = self.projector(x)
            x = x.reshape(B, L + 1, -1)

        x = x[:, 1:]
        return x


class VisionTransformerDecoder(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0
        self.patch_embed = None

        grid_size = kwargs['img_size'] // kwargs['patch_size']

        self.pos_embed = PositionalEmbedding(grid_size, self.embed_dim)

        self.patch_embed = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, random_crop, mask):
        B, L, C = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        p_a, p_b = self.pos_embed(random_crop, L ** 2)

        p = p_a * (1 - w) + p_b * w
        x = x + p
        x = self.pos_drop(x)

        rel_pos_bias = None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        return x


class COSiam(nn.Module):
    """Momentum encoder implementation stolen from MoCo v3"""
    def __init__(self, encoder, decoder, rho, lambd):
        super().__init__()

        self.F = None

        self.rho = rho
        self.lambd = lambd

        self.base_encoder = encoder
        self.momentum_encoder = copy.deepcopy(self.base_encoder)
        self.decoder = decoder

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def loss_unigrad(self, z1, z2, z1m, z2m):
        # calculate correlation matrix
        tmp_F = (torch.mm(z1m.t(), z1m) + torch.mm(z2m.t(), z2m)) / (2 * z1m.shape[0])
        torch.distributed.all_reduce(tmp_F)
        tmp_F = tmp_F / torch.distributed.get_world_size()
        if self.F is None:
            self.F = tmp_F.detach()
        else:
            self.F = self.rho * self.F + (1 - self.rho) * tmp_F.detach()

        # compute grad & loss
        grad1 = -z2m + self.lambd * torch.mm(z1, self.F)
        loss1 = (grad1.detach() * z1).sum(-1).mean()

        grad2 = -z1m + self.lambd * torch.mm(z2, self.F)
        loss2 = (grad2.detach() * z2).sum(-1).mean()

        loss = 0.5 * (loss1 + loss2)

        # compute positive similarity, just for observation
        pos_sim1 = torch.einsum('nc,nc->n', [z1, z2m]).mean().detach()
        pos_sim2 = torch.einsum('nc,nc->n', [z2, z1m]).mean().detach()
        pos_sim = 0.5 * (pos_sim1 + pos_sim2)

        return loss, pos_sim

    def forward(self, x1, x2, random_crop, mm, mask):
        assert mask is not None

        ya1 = self.base_encoder(x1)
        ya2 = self.base_encoder(x2)

        z1 = self.decoder(ya1, random_crop, mask)
        random_crop = torch.concat([random_crop[:, 4:], random_crop[:, :4]], dim=1)
        z2 = self.decoder(ya2, random_crop, mask)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(mm)    # update the momentum encoder
            z1m = self.momentum_encoder(x1)
            z2m = self.momentum_encoder(x2)

        B, L, C = z1.shape

        z1 = z1.reshape((B * L, C))
        z1m = z1m.reshape((B * L, C))
        z2 = z2.reshape((B * L, C))
        z2m = z2m.reshape((B * L, C))

        # normalize
        z1 = torch.nn.functional.normalize(z1)
        z2 = torch.nn.functional.normalize(z2)
        z1m = torch.nn.functional.normalize(z1m)
        z2m = torch.nn.functional.normalize(z2m)

        #loss, _ = self.loss_unigrad(z1, z2, z1m, z2m)
        loss = (torch.mm(z1m.t(), z1m) + torch.mm(z2m.t(), z2m)) / (2 * z1m.shape[0])
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = set()
        for field_name in ['base_encoder', 'momentum_encoder', 'decoder']:
            field_value = getattr(self, field_name)
            no_weight_decay.update({f'{field_name}.' + i for i in field_value.no_weight_decay()}) \
                if hasattr(field_value, 'no_weight_decay') else {}
        return no_weight_decay

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        no_weight_decay = set()
        for field_name in ['base_encoder', 'momentum_encoder', 'decoder']:
            field_value = getattr(self, field_name)
            no_weight_decay.update({f'{field_name}.' + i for i in field_value.no_weight_decay()}) \
                if hasattr(field_value, 'no_weight_decay_keywords') else {}
        return no_weight_decay


def build_cosiam(config):
    model_type = config.MODEL.TYPE
    if model_type == 'vit':
        encoder = VisionTransformerEncoder(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.ENCODER.VIT.PATCH_SIZE,
            embed_dim=config.MODEL.ENCODER.VIT.EMBED_DIM,
            num_classes=0,
            depth=config.MODEL.ENCODER.VIT.DEPTH,
            num_heads=config.MODEL.ENCODER.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.ENCODER.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.ENCODER.VIT.QKV_BIAS,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.ENCODER.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.ENCODER.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.ENCODER.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.ENCODER.VIT.USE_MEAN_POOLING,
            num_projection_layers=config.MODEL.ENCODER.VIT.NUM_PROJECTION_LAYERS)

        decoder = VisionTransformerDecoder(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.DECODER.VIT.PATCH_SIZE,
            num_classes=0,
            embed_dim=config.MODEL.DECODER.VIT.EMBED_DIM,
            depth=config.MODEL.DECODER.VIT.DEPTH,
            num_heads=config.MODEL.DECODER.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.DECODER.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.DECODER.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.DECODER.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.DECODER.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.DECODER.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.DECODER.VIT.USE_MEAN_POOLING)
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    model = COSiam(encoder=encoder,
                   decoder=decoder,
                   rho=config.MODEL.UNIGRAD.RHO,
                   lambd=config.MODEL.UNIGRAD.LAMBD)

    return model