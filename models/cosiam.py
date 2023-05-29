import torch
import torch.nn as nn


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

    def forward(self, x, mask):
        pass

    @torch.jit.ignore
    def no_weight_decay(self):
        no_weight_decay = set()
        for field_name in ['base_encoder', 'momentum_encoder', 'decoder']:
            field_value = getattr(self, field_name)
            no_weight_decay.update({field_name + i for i in field_value.no_weight_decay()}) \
                if hasattr(self.field_value, 'no_weight_decay') else {}
        return no_weight_decay

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        no_weight_decay = set()
        for field_name in ['base_encoder', 'momentum_encoder', 'decoder']:
            field_value = getattr(self, field_name)
            no_weight_decay.update({field_name + i for i in field_value.no_weight_decay()}) \
                if hasattr(self.field_value, 'no_weight_decay_keywords') else {}
        return no_weight_decay
