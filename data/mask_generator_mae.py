import numpy as np


class MaskGeneratorMAE:
    def __init__(self, input_size=192, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

    def __call__(self):
        pass