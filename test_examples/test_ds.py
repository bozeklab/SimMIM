import os
import cv2
import numpy as np

from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torchvision.datasets import ImageFolder
from yacs.config import CfgNode as CN
from mpl_toolkits.axes_grid1 import ImageGrid

from data.data_simmim import SimMIMTransform, collate_fn
from logger import create_logger

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 2
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = '/Users/piotrwojcik/sample_he/'
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 512
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6
# Fake log output
_C.OUTPUT = '/Users/piotrwojcik/cosiam_log/'


# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'vit'
# Model name
_C.MODEL.NAME = 'cosiam'
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16


def create_image_grid(images):
    # Determine the dimensions of each image in the grid
    rows, cols, _ = images[0].shape

    # Determine the number of images and columns in the grid
    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))

    # Create a blank grid image to hold the combined grid
    grid_height = rows * int(np.ceil(num_images / num_cols))
    grid_width = cols * num_cols
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

    # Populate the grid with the individual images
    for i, image in enumerate(images):
        row = int(i / num_cols)
        col = i % num_cols
        x = col * cols
        y = row * rows
        grid[y:y + rows, x:x + cols, :] = image

    # Display the grid image using OpenCV
    cv2.imshow("Image Grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = _C.clone()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=-1, name=f"{config.MODEL.NAME}")

    transform = SimMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler,
                            pin_memory=True, drop_last=True, collate_fn=collate_fn)

    images = []

    for idx, (img, mask, _) in enumerate(dataloader):
        img = img.permute(0, 2, 3, 1)
        images.append(img[0])
        if idx == 4:
            break

    create_image_grid(images)

