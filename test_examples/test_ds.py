import os
import cv2
import numpy as np

from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torchvision.datasets import ImageFolder
from yacs.config import CfgNode as CN

from data.data_cosiam import COSiamMIMTransform
from data.data_simmim import collate_fn
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


def add_border(image):
    # Get the dimensions of the image tensor
    height, width, channels = image.shape

    # Create a new image with the border
    bordered_image = np.ones((height + 10, width + 10, channels), dtype=np.uint8) * 255

    # Insert the original image into the bordered image
    bordered_image[5:height + 5, 5:width + 5, :] = image

    return bordered_image


def create_image_grid(images):
    # Determine the dimensions of each image in the grid
    rows, cols, _ = images[0].shape

    # Determine the number of images and columns in the grid
    num_images = len(images)
    num_cols = 2

    # Set the border size and color
    border_size = 5

    # Create a blank grid image to hold the combined grid
    grid_height = (rows + 2 * border_size) * (num_images // num_cols)
    grid_width = (cols + 2 * border_size) * num_cols
    grid = np.full((grid_height, grid_width, 3), 255)

    # Convert images to cv2 format with integer pixel values
    images = [np.uint8(image * 255) for image in images]

    # Populate the grid with the individual images and add borders
    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        x = col * cols
        y = row * rows

        # Add the image with border to the grid
        grid[y:y + rows + 2 * border_size, x:x + cols + 2 * border_size, :] = add_border(image)

    grid = grid.astype(np.uint8)

    # Display the grid image using OpenCV
    cv2.imshow("Augmentations", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = _C.clone()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT,
                           dist_rank=-1, name=f"{config.MODEL.NAME}")

    transform = COSiamMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImageFolder(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler,
                            pin_memory=True, drop_last=True, collate_fn=collate_fn)

    images = []

    for idx, (x1, x2, mask, _) in enumerate(dataloader):
        img1 = x1.permute(0, 2, 3, 1)
        img2 = x2.permute(0, 2, 3, 1)
        images.append(img1[0])
        images.append(img2[0])
        if idx == 4:
            break

    create_image_grid(images)

