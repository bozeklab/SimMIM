import os
import cv2
import numpy as np
import torch
from PIL.Image import Image

from torch.utils.data import RandomSampler, BatchSampler, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import draw_bounding_boxes
from yacs.config import CfgNode as CN
import torchvision.transforms as T
from torchvision.ops import masks_to_boxes

from data.data_cosiam import COSiamMIMTransform, collate_fn
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


def gray_out_square(image, x_start, y_start, size, alpha):
    # Get the dimensions of the image tensor
    height, width, _ = image.shape

    # Calculate the end coordinates of the square region
    x_end = min(x_start + size, width)
    y_end = min(y_start + size, height)

    # Create a gray overlay image
    gray_overlay = alpha * image[y_start:y_end, x_start:x_end]

    # Replace the square region with the gray overlay
    image[y_start:y_end, x_start:x_end] = gray_overlay

    return image


def gray_out_mask(image, mask, patch_size, alpha):
    mh, mw = mask.shape

    for i in range(mh):
        for j in range(mw):
            if mask[i][j]:
                image = gray_out_square(image, i * patch_size, j * patch_size, patch_size, alpha)
    return image


def create_image_grid(images):
    # Determine the dimensions of each image in the grid
    rows, cols, _ = images[0].shape

    # Determine the number of images and columns in the grid
    num_images = len(images)
    num_cols = 3

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
    cv2.imshow("image", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def interleave_lists(*lists):
    max_length = max(len(lst) for lst in lists)
    interleaved = [val for pair in zip(*lists) for val in pair]

    for lst in lists:
        if len(lst) > max_length:
            interleaved += lst[max_length:]

    return interleaved


normalize = transforms.Compose([
    lambda x: x.float() / 255.0,
    lambda x: torch.permute(x, (1, 2, 0)),
])

denormalize = transforms.Compose([
    lambda x: torch.permute(x, (2, 0, 1)),
    lambda x: x * 255.0,
    lambda x: x.to(torch.uint8)
])


def draw_crop_boxes(images, crops):
    boxes = crops.clone()

    annotated_images = []

    for idx, image in enumerate(images):
        view1_box = boxes[idx, :4]
        view1_box[2:], view1_box[3] = view1_box[:2] + view1_box[2:], view1_box[1] + view1_box[3]
        view1_box = view1_box.unsqueeze(0)
        view1_box[:, [0, 1, 2, 3]] = view1_box[:, [1, 0, 3, 2]]

        view2_box = boxes[idx, 4:]
        view2_box[2:], view2_box[3] = view2_box[:2] + view2_box[2:], view2_box[1] + view2_box[3]
        view2_box = view2_box.unsqueeze(0)
        view2_box[:, [0, 1, 2, 3]] = view2_box[:, [1, 0, 3, 2]]

        views_boxes = torch.cat([view1_box, view2_box], dim=0)

        annotated_image = draw_bounding_boxes(denormalize(image), views_boxes, width=2, colors=["yellow", "green"])
        annotated_images.append(normalize(annotated_image))
    annotated_images = [img for img in annotated_images]
    return annotated_images


def tensor_batch_to_list(tensor):
    tensor_list = [t for t in tensor]
    return tensor_list


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

    for idx, sample in enumerate(dataloader):
        x0 = sample['x0']
        x1 = sample['x1']
        x2 = sample['x2']
        pos = sample['pos']

        mask = sample['mask']

        img0 = x0.permute(0, 2, 3, 1)
        img1 = x1.permute(0, 2, 3, 1)
        img2 = x2.permute(0, 2, 3, 1)

        img0 = tensor_batch_to_list(img0)
        img1 = tensor_batch_to_list(img1)
        img2 = tensor_batch_to_list(img2)

        mask = tensor_batch_to_list(mask)

        img0 = draw_crop_boxes(img0, pos)
        img1 = [gray_out_mask(img, mask, config.MODEL.VIT.PATCH_SIZE, alpha=0.5) for img, mask in zip(img1, mask)]
        imgs = interleave_lists(img0, img1, img2)
        images.extend(imgs)
        if idx == 1:
            break

    create_image_grid(images)

