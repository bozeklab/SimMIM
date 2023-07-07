import random

import PIL
import torch.distributed as dist

import torch
import torchvision.transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import default_collate
from torchvision.datasets import ImageFolder

from data.img_with_picke_dataset import ImgWithPickleDataset
from data.mask_generator import MaskGenerator
from data.transforms import GaussianBlur, Solarization, RandomResizedCrop


def pad_boxes(boxes, num_boxes):
    fake_box = torch.tensor(4 * [-1])
    fake_class = torch.tensor([-1])

    if boxes is None:
        return torch.tensor([True] * num_boxes), \
               (fake_box.expand(num_boxes, -1), fake_class.expand(num_boxes, -1))

    boxes_available = boxes.shape[0]

    if boxes.shape[0] <= num_boxes:
        padding_length = num_boxes - boxes_available
        fake_box = fake_box.expand(padding_length, -1)
        boxes = torch.cat([boxes, fake_box])
        return boxes

    idx = random.sample(range(boxes_available), num_boxes)
    return boxes[idx]


class COSiamMIMTransform:
    def __init__(self, config):
        self.base_image = T.Compose([T.Resize(size=config.DATA.IMG_SIZE, interpolation=PIL.Image.BILINEAR),
                                     T.ToTensor()])

        self.transform_img = T.Compose([
            T.Resize(size=config.DATA.IMG_SIZE, interpolation=PIL.Image.BILINEAR),
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(0.1),
            Solarization(0.2),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.ToTensor(),
            RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.47, 1.), ratio=(3. / 4., 4. / 3.)),
            #T.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD)),
        ])

        if config.MODEL.TYPE == 'swin':
            model_patch_size = config.MODEL.SWIN.PATCH_SIZE
        elif config.MODEL.TYPE == 'vit':
            model_patch_size = config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError

        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE,
            mask_patch_size=config.DATA.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.DATA.MASK_RATIO,
        )

        self.num_boxes = config.DATA.NUM_BOXES

    def __call__(self, img, boxes):
        x1, pos1 = self.transform_img(img)
        x2, pos2 = self.transform_img(img)
        pos = pos1 + pos2
        mask = self.mask_generator()

        boxes = pad_boxes(torch.tensor(boxes), self.num_boxes)

        return {
            'x0': self.base_image(img),
            'x1': x1,
            'x2': x2,
            'random_crop': pos,
            'mask': mask,
            'boxes': boxes,
        }


def collate_fn(batch):
    batch_num = len(batch)

    keys = batch[0].keys()
    collated = {}

    for key in keys:
        if key == 'random_crop':
            crops = [batch[i][key] for i in range(batch_num)]
            crops = [item for tup in crops for item in tup]
            crops = default_collate(crops).view(batch_num, -1)
            collated[key] = crops
        else:
            collated[key] = default_collate([batch[i][key] for i in range(batch_num)])

    return collated


def build_loader_cosiam(config, logger):
    transform = COSiamMIMTransform(config)
    logger.info(f'Pre-train data transform:\n{transform}')

    dataset = ImgWithPickleDataset(config.DATA.DATA_PATH, transform)
    logger.info(f'Build dataset: train images = {len(dataset)}')

    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True)
    dataloader = DataLoader(dataset, config.DATA.BATCH_SIZE, sampler=sampler, num_workers=config.DATA.NUM_WORKERS,
                            pin_memory=True, drop_last=True, collate_fn=collate_fn)

    return dataloader