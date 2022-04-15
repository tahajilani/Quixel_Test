from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cv2
from pathlib2 import Path
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class customDataset(Dataset):
    """Dataset class for images"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.images_path =list(Path(root_dir).rglob("*.[jJ][pP][gG]"))
        self.transform = transform

    def __len__(self):
      return len(self.images_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images_path[idx]
        # print(img_name)
        image = cv2.imread(str(img_name))
        if self.transform:
            sample=image.astype(np.float32)
            norm_image = cv2.normalize(sample, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            sample = self.transform(norm_image)
        else:
          sample=image
        return sample
        
    def getSample(self, idx):

        img_name = self.images_path[idx]
        # print(img_name)
        image = cv2.imread(str(img_name))

        if self.transform:
            sample = self.transform(image)
        else:
          sample=image

        return sample

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image= sample
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)