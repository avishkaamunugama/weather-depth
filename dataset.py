from torch.utils.data import Dataset, DataLoader
from utils import DataAugmentation
from PIL import Image
import numpy as np
import os
import glob
import random
import torch
import cv2


class AdverseWeatherDataset(Dataset):

    def __init__(self, root_dir, img_size=(480,640), suff='.png', gt_left=False, is_train=True):
        self.root_dir = root_dir
        self.dirs = glob.glob(os.path.join(self.root_dir, '*'+suff))
        self.img_size = img_size
        self.gt_left = gt_left
        if is_train:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_crop=True)
        else:
            self.augm = DataAugmentation(
                img_size=self.img_size,
                with_random_hflip=False,
                with_random_crop=False)

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # input: color image (h x w x 3)
        this_dir = self.dirs[idx]
        img_pair = cv2.imread(this_dir, cv2.IMREAD_COLOR)
        img_pair = cv2.cvtColor(img_pair, cv2.COLOR_BGR2RGB)
        h, w, c = img_pair.shape

        if self.gt_left:
            depth = img_pair[:, 0:int(w/2), :]
            image = img_pair[:, int(w/2):, :]
        else:
            image = img_pair[:, 0:int(w / 2), :]
            depth = img_pair[:, int(w / 2):, :]

        depth = cv2.cvtColor(depth, cv2.COLOR_RGB2GRAY)
        
        image, depth = self.augm.transform_pair(image, depth)

        data = {
            'image': image,
            'depth': depth
        }

        return data


