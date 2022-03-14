from torchvision.transforms import functional
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils
import torch
import random

class DataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot90=False,
            with_random_rot180=False,
            with_random_rot270=False,
            with_random_crop=False,
            with_color_jittering=False,
            crop_ratio=(0.9, 1.1)
    ):
        self.img_size = img_size
        self.gt_size = (img_size[0]//2,img_size[1]//2)
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot90 = with_random_rot90
        self.with_random_rot180 = with_random_rot180
        self.with_random_rot270 = with_random_rot270
        self.with_random_crop = with_random_crop
        self.with_color_jittering = with_color_jittering
        self.crop_ratio = crop_ratio

    def transform(self, img):

        # resize image and covert to tensor
        img = functional.to_pil_image(img)
        img = functional.resize(img, self.img_size)

        if self.with_random_hflip and random.random() > 0.5:
            img = functional.hflip(img)

        if self.with_random_vflip and random.random() > 0.5:
            img = functional.vflip(img)

        if self.with_random_rot90 and random.random() > 0.5:
            img = functional.rotate(img, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img = functional.rotate(img, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img = functional.rotate(img, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img = functional.adjust_hue(img, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img = functional.adjust_saturation(img, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img, scale=(0.5, 1.0), ratio=self.crop_ratio)
            img = functional.resized_crop(
                img, i, j, h, w, size=self.img_size)

        # to tensor
        img = functional.to_tensor(img)
        return img

    def transform_pair(self, img, gt):

        # resize image and covert to tensor
        img = functional.to_pil_image(img)
        img = functional.resize(img, self.img_size)

        gt = functional.to_pil_image(gt)
        gt = functional.resize(gt, self.gt_size)

        if self.with_random_hflip and random.random() > 0.5:
            img = functional.hflip(img)
            gt = functional.hflip(gt)

        if self.with_random_vflip and random.random() > 0.5:
            img = functional.vflip(img)
            gt = functional.vflip(gt)

        if self.with_random_rot90 and random.random() > 0.5:
            img = functional.rotate(img, 90)
            gt = functional.rotate(gt, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img = functional.rotate(img, 180)
            gt = functional.rotate(gt, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img = functional.rotate(img, 270)
            gt = functional.rotate(gt, 270)

        if self.with_color_jittering and random.random() > 0.5:
            img = functional.adjust_hue(img, hue_factor=random.random()*0.5-0.25)  # -0.25 ~ +0.25
            img = functional.adjust_saturation(img, saturation_factor=random.random()*0.8 + 0.8)  # 0.8 ~ +1.6
            gt = functional.adjust_hue(gt, hue_factor=random.random() * 0.5 - 0.25)  # -0.25 ~ +0.25
            gt = functional.adjust_saturation(gt, saturation_factor=random.random() * 0.8 + 0.8)  # 0.8 ~ +1.6

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img, scale=(0.5, 1.0), ratio=self.crop_ratio)
            
            img = functional.resized_crop(
                img, i, j, h, w, size=self.img_size)
            gt = functional.resized_crop(
                gt, i//2, j//2, h//2, w//2, size=self.gt_size)
            
        
        # to tensor
        img = functional.to_tensor(img)
        gt = functional.to_tensor(gt).float() * 1000
        gt = torch.clamp(gt, 10, 1000)

        return img, gt