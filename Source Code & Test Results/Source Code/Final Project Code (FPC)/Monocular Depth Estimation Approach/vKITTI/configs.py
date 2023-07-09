import cv2
import torch
import random
import numpy as np
from PIL import Image

######################## HYPERPARAMETERS ########################

ROOT_DIR = 'data'
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test_medium (1).csv'
VAL_CSV = 'test_small.csv'
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = (828,250)
CHANNELS_IMG = 3
BETAS = (0.5, 0.999)
L1_LAMBDA = 100
SSIM_LAMBDA = 10
LAMBDA_GP = 10
NUM_EPOCHS = 30
MAX_DEPTH = 80
MIN_DEPTH = 1e-3
LOAD_MODEL = False
SAVE_MODEL = True
CMAP_STYLE = 'inferno'
CHECKPOINT_PATH = 'checkpoints'
VAL_EXAMPLES_DIR = 'evaluation'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


######################## TRANSFORMATIONS ########################

def _is_pil_image(img):
    return isinstance(img, Image.Image)

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

class Augmentation(object):
    def __init__(self, probability):
        from itertools import permutations
        self.probability = probability
        #generate some output like this [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        self.indices = list(permutations(range(3), 3))
        #followed by randomly picking one channel in the list above
    
    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        # flipping the image
        if random.random() < 0.5:
            #random number generated is less than 0.5 then flip image and depth
            image = cv2.flip(image,1)
            depth = cv2.flip(depth,1)

        return {'image': image, 'depth': depth}
    
class ToTensor(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        image = self.to_tensor(image)
        depth = self.to_tensor(depth) * MAX_DEPTH
        
        # put in expected range
        depth = torch.clamp(depth, MIN_DEPTH, MAX_DEPTH)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        if pic.ndim==2:
            pic=pic[..., np.newaxis]
            
        img = torch.from_numpy(pic.transpose((2, 0, 1)))

        return img.div(255)

