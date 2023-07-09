import torch
import random
import numpy as np
from PIL import Image

######################## HYPERPARAMETERS ########################

ROOT_DIR = 'data'
TRAIN_CSV = 'train.csv'
TEST_CSV = 'test.csv'
VAL_CSV = 'val.csv'
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
BETAS = (0.5, 0.999)
L1_LAMBDA = 100
SSIM_LAMBDA = 10
LAMBDA_GP = 10
NUM_EPOCHS = 10
MAX_DEPTH = 1000
MIN_DEPTH = 10
LOAD_MODEL = False
SAVE_MODEL = True
CMAP_STYLE = 'inferno'
CHECKPOINT_DISC = 'checkpoints/disc.pt'
CHECKPOINT_GEN = 'checkpoints/gen.pt'
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
        image, clear, depth = sample['image'], sample['clear'], sample['depth']

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(clear):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(clear)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))
        
        # flipping the image
        if random.random() < 0.5:
            #random number generated is less than 0.5 then flip image and depth
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            clear = clear.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        
        # rearranging the channels    
        if random.random() < self.probability:

            randNum = random.randint(0, len(self.indices) - 1)
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[randNum])])
            clear = np.asarray(clear)
            clear = Image.fromarray(clear[...,list(self.indices[randNum])])    

        return {'image': image, 'clear': clear, 'depth': depth}
    
class ToTensor(object):

    def __call__(self, sample):
        image, clear, depth = sample['image'], sample['clear'], sample['depth']
        
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        image = self.to_tensor(image)

        clear = clear.resize((IMAGE_SIZE, IMAGE_SIZE))
        clear = self.to_tensor(clear)
 
        depth = depth.resize((IMAGE_SIZE, IMAGE_SIZE))
        depth = self.to_tensor(depth).float() * MAX_DEPTH
        
        # put in expected range
        depth = torch.clamp(depth, MIN_DEPTH, MAX_DEPTH)

        return {'image': image, 'clear': clear, 'depth': depth}

    def to_tensor(self, pic):
        pic = np.array(pic)
        if not (_is_numpy_image(pic) or _is_pil_image(pic)):
                raise TypeError(  'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
             
        if isinstance(pic, np.ndarray):
            if pic.ndim==2:
                pic=pic[..., np.newaxis]
                
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

