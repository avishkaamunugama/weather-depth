import torch
from torchvision.utils import save_image
import numpy as np
import os
import cv2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import permutations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as TF
import kornia.metrics as metrics
import matplotlib.cm as cm
import torchvision.utils as vutils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


######################## DATASET ########################

class WeatherDepthDataset(Dataset):
    def __init__(self, csv_file, root_dir, is_train=False):
        
        csv_path = os.path.join(root_dir,csv_file)
        input_data = pd.read_csv(csv_path)
        list_files = input_data.values.tolist()
        
        self.root_dir = root_dir
        self.is_train = is_train
        self.list_files = list_files

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        sample = self.list_files[index]
        
        img_name = os.path.join(self.root_dir,sample[0])
        input_image = Image.open(img_name)

        depth_name = os.path.join(self.root_dir,sample[2])
        depth_image = Image.open(depth_name).convert('L')
        
        sample = {'image': input_image, 'depth': depth_image}
        
        if self.is_train: 
            transform = TF.Compose([
                Augmentation(0.5),
                ToTensor()
                ])
        else:
            transform = TF.Compose([
                ToTensor()
                ])
            
        return transform(sample)
    

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

        if not _is_pil_image(image):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(depth)))
        
        # flipping the image
        if random.random() < 0.5:
            #random number generated is less than 0.5 then flip image and depth
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        
        # rearranging the channels    
        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(image[...,list(self.indices[random.randint(0, len(self.indices) - 1)])])    

        return {'image': image, 'depth': depth}
    
class ToTensor(object):

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        
        image = self.to_tensor(image)

        depth = depth.resize((256, 256))
        
        depth = self.to_tensor(depth).float() * 1000
        
        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {'image': image, 'depth': depth}

    def to_tensor(self, pic):
        pic = np.array(pic)
        if not (_is_numpy_image(pic) or _is_pil_image(pic)):
                raise TypeError(  'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))
             
        if isinstance(pic, np.ndarray):
            if pic.ndim==2:
                pic=pic[..., np.newaxis]
                
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)


######################## TENSORBOARD configs ########################

def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    value = value.cpu().numpy()[0,:,:]

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    #value = value.squeeze(axis=0)

    cmapper = cm.get_cmap(cmap)
    value = cmapper(value,bytes=True) # (nxmx4)

    img = value[:,:,:3]

    return img.transpose((2,0,1))

def LogProgress(model, writer, test_loader, epoch):
  model.eval()
  sequential = test_loader
  sample_batched = next(iter(sequential))
  image = torch.autograd.Variable(sample_batched['image'].cuda())
  depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
  if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
  if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)
  output = DepthNorm( model(image) )
  writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
  writer.add_image('Train.3.Diff', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)
  del image
  del depth
  del output


######################## METRICS & NORMALISATION ########################

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    ssim = metrics.SSIM(window_size=11,max_val=val_range)
    return ssim(img1, img2)

def DepthNorm(depth, maxDepth=1000.0): 
    return maxDepth / depth

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


######################## SAVE/LOAD CHECKPOINTS ########################

def save_some_examples(gen, val_loader, epoch, folder):
    sample = next(iter(val_loader))
    x, y = sample['image'].to(DEVICE), sample['depth'].to(DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = F.interpolate(y_fake, size=(512, 512)) # resize depth map
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(y, folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def upscale_depth_img(depth_img, scale_factor=2):
    depth_img=depth_img.reshape(256, 256)
    scale_percent = (scale_factor * 100) # percent of original size
    width = int(depth_img.shape[1] * scale_percent / 100)
    height = int(depth_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    depth_img_resized = cv2.resize(depth_img, dim, interpolation = cv2.INTER_LINEAR)
    return depth_img_resized

def upscale_depth_tensor(depth_tensor, scale_factor=2):
    depth_tensor = F.interpolate( depth_tensor, 
                                  scale_factor=scale_factor, 
                                  mode='bilinear', 
                                  align_corners=True
                                  )
    return depth_tensor


def evaluate_model(model,eval_dataloader):

    model.eval()

    preds = np.zeros((512, 512, len(eval_dataloader.dataset)), dtype=np.float32)
    labels = np.zeros((512, 512, len(eval_dataloader.dataset)), dtype=np.float32)

    # Starting evaluation ...
    for i,sample_test  in enumerate (eval_dataloader):
        img = torch.autograd.Variable(sample_test['image'].cuda())
        label = torch.autograd.Variable(DepthNorm(sample_test['depth']))

        # inference
        pred_depth= model(img)
        pred_depth = F.relu(pred_depth).detach().cpu().numpy()
        
        # up-sampling
        label = upscale_depth_tensor(label)
        pred_resized = upscale_depth_img(pred_depth)

        # store the label and the corresponding prediction
        labels[:, :, i] = label
        preds[:, :, i] = pred_resized

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(labels, preds)
    print('Abs_Rel: {:.6f}, Sq_Rel: {:.6f}, RMSE: {:.6f}, RMSE_Log: {:.6f}'.format(abs_rel, sq_rel, rmse, rmse_log))
    print('Accuracy Metrics a1: {:.6f}, a2: {:.6f}, a3: {:.6f}\n'.format(a1, a2, a3))

    model.train()

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt+1e-10) - np.log(pred+1e-10)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr