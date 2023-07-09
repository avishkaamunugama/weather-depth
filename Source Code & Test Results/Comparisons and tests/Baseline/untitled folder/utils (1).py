import torch
from torchvision.utils import save_image
import numpy as np
import random
from PIL import Image
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from itertools import permutations
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as TF
import kornia.metrics as metrics
import matplotlib.cm as cm
import torchvision.utils as vutils
import configs


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

        clear_name = os.path.join(self.root_dir,sample[1])
        clear_image = Image.open(clear_name)

        depth_name = os.path.join(self.root_dir,sample[2])
        depth_image = Image.open(depth_name).convert('L')
        
        sample = {'image': input_image, 'clear': clear_image, 'depth': depth_image}
        
        if self.is_train: 
            transform = TF.Compose([
                configs.Augmentation(0.5),
                configs.ToTensor()
                ])
        else:
            transform = TF.Compose([
                configs.ToTensor()
                ])
            
        return transform(sample)

def upscale_depth_tensor(depth_tensor, scale_factor=2):
    depth_tensor = F.interpolate( depth_tensor, 
                                  scale_factor=scale_factor, 
                                  mode='bilinear', 
                                  align_corners=True
                                  )
    return depth_tensor

######################## TENSORBOARD configs ########################

def colorize(value, vmin=configs.MIN_DEPTH, vmax=configs.MAX_DEPTH, cmap=configs.CMAP_STYLE):
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

  sample = next(iter(test_loader))

  image = torch.autograd.Variable(sample['image'].to(configs.DEVICE))
  depth = torch.autograd.Variable(sample['depth'].to(configs.DEVICE))

  if epoch == 0: writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
  if epoch == 0: writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth.data, nrow=6, normalize=False)), epoch)

  output = model(image)
  output = upscale_depth_tensor(output)

  writer.add_image('Train.3.Ours_depth', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
  writer.add_image('Train.4.Diff_depth', colorize(vutils.make_grid(torch.abs(output-depth).data, nrow=6, normalize=False)), epoch)

  del image
  del depth
  del output

  model.train()


######################## METRICS & NORMALISATION ########################

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=True, full=False):
    ssim = metrics.SSIM(window_size=11,max_val=val_range)
    return ssim(img1, img2)

def DepthNorm(depth, maxDepth=configs.MAX_DEPTH): 
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
     
def upscale_depth_img(depth_img, scale_factor=2):
    depth_img=depth_img.reshape(configs.IMAGE_SIZE//2, configs.IMAGE_SIZE//2)
    scale_percent = (scale_factor * 100) # percent of original size
    width = int(depth_img.shape[1] * scale_percent / 100)
    height = int(depth_img.shape[0] * scale_percent / 100)
    dim = (width, height)
    depth_img_resized = cv2.resize(depth_img, dim, interpolation = cv2.INTER_LINEAR)
    return depth_img_resized
     

def save_some_examples(model, test_loader, epoch, folder=configs.VAL_EXAMPLES_DIR):
    # model.eval()

    sample = next(iter(test_loader))
    image = torch.autograd.Variable(sample['image'].to(configs.DEVICE))
    depth = torch.squeeze(sample['depth'])

    with torch.no_grad():
        save_image(image, folder + f'/input_{epoch}.png')
        if epoch == 1:
            plt.imsave(folder + f'/label_model_{epoch}.png', depth, cmap=configs.CMAP_STYLE)

        depth_fake = model(image)
        depth_fake = depth_fake.detach().cpu().numpy()

        depth_fake_resized = upscale_depth_img(depth_fake)
        plt.imsave(folder + f'/y_model_{epoch}.png', depth_fake_resized, cmap=configs.CMAP_STYLE)

    # model.train()


def evaluate_model(model,eval_dataloader):
    model.eval()

    preds = np.zeros((configs.IMAGE_SIZE, configs.IMAGE_SIZE, len(eval_dataloader.dataset)), dtype=np.float32)
    labels = np.zeros((configs.IMAGE_SIZE, configs.IMAGE_SIZE, len(eval_dataloader.dataset)), dtype=np.float32)

    # Starting evaluation ...
    for i,sample_test  in enumerate (eval_dataloader):
        img = torch.autograd.Variable(sample_test['image'].cuda())
        label = torch.autograd.Variable(DepthNorm(sample_test['depth']))

        # inference
        pred_depth= model(img)
        pred_depth = F.relu(pred_depth).detach().cpu().numpy()
        
        # up-sampling
        pred_resized = upscale_depth_img(pred_depth)

        # store the label and the corresponding prediction
        labels[:, :, i] = label
        preds[:, :, i] = pred_resized

    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(labels, preds)
    print('Abs_Rel: {:.6f}, Sq_Rel: {:.6f}, RMSE: {:.6f}, RMSE_Log: {:.6f}'.format(abs_rel, sq_rel, rmse, rmse_log))
    print('Accuracy Metrics a1: {:.6f}, a2: {:.6f}, a3: {:.6f}\n'.format(a1, a2, a3))

    # save_some_examples(model, eval_dataloader, epoch)

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


def save_checkpoint(model, optimizer, filename='my_checkpoint.pth.tar', message = '=> Saving checkpoint'):
    print(message)
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, message = '=> Loading checkpoint'):
    print(message)
    checkpoint = torch.load(checkpoint_file, map_location=configs.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr