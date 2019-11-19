import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.models as models
import torch

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
def get_im(dset, idx):
    im = dset_val[idx][0]
    im_np = deepcopy(im.numpy()).transpose((1, 2, 0))
    im_torch = normalize(im).unsqueeze_(0)
    return im_np, im_torch

# convert im_torch back to unnormalized numpy im
# 1 x 3 x 224 x 224 -> 224 x 224 x 3
def im_to_np(im_torch):
    means = np.array([0.485/0.229, 0.456/0.224, 0.406/0.255]).T
    stds = np.array([0.229, 0.224, 0.255]).T
    im_np = deepcopy(im_torch.cpu().detach().numpy()[0]).transpose((1, 2, 0))
    im_np +=  means
    im_np *=  stds
    return im_np

def zero_one(im):
    return (im - np.min(im)) / (np.max(im) - np.min(im))

# show an image
# if given list of images, show them all
def show(im, dpi=100, center_crop=False):
    plt.figure(dpi=dpi)
    if type(im) == list:
        plt.figure(figsize=(3 * len(im), 3), dpi=dpi)
        for i in range(len(im)):
            plt.subplot(1, len(im), i + 1)
            im[i] = np.array(im[i])
            im[i] = im[i].squeeze()
            plt.imshow(zero_one(im[i]))
            plt.grid(False)
            plt.axis('off')   
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.tight_layout()
    else:
        if center_crop:
            mid = im.data.shape[0] // 2
            low = mid - center_crop // 2
            high = mid + center_crop // 2
            im = im[low: high, low: high]
        plt.imshow(zero_one(im))
        plt.grid(False)
        plt.axis('off')        