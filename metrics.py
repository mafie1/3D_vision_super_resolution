import math
import numpy as np
import torch.nn as nn
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio

def calc_psnr(img1, img2):
    """Implmentation of PSNR for torch tensors"""
    return 10. * torch.log10(1.0 / torch.mean((img1- img2) ** 2))

def PSNR(pred, gt):
    """Implementation of PSNR for images as numpy arrays"""
    return peak_signal_noise_ratio(pred, gt, data_range=pred.max()-pred.min())

def SSIM(pred, gt):
    """Implementation of the Structural Similarity Index for images as numpy arrays """
    return ssim(pred, gt, data_range=pred.max() - pred.min(), channel_axis=2)

def calc_ssim(pred, gt):
    """SSIM Implementation for torch tensors"""
    pred = pred.squeeze(0).cpu().detach().numpy()
    gt = gt.squeeze(0).cpu().detach().numpy()
    return ssim(pred, gt, data_range=pred.max() - pred.min(), channel_axis=0)


def MSE(pred, gt):
    #assert size is the same
    #assert pred.mode == pred.mode, "Different kinds of images."
    #assert pred.size == pred.size, "Different sizes."
    assert pred.shape == gt.shape, "Different image sizes"
    return mean_squared_error(pred, gt)


def weighted_loss(original, compressed):
    '''
    Original and compressed are torch tensors
    0.4 = MSE
    0.5 = PSNR
    0.1 = SSIM
    '''
    mseLoss = nn.MSELoss()
    mse = mseLoss(original, compressed)

    psnr_score = PSNR(original, compressed)
    # PSNR is maximized so 100-PSNR is a loss function (Or -PSNR)
    psnr_loss = 100 - psnr_score

    ssim_score = SSIM(original, compressed)
    ssim_loss = 1 - ssim_score.item()

    weighted_loss = (0.4 * mse) + (0.5 * (psnr_loss / 100)) + (0.1 * ssim_loss)
    return weighted_loss


if __name__ == '__main__':
    from datasets import TrainDatasetH5
    import matplotlib.pyplot as plt

    TRAIN_FILE = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data/Set5/91-image_x2.h5"
    dataset_h5 = TrainDatasetH5(TRAIN_FILE)

    image, label = dataset_h5.__getitem__(1)

    plt.imshow(image[0,:,:], cmap='gray')
    plt.show()

    plt.imshow(label[0], cmap='gray')
    plt.show()

    gt = label[0,:,:]
    pred = image[0,:,:]

    print(SSIM(pred, gt))
    print(SSIM(gt, gt))

    print(PSNR(pred, gt), 'dB')
    print(PSNR(gt, gt), 'dB')

    print(MSE(pred, gt))
    print(MSE(gt, gt))

    print(weighted_loss(pred, gt))

