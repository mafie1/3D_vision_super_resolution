import torch
import numpy as np
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from metrics import _ssim, calc_ssim, _psnr, calc_psnr
from utils import display_tensor
from utils import AverageMeter
from datasets import TrainDatasetH5, EvalDatasetH5, BSD100
from transformations import SIMPLE_TRANSFORM, MINIMALIST_TRANSFORM



if __name__ == '__main__':
    SEED = 0
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 50
    BATCH_SIZE = 4

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    """Set 5 Dataset --> use for example as validatation set"""
    # TRAIN_FILE = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data_SR/Set5/Set5_x2.h5"
    # dataset_h5 = TrainDatasetH5(TRAIN_FILE)
    # dataset_small = Subset(dataset_h5, np.arange(5))
    # CHANNELS = 1

    # """91-Images Dataset"""
    train_file_name = '91-image_x2.h5'
    TRAIN_FILE = os.path.abspath(os.path.join(__file__, f'../data_SR/91-Images/{train_file_name}'))
    dataset_h5 = TrainDatasetH5(TRAIN_FILE)
    dataset_small = Subset(dataset_h5, np.arange(len(dataset_h5)))
    CHANNELS = 1

    """BSD100 Dataset"""
    #link_folder_name = 'image_SRF_2'
    #link = os.path.abspath(os.path.join(__file__, f'../data_SR/BSD100/{link_folder_name}'))
    #dataset_h5 = BSD100(root_dir=link, scale=2, transform=MINIMALIST_TRANSFORM)
    # print(len(BSD100_dataset))
    #dataset_small = Subset(dataset_h5, np.arange(10))
    #CHANNELS = 3

    TRAIN_DATASET, EVAL_DATASET = train_test_split(dataset_small, test_size=0.1, random_state=SEED)
    print('Train-Eval-Split is done. \nThere are x images in the \nTraining Set: {}\nEval Set: {}'.format(
        len(TRAIN_DATASET), len(EVAL_DATASET)))
