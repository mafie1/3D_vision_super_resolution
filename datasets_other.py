import h5py
import numpy as np
import os
import io
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean

class DIV2K(Dataset):
    """Dataset from the Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500) with bicubic 2x downsampling"""
    def __init__(self, root_HR, root_LR, transform=None):
        """
        Args:
            root_HR (string): Directory path with the original HR images. Those are the labels
                            here: "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/3D_vision_super_resolution/data/DIV2K_valid_HR"
            root_LR (string): Directory path with the low-resolution (downnsampled) images.
                            Different downsampling methods are available (bicubic, ...)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir_HR = root_HR
        self.root_dir_LR = root_LR
        self.transform = transform

    def __len__(self):
        #list = os.listdir(self.root_dir)  # dir is your directory path
        onlyfiles = sorted(next(os.walk(self.root_dir_HR))[2])  # dir is your directory path as string
        number_files = len(onlyfiles)
        return number_files

    def __getitem__(self, idx):
        onlyfiles_HR = sorted(next(os.walk(self.root_dir_HR))[2])

        img_path_HR = self.root_dir_HR + '/' + onlyfiles_HR[idx]
        img_path_LR = self.root_dir_LR + '/' + onlyfiles_HR[idx][:-4] + 'x2.png'

        #Reading the File as PIL Image
        #img_HR = Image.open(img_path_HR)
        #img_LR = Image.open(img_path_LR)
        #print(type(img_HR))

        #Reading using scikit-image
        img_HR = io.imread(img_path_HR) #full-sized image in HR
        img_LR = io.imread(img_path_LR)
        #print(type(img_HR))

        #TO DO: standardize size of all LR and HR images or crop
        #H, W, C = img_HR.shape
        #img_LR = resize(img_LR, output_shape= (H,W,C), order = 0)

        if self.transform is not None:
            img_HR = self.transform(img_HR)
            img_LR = self.transform(img_LR)

        print('SIZE HR image:', img_HR.shape)
        print('SIZE LR image:', img_LR.shape)
        return img_LR, img_HR


class domain_transfer_DS(Dataset): #unfinished
    """Dataset from the Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500) with bicubic 2x downsampling"""
    def __init__(self, root_HR, root_LR, transform=None, domain1=True):
        """
        Args:
            root_HR (string): Directory path with the original HR images. Those are the labels.
            root_LR (string: Directory path with low resolution images

            domain1 (bool): True for domain 1 (e.g. Dogs), False for domain 2 (e.g.Cars).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir_HR = root_HR
        self.root_dir_LR = root_LR
        self.domain_one = domain1

    def __len__(self):
        #list = os.listdir(self.root_dir)  # dir is your directory path
        onlyfiles = sorted(next(os.walk(self.root_dir_HR))[2])  # dir is your directory path as string
        number_files = len(onlyfiles)
        return number_files

    def __getitem__(self, idx):
        #if self.domain_one:
         #   pass

        onlyfiles_HR = sorted(next(os.walk(self.root_dir_HR))[2])

        img_path_HR = self.root_dir_HR + '/' + onlyfiles_HR[idx]
        img_path_LR = self.root_dir_LR + '/' + onlyfiles_HR[idx][:-4] + 'x2.png'

        #Reading the File as PIL Image
        #img_HR = Image.open(img_path_HR)
        #img_LR = Image.open(img_path_LR)
        #print(type(img_HR))

        #Reading using scikit-image
        img_HR = io.imread(img_path_HR) #full-sized image in HR
        img_LR = io.imread(img_path_LR)
        #print(type(img_HR))

        #TO DO: standardize size of all LR and HR images or crop
        #H, W, C = img_HR.shape
        #img_LR = resize(img_LR, output_shape= (H,W,C), order = 0)

        if self.transform is not None:
            img_HR = self.transform(img_HR)
            img_LR = self.transform(img_LR)

        print('SIZE HR image:', img_HR.shape)
        print('SIZE LR image:', img_LR.shape)
        return img_LR, img_HR


if __name__ == '__main__':
    ROOT_labels = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data/DIV2K_train_HR"
    ROOT_images = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data/DIV2K_train_LR_bicubic/X2"
    dataset_valid = DIV2K(root_HR=ROOT_labels, root_LR=ROOT_images)
    LR_image, HR_image = dataset_valid.__getitem__(idx=2)
    plt.imshow(LR_image)
    plt.show()
    plt.imshow(HR_image)
    plt.show()




    """ if only head or backbone shall be trained, use a version of this code snippet
    optimizer = optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), 'lr': LEARNING_RATE * 0.1}
        ], lr=LEARNING_RATE)
    """