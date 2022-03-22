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


"""Datasets for H5 """
class TrainDatasetH5(Dataset):
    """Train Dataset for H% and 92"""
    def __init__(self, h5_file):
        super(TrainDatasetH5, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDatasetH5(Dataset):
    def __init__(self, h5_file):
        super(EvalDatasetH5, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class BSDS500(Dataset):
    """Dataset from the Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500)"""
    def __init__(self, root_dir=None, transform=None, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            train:
        """
        if root_dir:
            self.root = root_dir + mode
        else:
            self.root = '/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data/BSR/BSDS500/images/'+mode

        print(self.root)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        onlyfiles = sorted(next(os.walk(self.root))[2])  # dir is your directory path as string
        number_files = len(onlyfiles)
        return number_files

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        onlyfiles_HR = sorted(next(os.walk(self.root))[2])
        img_path_HR = self.root + '/' + onlyfiles_HR[idx]
        img_HR = io.imread(img_path_HR)

        img_LR = img_HR #to do: downsample bicubic


        if self.transform is not None:
            img_HR = self.transform(img_HR)
            img_LR = self.transform(img_LR)

        print('SIZE HR image:', img_HR.shape)
        print('SIZE LR image:', img_LR.shape)
        return img_LR, img_HR

       # img_name = os.path.join(self.root_dir,
       #                        self.landmarks_frame.iloc[idx, 0])
        #image = io.imread(img_name)
        #landmarks = self.landmarks_frame.iloc[idx, 1:]
        #landmarks = np.array([landmarks])
        #landmarks = landmarks.astype('float').reshape(-1, 2)
        #sample = {'image': image, 'landmarks': landmarks}

        if self.transform is not None:
            sample = self.transform(sample)

        print(sample.shape())
        return sample



if __name__ == '__main__':
    #ROOT_labels= "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data/DIV2K_train_HR"
    #ROOT_images = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data/DIV2K_train_LR_bicubic/X2"
    #dataset_valid = DIV2K(root_HR=ROOT_labels, root_LR = ROOT_images)
    #LR_image, HR_image = dataset_valid.__getitem__(idx=2)
    #plt.imshow(LR_image)
    #plt.show()
    #plt.imshow(HR_image)
    #plt.show()


    TRAIN_FILE = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data/Set5/91-image_x4.h5"
    dataset_h5 = TrainDatasetH5(TRAIN_FILE)
    image, label = dataset_h5.__getitem__(1)

    plt.imshow(image[0,:,:], cmap='gray', interpolation='bicubic')
    plt.show()
    plt.imshow(label[0,:,:], cmap='gray', interpolation='bicubic')
    plt.show()

    print(image.shape)
    print(label.shape)

    #dataset = BSDS500(mode='test')
    #print(len(dataset))

    #image, label = dataset.__getitem__(1)




