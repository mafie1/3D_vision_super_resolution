import h5py
import numpy as np
import os
import io
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io

"""Datasets for H5 """
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class BSDS500(Dataset):
    """Dataset from the Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500)"""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class DIV2K(Dataset):
    """Dataset from the Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500)"""
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            here: "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/3D_vision_super_resolution/data/DIV2K_valid_HR"
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        #list = os.listdir(self.root_dir)  # dir is your directory path
        onlyfiles = sorted(next(os.walk(self.root_dir))[2])  # dir is your directory path as string
        number_files = len(onlyfiles)
        return number_files


    def __getitem__(self, idx, show_image = False):
        onlyfiles = sorted(next(os.walk(self.root_dir))[2])
        img_name = onlyfiles[idx]
        img_path = self.root_dir + '/' + img_name

        #Reading the File as PIL Image
        img = Image.open(img_path)
        print(type(img))

        #Reading using scikit-image
        img = io.imread(img_path)
        print(type(img))

        if show_image:
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            plt.show()

        if self.transform is not None:
            img = self.transform(img)

        return img


if __name__ == '__main__':
    dataset_valid = DIV2K(root_dir="/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/3D_vision_super_resolution/data/DIV2K_valid_HR")
    print(len(dataset_valid))
    dataset_valid.__getitem__(idx=1, show_image=True)
