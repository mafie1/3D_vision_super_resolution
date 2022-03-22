import h5py
import numpy as np
import os
import io
import torch
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import rescale as rescale  # , resize, downscale_local_mean
from skimage.transform import rotate as rotate

from utils import calc_psnr, set_all_seeds
from metrics import PEAK_SIGNAL_TO_NOISE, _ssim

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


class BSD100(Dataset):
    """Description"""

    def __init__(self, root_dir=None, transform=None, scale=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.scale = scale
        self.root = root_dir  # '/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data/BSR/BSDS500/images/'
        self.transform = transform

    def __len__(self):
        only_files = sorted(next(os.walk(self.root))[2])  # dir is your directory path as string
        number_files = len(only_files)
        return int(number_files * 0.5)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        only_files = sorted(next(os.walk(self.root))[2])
        only_files_hr = only_files[0::2]
        only_files_lr = only_files[1::2]

        img_path_hr = self.root + '/' + only_files_hr[idx]
        img_path_lr = self.root + '/' + only_files_lr[idx]

        img_hr = io.imread(img_path_hr)
        img_lr = io.imread(img_path_lr)

        # make sure all images are horizontally aligned
        if img_hr.shape[0] > img_hr.shape[1]:
            img_hr = rotate(img_hr, angle=90, resize=True, center=None, order=None, mode='constant',
                            clip=True, preserve_range=False)
            img_lr = rotate(img_lr, angle=90, resize=True, center=None, order=None, mode='constant',
                            clip=True, preserve_range=False)
        else:
            img_hr = rotate(img_hr, angle=0, resize=True, center=None, order=None, mode='constant',
                            clip=True, preserve_range=False)
            img_lr = rotate(img_lr, angle=0, resize=True, center=None, order=None, mode='constant',
                            clip=True, preserve_range=False)

        """Upsample Low Resolution Image"""
        if self.scale is not None:
            img_lr_upsampled = rescale(img_lr, self.scale, order=3, channel_axis=2)  # bicubic
            img_lr = img_lr_upsampled
            assert img_hr.shape == img_lr.shape

        if self.transform is not None:
            transform_seed = np.random.randint(0, 10000)

            set_all_seeds(transform_seed)
            img_hr = self.transform(img_hr)

            set_all_seeds(transform_seed)
            img_lr = self.transform(img_lr)

        return img_lr, img_hr


def test_h5():
    train_file = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data_own/Set5/91-image_x4.h5"
    dataset_h5 = TrainDatasetH5(train_file)
    print(len(dataset_h5))
    image, label = dataset_h5.__getitem__(0)

    plt.imshow(image[0, :, :], cmap='gray', interpolation='bicubic')
    plt.title('Low Resolution Image')
    plt.show()

    plt.imshow(label[0, :, :], cmap='gray', interpolation='bicubic')
    plt.title('High Resolution Image')
    plt.show()

    print(image.shape)
    print(label.shape)


def test_bsd100():
    link = '/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data_github/BSD100/image_SRF_2'
    bsd100_dataset = BSD100(root_dir=link, scale=2)
    print(len(bsd100_dataset))

    psnes = np.ones(5)
    ssims = np.ones(5)

    for i in range(0, 5):
        lr_image, hr_image = bsd100_dataset.__getitem__(i + 1)

        plt.imshow(lr_image)
        plt.title('Low Resolution')
        plt.show()
        plt.imshow(hr_image)
        plt.title('High Resolution')
        plt.show()

        lr_image = torch.from_numpy(lr_image)
        hr_image = torch.from_numpy(hr_image)

        # print(calc_psnr(lr_image, hr_image))

        lr_image = lr_image.squeeze(0).cpu().detach().numpy()
        hr_image = hr_image.squeeze(0).cpu().detach().numpy()

        psnes[i] = PEAK_SIGNAL_TO_NOISE(lr_image, hr_image)
        ssims[i] = _ssim(lr_image, hr_image)

        print(PEAK_SIGNAL_TO_NOISE(lr_image, hr_image))
        print(_ssim(lr_image, hr_image))

    plt.plot(psnes)
    plt.plot(ssims)
    plt.show()


if __name__ == '__main__':
    # test_h5()
    test_bsd100()

    # dataset = BSDS500(mode='test')
    # print(len(dataset))

    # image, label = dataset.__getitem__(1)
