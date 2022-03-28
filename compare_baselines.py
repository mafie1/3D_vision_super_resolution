import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.model_selection import train_test_split
from metrics import ssim, calc_ssim, _psnr, calc_psnr
from utils import display_tensor
from datasets import TrainDatasetH5, EvalDatasetH5, BSD100
from models import SRCNN, VDSR
from transformations import SIMPLE_TRANSFORM, MINIMALIST_TRANSFORM
from utils import set_all_seeds, AverageMeter
from metrics import _psnr, _ssim, calc_ssim, calc_psnr

from skimage.transform import rotate as rotate
from skimage import io

class BSD100_comp(Dataset):
    """Description"""

    def __init__(self, root_dir=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root_dir  # '/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data_SR/BSDS100/images/'
        self.transform = transform

    def __len__(self):
        onlyfiles = sorted(next(os.walk(self.root))[2])  # dir is your directory path as string
        number_files = len(onlyfiles)
        return int(number_files * 0.5)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        onlyfiles = sorted(next(os.walk(self.root))[2])
        onlyfiles_HR = onlyfiles[0::2]
        onlyfiles_LR = onlyfiles[1::2]

        img_path_HR = self.root + '/' + onlyfiles_HR[idx]
        img_path_LR = self.root + '/' + onlyfiles_LR[idx]

        img_HR = io.imread(img_path_HR)
        img_LR = io.imread(img_path_LR)

        # make sure all images are horzontally aligned
        if img_HR.shape[0] > img_HR.shape[1]:
            img_HR = rotate(img_HR, angle=90, resize=True, center=None, order=None, mode='constant',
                            clip=True, preserve_range=False)
            img_LR = rotate(img_LR, angle=90, resize=True, center=None, order=None, mode='constant',
                            clip=True, preserve_range=False)
        else:
            img_HR = rotate(img_HR, angle=0, resize=True, center=None, order=None, mode='constant',
                            clip=True, preserve_range=False)
            img_LR = rotate(img_LR, angle=0, resize=True, center=None, order=None, mode='constant',
                            clip=True, preserve_range=False)


        if self.transform is not None:
            transform_seed = np.random.randint(0, 10000)

            set_all_seeds(transform_seed)
            img_HR = self.transform(img_HR)

            set_all_seeds(transform_seed)
            img_LR = self.transform(img_LR)
        return img_LR, img_HR


def compare_non_parametric():
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    #path = '/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data_SR/BSD100_all/image_SRF_2_nearest'

    link_folder_name = 'image_SRF_2_SRCNN'
    link = os.path.abspath(os.path.join(__file__, f'../data_SR/BSD100_all/{link_folder_name}'))
    dataset = BSD100_comp(root_dir=link, transform=MINIMALIST_TRANSFORM)
    print(len(dataset))

    TRAIN_DATASET, EVAL_DATASET = train_test_split(dataset, test_size=0.1, random_state=SEED)
    print('Train-Eval-Split is done. \nThere are x images in the \nTraining Set: {}\nEval Set: {}'.format(
        len(TRAIN_DATASET), len(EVAL_DATASET)))

    eval_dataloader = DataLoader(dataset=EVAL_DATASET, batch_size=1)
    psnr_baseline = AverageMeter()
    SSIM_baseline = AverageMeter()

    for data in eval_dataloader:
        inputs, labels = data
        inputs = inputs
        labels = labels

        psnr_baseline.update(calc_psnr(inputs, labels), len(inputs))
        SSIM_baseline.update(calc_ssim(inputs, labels), len(inputs))

    print(SSIM_baseline.avg)
    print(psnr_baseline.avg)

def count_parameters():
    #model = SRCNN(num_channels=3)
    model = VDSR(num_channels=3, d=20)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(params)

def show_results_in_plot():
    scales = [2,3,4]

    bicubic_PSNR = [28.34, 25.86, 24.76,]
    bicubic_SSIM = [0.83, 0.72, 0.63 ]

    NN_PSNR = [27.09, 24.82, 23.75]
    NN_SSIM = [0.81, 0.68, 0.59]

    SRCNN_PSNR = [28.57, 26.17, 24.84]
    SRCNN_SSIM = [0.85, 0.74, 0.65]

    VDSR_PSNR = [29.47, 26.58, 25.26]
    VDSR_SSIM = [0.87, 0.75, 0.66]

    plt.figure(figsize=(8, 6))

    color = 'tab:blue'
    plt.plot(scales, bicubic_PSNR, marker = 'x', label = 'Bicubic')
    plt.plot(scales, NN_PSNR, label='Nearest Neighbor', marker = '*')
    plt.plot(scales, SRCNN_PSNR, marker = 's', label='SRCNN')
    plt.plot(scales, VDSR_PSNR, marker='P', label='VDSR')
    plt.ylabel('PSNR', size = 'large')  # we already handled the x-label with ax1
    plt.xlabel('Scale Factor', size = 'large')
    plt.xticks([2, 3, 4])
    plt.legend()
    plt.grid()
    plt.savefig('PSNR.png', dpi = 500)
    plt.show()

    plt.figure(figsize=(8,6))
    plt.plot(scales, bicubic_SSIM, marker = 'x', label = 'Bicubic')
    plt.plot(scales, NN_SSIM, marker = 'o', label = 'Nearest Neighbor')
    plt.plot(scales, SRCNN_SSIM, marker = 's', label = 'SRCNN')
    plt.plot(scales, VDSR_SSIM, marker = 'P', label = 'VDSR')
    plt.ylabel('SSIM', size = 'large')
    plt.xlabel('Scale Factor', size = 'large')
    plt.xticks([2, 3, 4])
    plt.grid()
    plt.legend()
    plt.savefig('SSIM.png', dpi=200)
    plt.show()


def get_pictures(dataset=None, weights=None):
    SEED = 0
    CHANNELS = 3
    DEVICE = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    #link = '/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data_SR/91-Images/91-image_x2.h5'
    #SD100_dataset = TrainDatasetH5(link)
    link_folder_name = 'image_SRF_3'
    link = os.path.abspath(os.path.join(__file__, f'../data_SR/BSD100/{link_folder_name}'))
    dataset = BSD100_comp(root_dir=link, transform=None)
    BSD100_dataset = BSD100(link)
    print(len(BSD100_dataset))

    lr_image, hr_image = BSD100_dataset.__getitem__(6)

    """Load Model:"""
    model = SRCNN(num_channels=CHANNELS).to(DEVICE).double()
    #model = VDSR(num_channels=CHANNELS, d=5).to(DEVICE).double()
    PATH = 'outputs/SRCNN-BSD100-X3/best.pth'
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()

    MINIMALIST_TRANSFORM = transforms.Compose([
        transforms.ToTensor()])

    #input = MINIMALIST_TRANSFORM(torch.from_numpy(lr_image).unsqueeze(0))
    input = torch.from_numpy(lr_image).unsqueeze(0).double().permute(0,3,2,1)
    print(input.shape)
    pred = model(input)
    print(pred.shape)

    plt.figure(figsize = (10,10))
    plt.imshow(lr_image.squeeze(), cmap = 'gray')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(hr_image.squeeze(), cmap = 'gray')
    plt.title('HR Image')
    plt.show()

    plt.figure(figsize=(10, 10))
    display_tensor(pred)
    #plt.imshow(pred)
    #plt.show()

    #link_folder_name = 'image_SRF_2'
    #link = os.path.abspath(os.path.join(__file__, f'../data_SR/BSD100/{link_folder_name}'))

    #link = '/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data_SR/BSD100/image_SRF_2'
    #BSD100_dataset = BSD100(root_dir=link)
    #print(len(BSD100_dataset))
    #dataset_small = Subset(dataset,np.arange(len(BSD100_dataset)))
    #CHANNELS = 3

    #image, label = BSD100_dataset.__getitem__(idx=0)

    #TRAIN_DATASET, EVAL_DATASET = train_test_split(dataset_small, test_size=0.1, random_state=SEED)

    #print('Train-Eval-Split is done. \nThere are x images in the \nTraining Set: {}\nEval Set: {}'.format(
     #   len(TRAIN_DATASET), len(EVAL_DATASET)))

    #image, label = EVAL_DATASET.__getitem__(0)

    #plt.imshow(image)
    #plt.show()

    pass

if __name__ == '__main__':
    #compare_non_parametric()
    #count_parameters()
    #show_results_in_plot()
    get_pictures()


