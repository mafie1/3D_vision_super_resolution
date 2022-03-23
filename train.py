import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from models import SRCNN, VDSR
from metrics import _ssim, calc_ssim, _psnr, calc_psnr
from utils import display_tensor
from utils import AverageMeter
from datasets import TrainDatasetH5, EvalDatasetH5, BSD100
from transformations import SIMPLE_TRANSFORM, MINIMALIST_TRANSFORM


# in command line:go to the directory of the project, activate environment and run tensorboard --logdir=runs

def train_function(model, criterion, optimizer,
                   train_dataset, eval_dataset,
                   num_epochs, batch_size, num_workers,
                   OUT_DIR):
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    writer = SummaryWriter('runs/training/VDSR')

    images, labels = iter(train_dataloader).next()
    img_grid = torchvision.utils.make_grid(torch.cat((images[0:2], labels[0:2]), 0), nrow=2, padding=2)

    writer.add_image('Starting Point: Sample Pairs of HR-LR Images', img_grid)
    writer.add_graph(model.to(DEVICE), images.to(DEVICE))

    best_epoch = 0  # at the start, the best epoch is the first
    best_psnr = 0.0  # best Peak-Signal-To-Noise Ration across all epochs

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = AverageMeter()  # this is the (MSE) loss for all epochs

        epoch_psnr = AverageMeter()  # this is the PSNR score for the eval set
        epoch_psnr_baseline = AverageMeter()

        epoch_SSIM = AverageMeter()
        epoch_SSIM_baseline = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description('Epoch: {}/{}; Batches in Progress'.format(epoch, num_epochs - 1))

            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # inputs = inputs.to(DEVICE).float()
                # labels = labels.to(
                #     DEVICE).float()  # labels are high-resolution ground truth, float type is 32bit, double is 64bit floating point number

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                # Tensorboard loss-tracking
                # writer.add_scalars('data/losses_scalar_group',
                #                  {'Iteration Loss': epoch_losses.val, 'Iteration Loss Average': epoch_losses.avg} ,
                #                 epoch*len(train_dataloader)+i)
                # writer.add_scalar()
                # writer.add_scalars('data/scalar_group', {'Iteration Loss Average': epoch_losses.avg}, epoch*len(train_dataloader)+i)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.val))
                t.update(len(inputs))

        writer.add_scalar('training loss', epoch_losses.avg / len(train_dataloader), epoch)

        torch.save(model.state_dict(), os.path.join(OUT_DIR, 'epoch_{}.pth'.format(epoch + 1)))

        """Evaluation after each training epoch"""
        model.eval()

        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # print(torch.max(inputs))
            # print(torch.max(labels))

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
                # print(torch.max(preds))

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))
            epoch_psnr_baseline.update(calc_psnr(inputs, labels), len(inputs))

            epoch_SSIM.update(calc_ssim(preds, labels), len(inputs))
            epoch_SSIM_baseline.update(calc_ssim(inputs, labels), len(inputs))

            # writer.add_scalar('PSNR',
            #                 epoch_psnr.val,
            #                 epoch * len(train_dataloader) + i)
            # plt.imshow(preds.squeeze(0).permute(1,2,0).cpu().detach().numpy())
            # plt.show()

        img_grid_end = torchvision.utils.make_grid(torch.cat((inputs, labels, preds), 0), nrow=3)
        writer.add_image('Images after {} epochs: LR, HR, Prediction '.format(epoch), img_grid_end)
        writer.add_scalar('Eval PSNR', epoch_psnr.avg / len(eval_dataloader), epoch)
        writer.add_scalar('Baseline PSNR', epoch_psnr_baseline.avg / len(eval_dataloader), epoch)
        writer.add_scalar('Eval SSIM', epoch_SSIM.avg / len(eval_dataloader), epoch)
        writer.add_scalar('Baseline SSIM', epoch_SSIM_baseline.avg / len(eval_dataloader), epoch)

        print('Eval PSNR: {:.2f}dB'.format(epoch_psnr.avg))
        print('Eval PSNR on baseline: {:.2f}dB'.format(epoch_psnr_baseline.avg))

        print('Eval SSIM: {:.2f}'.format(epoch_SSIM.avg))
        print('Eval SSIM on baseline: {:.2f}'.format(epoch_SSIM_baseline.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    """End of Training Statistics & Save Best Model"""
    print('best epoch: {}, psnr: {:.2f}dB'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(OUT_DIR, 'best.pth'))

    img_grid_end = torchvision.utils.make_grid(torch.cat((inputs, labels, preds), 0), nrow=3, padding=2)
    writer.add_image('End of Training: LR, HR, Prediction ', img_grid_end)
    writer.close()


if __name__ == '__main__':
    SEED = 0
    LEARNING_RATE = 5e-4
    NUM_EPOCHS = 4
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    """Set 5 Dataset --> use for example as validatation set"""
    # TRAIN_FILE = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/super_resolution/data_SR/Set5/Set5_x2.h5"
    # dataset_h5 = TrainDatasetH5(TRAIN_FILE)
    # dataset_small = Subset(dataset_h5, np.arange(5))
    # CHANNELS = 1

    # """91-Images Dataset"""
    # train_file_name = '91-image_x4.h5'
    # TRAIN_FILE = os.path.abspath(os.path.join(__file__, f'../data_SR/91-Images/{train_file_name}'))
    # dataset_h5 = TrainDatasetH5(TRAIN_FILE)
    # dataset_small = Subset(dataset_h5, np.arange(1000))
    # CHANNELS = 1

    """BSD100 Dataset"""
    link_folder_name = 'image_SRF_2'
    link = os.path.abspath(os.path.join(__file__, f'../data_SR/BSD100/{link_folder_name}'))
    dataset_h5 = BSD100(root_dir=link, scale=2, transform=MINIMALIST_TRANSFORM)
    # print(len(BSD100_dataset))
    dataset_small = Subset(dataset_h5, np.arange(10))
    CHANNELS = 3

    TRAIN_DATASET, EVAL_DATASET = train_test_split(dataset_h5, test_size=0.1, random_state=SEED)
    print('Train-Eval-Split is done. \nThere are x images in the \nTraining Set: {}\nEval Set: {}'.format(
        len(TRAIN_DATASET), len(EVAL_DATASET)))

    OUT_DIR = "outputs/SRCNN-BSD100-X4"
    MODEL = SRCNN(num_channels=CHANNELS).to(DEVICE).double()  # num_channels = 1 for gray scale images, 3 for color images
    # MODEL = VDSR(num_channels=CHANNELS, d=4).to(DEVICE).double()  # default is three channel (e.g. RGB) images

    OPTIMIZER = optim.Adam(MODEL.parameters(),
                           lr=LEARNING_RATE)  # all training, later: train head and backbone separate
    # OPTIMIZER = optim.SGD(MODEL.parameters(), lr=LEARNING_RATE)

    # CRITERION = nn.MSELoss() #Mean-Squared-Error Loss = L2 loss
    CRITERION = nn.L1Loss().to(DEVICE)  # L1 Loss

    train_function(model=MODEL,
                   criterion=CRITERION,
                   optimizer=OPTIMIZER,
                   train_dataset=TRAIN_DATASET, eval_dataset=EVAL_DATASET,
                   num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE,
                   num_workers=NUM_WORKERS,
                   OUT_DIR=OUT_DIR)
