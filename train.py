import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from tqdm import tqdm
import os
import copy
import numpy as np
from sklearn.model_selection import train_test_split

from models import SRCNN, VDSR
from utils import display_tensor
from utils import AverageMeter, calc_psnr
from datasets import TrainDatasetH5, EvalDatasetH5, DIV2K


#in command line: tensorboard --logdir=runs

def train_function(model, criterion, optimizer,
                   train_dataset, eval_dataset,
                   num_epochs, batch_size, num_workers,
                   OUT_DIR, transform = None):

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)

    eval_dataloader = DataLoader(dataset=EVAL_DATASET, batch_size=1)



    writer = SummaryWriter('runs/training')
    dataiter = iter(train_dataloader)
    images, labels = dataiter.next()
    img_grid = torchvision.utils.make_grid(images)

    writer.add_image('sample images', img_grid)
    writer.add_graph(model, images)


    running_loss = 0.0
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, num_epochs - 1))

            for i, data in enumerate(train_dataloader,0):
                inputs, labels = data

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE) #labels are high-resolution ground truth

                preds = model(inputs)
                loss = criterion(preds, labels)

                epoch_losses.update(loss.item(), len(inputs))

                #writer.add_scalar('Iteration Loss ',
                 #                 epoch_losses.val,
                  #                epoch * len(train_dataloader) + i)
                #writer.add_scalar('Iteration Loss Average',
                 #                 epoch_losses.avg, #epoch_losses.val
                  #                epoch * len(train_dataloader) + i)

                writer.add_scalars('data/losses_scalar_group', {'Iteration Loss': epoch_losses.val, 'Iteration Loss Average': epoch_losses.avg} , epoch*len(train_dataloader)+i)
                #writer.add_scalars('data/scalar_group', {'Iteration Loss Average': epoch_losses.avg}, epoch*len(train_dataloader)+i)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(inputs))

        torch.save(model.state_dict(), os.path.join(OUT_DIR, 'epoch_{}.pth'.format(epoch+1)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        writer.add_scalar('PSNR',
                          epoch_psnr.val,
                          epoch * len(train_dataloader) + i)

        print('eval psnr: {:.2f}'.format(epoch_psnr.avg))
        writer.add_scalar('Epoch PSNR',
                          epoch_psnr.avg,
                          epoch)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())


    """
    writer.add_figure()
    """

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(OUT_DIR, 'best.pth'))
    writer.close()


if __name__ == '__main__':

    SEED = 0
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 2
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(SEED)
    np.random.seed(SEED)


    TRAIN_FILE = "data/91-image_x2.h5"
    EVAL_FILE = "data/Set5_x2.h5"

    TRAIN_DATASET = TrainDatasetH5(TRAIN_FILE) #add transformations!
    EVAL_DATASET = EvalDatasetH5(EVAL_FILE)

    OUT_DIR = "outputs"

    #MODEL = SRCNN(num_channels=1).to(DEVICE)  #num_channels = 1 for gray scale images
    MODEL = VDSR(num_channels=1 , d=4).to(DEVICE)  #defaul is three channel (e.g. RGB) images

    OPTIMIZER = optim.Adam(MODEL.parameters(),
                           lr=LEARNING_RATE)  # all training, later: train head and backbone separate

    """ if only head or backbone shall be trained, use a version of this code snippet
    optimizer = optim.Adam([
            {'params': model.conv1.parameters()},
            {'params': model.conv2.parameters()},
            {'params': model.conv3.parameters(), 'lr': LEARNING_RATE * 0.1}
        ], lr=LEARNING_RATE)
    """

    CRITERION = nn.MSELoss() #Mean-Squared-Error Loss


    """ROOT_labels = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/3D_vision_super_resolution/data/DIV2K_train_HR"
    ROOT_images = "/Users/luisaneubauer/Documents/WS 2021:22/3D Reconstruction/3D_vision_super_resolution/data/DIV2K_train_LR_bicubic/X2"
    TRAIN_DATASET = DIV2K(root_labels=ROOT_labels, root_images=ROOT_images)#[0:600]
    EVAL_DATASET = ... """



    train_function(model = MODEL,
                   criterion = CRITERION,
                   optimizer = OPTIMIZER,
                   train_dataset = TRAIN_DATASET, eval_dataset = EVAL_DATASET,
                   num_epochs = NUM_EPOCHS, batch_size = BATCH_SIZE,
                   num_workers=NUM_WORKERS,
                   OUT_DIR = OUT_DIR)