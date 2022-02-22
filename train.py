import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
import os
import copy

from models import SRCNN
from utils import display_tensor
from utils import AverageMeter, calc_psnr
from datasets import TrainDataset, EvalDataset

SEED = 0
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
BATCH_SIZE = 4
NUM_WORKERS = 0
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)

TRAIN_FILE = "data/91-image_x2.h5"
EVAL_FILE = "data/Set5_x2.h5"
OUT_DIR = "outputs"

"""
optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
    ], lr=args.lr)
    
"""
model = SRCNN(num_channels=1).to(DEVICE) #color image
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE) #all training, later: train head and backbone separate
criterion = nn.MSELoss()


train_dataset = TrainDataset(TRAIN_FILE)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=NUM_WORKERS,
                              pin_memory=True,
                              drop_last=True)

eval_dataset = EvalDataset(EVAL_FILE)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

running_loss = 0.0
best_epoch = 0
best_psnr = 0.0

if __name__ == '__main__':
    for epoch in range(NUM_EPOCHS):
            model.train()
            epoch_losses = AverageMeter()

            with tqdm(total=(len(train_dataset) - len(train_dataset) % BATCH_SIZE)) as t:
                t.set_description('epoch: {}/{}'.format(epoch, NUM_EPOCHS - 1))

                for data in train_dataloader:
                    inputs, labels = data

                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)

                    preds = model(inputs)

                    loss = criterion(preds, labels)

                    epoch_losses.update(loss.item(), len(inputs))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    t.update(len(inputs))

            torch.save(model.state_dict(), os.path.join(OUT_DIR, 'epoch_{}.pth'.format(epoch)))

            model.eval()
            epoch_psnr = AverageMeter()

            for data in eval_dataloader:
                inputs, labels = data

                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                with torch.no_grad():
                    preds = model(inputs).clamp(0.0, 1.0)

                epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

            print('eval psnr: {:.2f}'.format(epoch_psnr.avg))

            if epoch_psnr.avg > best_psnr:
                best_epoch = epoch
                best_psnr = epoch_psnr.avg
                best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(OUT_DIR, 'best.pth'))