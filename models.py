import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from utils import display_tensor

class SRCNN(nn.Module):
    """
    SRCNN model from [paper citation]:
    feature maps are 1 (original) --> 64 --> 32 --> 1 (output)
    """
    def __init__(self, num_channels=3):

        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size = 9, padding= 9//2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size = 5, padding = 5//2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size = 5, padding= 5//2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        return x


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        #self.resnet18 = models.resnet18(pretrained=True)
        self.resnet50 = models.resnet50(pretrained=True)
    def forward(self, x):
        x = self.resnet50(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

class UNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x


if __name__ == '__main__':
    writer = SummaryWriter('runs/experimenting')
    print(SRCNN())

    test_tensor = torch.rand(2, 3, 200, 200)  #B, C, H, W


    #test_model = ResNet()
    test_model = SRCNN()

    y = test_model(test_tensor)
    print(y.size())

    writer.add_graph(test_model, test_tensor)
    writer.close()
    #display_tensor(y)


