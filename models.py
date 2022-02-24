import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from math import sqrt

from utils import display_tensor


class Conv_ReLU_Block(nn.Module):
    def __init__(self, IN = 64, OUT = 64, SIZE = 3, STRIDE = 1, PADDING = 1):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=IN, out_channels=OUT, kernel_size=SIZE, stride=STRIDE, padding=PADDING, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    """Accurate Image Super-Resolution Using Very Deep Convolutional Networks
    d: number of layers
    f_map: number of features in each layer
    filter_size = 3x3
    zero-padding = True

    source: https://github.com/twtygqyy/pytorch-vdsr/blob/master

    """
    def __init__(self, num_channels=3, d = 5, fmap = 64):
        super(VDSR, self).__init__()

        self.residual_layer = self.make_layer(Conv_ReLU_Block, d)

        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


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


    #test_tensor = torch.rand(2, 3, 200, 200)  #B, C, H, W
    test_tensor = torch.rand(2, 1, 200, 200)

    #test_model = ResNet()
    #test_model = SRCNN(num_channels=3)
    test_model = VDSR()

    print(test_model)

    y = test_model(test_tensor)
    print(y.size())

    writer.add_graph(test_model, test_tensor)
    writer.close()
    #display_tensor(y)


