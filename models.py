import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from math import sqrt
from utils import display_tensor
from datasets import BSD100


def make_layer(block, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block())
    return nn.Sequential(*layers)


class Conv_ReLU_Block(nn.Module):
    def __init__(self, _in=64, _out=64, _size=3, _stride=1, _padding=1):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=_size, stride=_stride, padding=_padding,
                              bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class Net(nn.Module):
    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.residual_layer = make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out


class VDSR(nn.Module):
    """Accurate Image Super-Resolution Using Very Deep Convolutional Networks
    d: number of layers
    f_map: number of features in each layer
    filter_size = 3x3
    zero-padding = True

    source: https://github.com/twtygqyy/pytorch-vdsr/blob/master

    """

    def __init__(self, num_channels=3, d=5, scale=2, fmap=64):
        super(VDSR, self).__init__()
        # self.scale = scale
        # if scale is not None:
        #   self.pre_upsample = torch.nn.Upsample(size=None, scale_factor=scale, mode='bicubic', align_corners=True)
        self.residual_layer = make_layer(Conv_ReLU_Block, d)

        self.input = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_channels, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        # if self.scale is not None:
        #   x = self.pre_upsample(x)
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
    see https://github.com/yjn870/SRCNN-pytorch/blob/master/models.py
    """

    def __init__(self, num_channels=3, scale=None):
        super(SRCNN, self).__init__()

        # self.scale = scale
        # if scale is not None:
        #   self.pre_upsample = torch.nn.Upsample(size=None, scale_factor=scale, mode='bicubic', align_corners=True)

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        # if self.scale is not None:
        #   x = self.pre_upsample(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        return x


class ResNet(nn.Module):
    """Do not use, is trained to extract 1000-dim feature vector"""

    def __init__(self):
        super(ResNet, self).__init__()
        # self.resnet18 = models.resnet18(pretrained=True)
        self.model = models.resnet50(pretrained=True)
        self.stripped = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        # model = models.resnet152(pretrained=True)
        # newmodel = torch.nn.Sequential(*(list(model.children())[:-1]))
        print(self.stripped)

    def forward(self, x):
        # x = self.resnet50(x)
        x = self.stripped(x)
        return x


if __name__ == '__main__':
    writer = SummaryWriter('runs/experimenting')

    # test_tensor = torch.rand(2, 3, 200, 200)  #B, C, H, W
    test_tensor = torch.rand(4, 3, 200, 200)
    print(test_tensor.dtype)

    """Pre-Upsampling Methods"""
    # test_model = SRCNN(num_channels=3) #is working :)
    test_model_vdsr = VDSR(num_channels=3)  # is working :)

    """Other"""
    # test_model = Net(num_channels=3)
    test_model_resnet = ResNet()

    # print(test_model)
    y = test_model_vdsr(test_tensor)
    print(y.size())

    # writer.add_graph(test_model, test_tensor)
    # writer.close()
    # display_tensor(y)
