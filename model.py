import torch
import torch.nn as nn
import torchvision.models as models

from utils import display_tensor


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        #self.resnet18 = models.resnet18(pretrained=True)
        self.resnet50 = models.resnet50(pretrained=True)
    def forward(self, x):
        x = self.resnet50(x)
        return x


if __name__ == '__main__':
    test_tensor = torch.rand(2, 3, 200, 200)  #B, C, H, W
    test_model = ResNet()

    y = test_model(test_tensor)
    print(y.size())
    #display_tensor(y)


