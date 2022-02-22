import torch
from torchvision import datasets
from torchvision import transforms as T


normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])


def degradation():
    pass

def preprocess():
    pass