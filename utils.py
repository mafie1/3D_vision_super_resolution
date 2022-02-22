import matplotlib.pyplot as plt
import torch

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def display_tensor(t):
    if len(t.size()) == 4:  #in case tensor is Batch Tensor, choose first image in Batch
        img = t[0]
    else:
        img = t

    #print(img.size())
    img = img.permute(1,2,0).cpu().detach().numpy()

    #plotting
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    test_tensor = torch.rand(2, 3, 200, 200)

    display_tensor(test_tensor)