import torch
import torch.nn as nn

class Charbonnier(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(Charbonnier, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)
        C = torch.mean(torch.sqrt((inputs-targets)**2+self.epsilon**2))
        return C


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)
    tensor1 = torch.rand(4, 3, 200, 200)
    tensor2 = torch.rand(4, 3, 200, 200)

    loss = Charbonnier()
    loss_L1 = nn.L1Loss()
    loss_MSE = nn.MSELoss()

    print(loss(tensor1, tensor2))
    print(loss_L1(tensor1, tensor2))
    print(loss_MSE(tensor1, tensor2))