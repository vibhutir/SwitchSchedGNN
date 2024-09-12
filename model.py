import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP).
    """
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(5,200, dtype=torch.float64)   # fully connected layer 1
        self.fc2 =  nn.Linear(200,300, dtype=torch.float64)  # fully connected layer 2 (output layer)
        self.fc3 = nn.Linear(300,300, dtype=torch.float64)
        self.fc4 = nn.Linear(300,200, dtype=torch.float64)
        self.fc5 = nn.Linear(200,5*5, dtype=torch.float64)
        
        self.init_weights()

    def init_weights(self):
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            f_in = fc.weight.size(1)
            nn.init.normal_(fc.weight, 0.0, 1 / math.sqrt(f_in/2))
            nn.init.constant_(fc.bias, 0.0)

    def forward(self, x):

        z = F.relu(self.fc1(x))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = F.relu(self.fc4(z))
        z = self.fc5(z)
        z = z.reshape(-1, 5, 5)  # reshape output to [batch_size, 5, 5]
        
        return z


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from dataset import SwitchDataset
    net = MLP()
    print(net)
    print('Number of CNN parameters: {}'.format(count_parameters(net)))
    dataset = SwitchDataset ()
    VOQs, Matchings = next(iter(dataset.train_loader))
    print('Size of model output:', net(VOQs.double()).size())