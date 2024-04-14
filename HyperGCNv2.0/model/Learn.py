import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

# Expert_shared, Expert_task1, Expert_task2 are similar. Just a Linear(in, out)
class Expert_net(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Expert_net, self).__init__()
        self.fc1 = nn.Linear(input_shape, output_shape)
        # init.xavier_normal_(self.fc1.weight)

    def forward(self, x):
        return self.fc1(x)


# Gate_shared, Gate_task1, Gate_task2 are similar. Just a Linear(in, out)
class Gate_net(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Gate_net, self).__init__()
        self.fc1 = nn.Linear(input_shape, output_shape)
        # init.xavier_normal_(self.fc1.weight)

    def forward(self, x):
        return self.fc1(x)

class Learn(nn.Module):
    def __init__(self):
        super(Learn,self).__init__()
        