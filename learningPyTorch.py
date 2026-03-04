import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Create a model class that inherits from nn.Module
class Model(nn.Module):
    # input layer
    # hidden layer 1
    # hidden layer 2
    # output layer

    def __init__(self, inFeatures = 4, h1 = 8, h2 = 9, outFeatures = 3):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(inFeatures, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, outFeatures)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
    torch.manual_seed(42)


model = Model()

url = ''
df = 
