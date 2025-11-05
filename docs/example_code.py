# Workshop from the 4th of November
import torch
import torch.nn as nn


class myNN(nn.Module):
    def __init__(self):
        self.layer1 = nn.Linear(2, 3) # 2 inputs, 3 outputs.
        self.layer2 = nn.Linear(3, 2)
        self.layer3 = nn.Linear(2, 1)

        self.relu = nn.ReLU() # relu funtion, essentially f(x) = max(0, x)

    def forward(self, input):
        x = self.layer1(input)
        x = self.relu(x)

        x = self.layer2(input)
        x = self.relu(x)
        
        x = self.layer3(input)
        x = self.relu(x)
        
