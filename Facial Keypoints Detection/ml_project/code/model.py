import torch.nn as nn
import torch


class facemodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )
        self.conv3=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten=nn.Flatten()
        self.linear1=nn.Linear(in_features=12800,out_features=500)
        self.linear2=nn.Linear(in_features=500,out_features=500)
        self.dropout=nn.Dropout(0.5)
        self.relu=nn.ReLU()
        self.linear3=nn.Linear(in_features=500,out_features=30)
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.flatten(x)
        x=self.linear1(x)
        x=self.relu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        x=self.relu(x)
        return x
