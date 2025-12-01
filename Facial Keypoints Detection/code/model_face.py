import torch
import torch.nn as nn


class Hourglass(nn.Module):
    def __init__(self,n=4,f=256):
        """
        :param n: hourglass模块的层级数目
        :param f: hourglass模块中的特征图数量
        :return:
        """
        self._n=n
        self._f=f
        super().__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=64,kernel_size=1)
        self.conv2=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=1)
        self.conv3=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=1)
        self.upsample=nn.Upsample(scale_factor=2)
        self.maxpooling=nn.MaxPool2d(kernel_size=2)
        self.residual=Residual(self._f,self._f)
    def forward(self,x):
        x=self.relu(self.conv1(x))
        x=self.relu(self.conv2(x))
        x=self.relu(self.conv3(x))
        init_layer=self.residual(x)
        low1=(self.maxpooling(x))
        low2=(self.residual(low1))
        low3=(self.residual(low2))
        low4=(self.residual(low3))
        up=self.relu((self.upsample(low4)))
        res=up+init_layer
        
        return res
    




class Residual(nn.Module):
    def __init__(self,ins_channels,out_channels):
        super().__init__()
        self.ConvBlock=nn.Sequential(
            nn.Conv2d(in_channels=ins_channels,out_channels=(ins_channels//2),stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(ins_channels//2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=(ins_channels//2),out_channels=(ins_channels//2),stride=1,kernel_size=3,padding=1),
            nn.BatchNorm2d(ins_channels//2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=(ins_channels//2),out_channels=out_channels,kernel_size=1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.ins_channels=ins_channels
        self.out_channels=out_channels
    def forward(self,x):
        residual=x
        x=self.ConvBlock(x)
        x=x+residual
        return x
    
class Lin(nn.Module):
    def __init__(self,ins_channels,keypoints_num):
        super().__init__()
        self.conv=nn.Conv2d(in_channels=ins_channels,out_channels=keypoints_num,kernel_size=1)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        self.bn=nn.BatchNorm2d(keypoints_num)
    def forward(self,x):
        x=self.sigmoid(self.bn(self.conv(x)))
        return x
    
class transform_to_heatmap(nn.Module):
    def __init__(self,stride):
        super().__init__()
        self.stride=stride
    def forward(self,x):
        x=nn.MaxPool2d(kernel_size=self.stride)(x)
        return x
    
class HourglassSingle(nn.Module):
    def __init__(self,n=4,f=256,keypoints=15,stride=1):
        super().__init__()
        self.n=n
        self.f=f
        self.keypoints=keypoints
        self.stride=stride
        self.hourglass=Hourglass()
        self.lin=Lin(ins_channels=self.f,keypoints_num=self.keypoints)
        self.transform_to_heatmp=transform_to_heatmap(self.stride)
    def forward(self,x):
        conv=self.hourglass(x)
        res=self.lin(conv)
        res=self.transform_to_heatmp(res)  #转化为热力图大小
        return res

