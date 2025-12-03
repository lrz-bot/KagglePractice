import torch
import torch.nn as nn
from torch.nn import Upsample

# --- Residual 模块 (保持不变，因为它在 Hourglass 中只用于匹配通道) ---
class Residual(nn.Module):
    def __init__(self, ins_channels, out_channels):
        super().__init__()
        # 使用 Sequential 确保所有子模块都有独立的参数
        self.ConvBlock = nn.Sequential(
            nn.Conv2d(in_channels=ins_channels, out_channels=(ins_channels//2), stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(ins_channels//2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=(ins_channels//2), out_channels=(ins_channels//2), stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(ins_channels//2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=(ins_channels//2), out_channels=out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.ins_channels = ins_channels
        self.out_channels = out_channels
        
        # 确保通道数匹配的跳跃连接 (在 Hourglass 中 ins=outs，所以这里是恒等映射)
        if ins_channels != out_channels:
             self.skip_connection = nn.Conv2d(ins_channels, out_channels, kernel_size=1)
        else:
             self.skip_connection = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.ConvBlock(x)
        
        if self.ins_channels != self.out_channels:
            residual = self.skip_connection(residual)
            
        x = x + residual
        return x

# --- Hourglass 模块 (修改以消除权重共享) ---
class Hourglass(nn.Module):
    def __init__(self, n=4, f=128):
        """
        :param f: hourglass模块中的特征图数量
        :param n: hourglass模块的层级数目
        """
        super().__init__()
        self._f = f
        self._n = n
        
        # 预先创建所有 n 级的残差块，确保权重独立
        
        # 1. 上分支残差块 (n个)
        self.up1_residuals = nn.ModuleList([Residual(f, f) for _ in range(n)])
        
        # 2. 下分支残差块 (n个)
        self.low1_residuals = nn.ModuleList([Residual(f, f) for _ in range(n)])
        
        # 3. 连接分支残差块 (n个)
        self.low3_residuals = nn.ModuleList([Residual(f, f) for _ in range(n)])
        
        # 4. 中心残差块 (n=1时的中心层，独立于列表)
        self.center_residual = Residual(f, f)

        # 池化和上采样（无学习参数，可以共享）
        self.maxpooling = nn.MaxPool2d(kernel_size=2)
        # 建议使用 nn.Upsample 并指定 mode
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest') 


    def _forward(self, x, n_level):
        # n_level 从 self._n 递减到 1
        
        # 索引：n_level - 1 (从 n-1 递减到 0)
        index = n_level - 1
        
        # 上分支
        up1 = x
        up1 = self.up1_residuals[index](up1) # 使用独立的模块

        # 下分支
        low1 = self.maxpooling(x)
        low1 = self.low1_residuals[index](low1) # 使用独立的模块
        
        if n_level > 1:
            low2 = self._forward(low1, n_level - 1)
        else:
            # 基础情况：中心层使用独立的模块
            low2 = self.center_residual(low1) 
            
        low3 = low2
        low3 = self.low3_residuals[index](low3) # 使用独立的模块
        
        up2 = self.upsample(low3)
        return up1 + up2
    
    def forward(self, x):
        res = self._forward(x, self._n)
        return res

# --- 辅助模块 (Lin, transform_to_heatmap, expand_channels 保持不变) ---
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

class expand_channels(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.relu=nn.ReLU()
    def forward(self,x):
        x=self.conv(x)
        x=self.relu(x)
        return x

class HourglassSingle(nn.Module):
    def __init__(self,n=4,f=128,keypoints=15,stride=1):
        super().__init__()
        self.n=n
        self.f=f
        self.keypoints=keypoints
        self.stride=stride
        self.expand_channels=expand_channels(1,f)
        # 确保这里传入的 n 和 f 正确初始化 Hourglass
        self.hourglass=Hourglass(n=n, f=f) 
        self.lin=Lin(ins_channels=self.f,keypoints_num=self.keypoints)
        self.transform_to_heatmp=transform_to_heatmap(self.stride)
    def forward(self,x):
        x=self.expand_channels(x)
        conv=self.hourglass(x)
        res=self.lin(conv)
        res=self.transform_to_heatmp(res)  #转化为热力图大小
        return res