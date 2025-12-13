import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=False):
        super().__init__()
        
        layers = []
        
        # 第一次卷积
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 第二次卷积
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    


class UNet(nn.Module):
    def __init__(self, in_channels, num_classes=1, batch_norm=False):
        super().__init__()
        
        # Keras代码中的通道数 (32, 64, 128, 256, 512)
        
        # ---------------------
        # 1. 编码器/收缩路径 (Encoder/Downsampling)
        # ---------------------
        
        # Level 1 (32通道)
        self.conv1 = ConvBlock(in_channels, 32, use_batchnorm=batch_norm)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Level 2 (64通道)
        self.conv2 = ConvBlock(32, 64, use_batchnorm=batch_norm)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Level 3 (128通道)
        self.conv3 = ConvBlock(64, 128, use_batchnorm=batch_norm)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Level 4 (256通道)
        self.conv4 = ConvBlock(128, 256, use_batchnorm=batch_norm)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ---------------------
        # 2. 底部 (Bottleneck)
        # ---------------------
        
        # Level 5 (512通道)
        self.conv5 = ConvBlock(256, 512, use_batchnorm=batch_norm)
        
        # ---------------------
        # 3. 解码器/扩张路径 (Decoder/Upsampling)
        # ---------------------
        
        # Level 6 - Up from 5, concat with 4 (256通道)
        # Keras中使用 Conv2DTranspose(256, (2, 2), strides=(2, 2))
        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = ConvBlock(512, 256, use_batchnorm=batch_norm) # 512 = 256 (up) + 256 (conv4)
        
        # Level 7 - Up from 6, concat with 3 (128通道)
        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = ConvBlock(256, 128, use_batchnorm=batch_norm) # 256 = 128 (up) + 128 (conv3)
        
        # Level 8 - Up from 7, concat with 2 (64通道)
        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = ConvBlock(128, 64, use_batchnorm=batch_norm) # 128 = 64 (up) + 64 (conv2)
        
        # Level 9 - Up from 8, concat with 1 (32通道)
        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv9 = ConvBlock(64, 32, use_batchnorm=batch_norm) # 64 = 32 (up) + 32 (conv1)

        # ---------------------
        # 4. 输出层 (Output)
        # ---------------------
        
        # Keras中使用 Conv2D(1, (1, 1), activation='sigmoid')
        self.conv10 = nn.Conv2d(32, num_classes, kernel_size=1)
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        # 编码器部分 (Encoder)
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        # 底部 (Bottleneck)
        conv5 = self.conv5(pool4)

        # 解码器部分 (Decoder)
        
        # Level 6
        up6 = self.up6(conv5)
        # Keras中的 concatenate([up6, conv4], axis=3) 对应 PyTorch中的 torch.cat
        # 注意：PyTorch中可能需要裁剪或调整尺寸以保证拼接成功
        # PyTorch中的拼接发生在通道维度 (axis=1)
        up6 = torch.cat([up6, conv4], dim=1) 
        conv6 = self.conv6(up6)

        # Level 7
        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.conv7(up7)

        # Level 8
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.conv8(up8)

        # Level 9
        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.conv9(up9)

        # 输出层 (Output)
        conv10 = self.conv10(conv9)
        output=conv10

        return output