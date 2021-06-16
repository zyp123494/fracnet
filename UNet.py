#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock(nn.Module):           #卷积块
    def __init__(self, in_channels, out_channels,kernel_size = 3,stride = 1,padding = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                             stride = stride , padding = padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.01,inplace = True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
        
class ConvTranspose(nn.Module):     #反卷积块
    def __init__(self, in_channels, out_channels,kernel_size = 2,stride = 2,padding = 0):
        super(ConvTranspose, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                             stride = stride , padding = padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.01,inplace = True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        out = self.relu(x)
        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels=1,out_channels = 1,hidden_channels = 32):
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(ConvBlock(in_channels = in_channels , out_channels = hidden_channels),
                                      ConvBlock(in_channels = hidden_channels , out_channels = hidden_channels*2))
        
        self.encoder2 = nn.Sequential(ConvBlock(in_channels = hidden_channels*2 , out_channels = hidden_channels*2),
                                      ConvBlock(in_channels = hidden_channels*2 , out_channels = hidden_channels*4))
        
        self.encoder3 = nn.Sequential(ConvBlock(in_channels = hidden_channels*4 , out_channels = hidden_channels*4),
                                      ConvBlock(in_channels = hidden_channels*4 , out_channels =hidden_channels*8))
        
        self.encoder4 = nn.Sequential(ConvBlock(in_channels = hidden_channels*8 , out_channels = hidden_channels*8),
                                      ConvBlock(in_channels = hidden_channels*8 , out_channels = hidden_channels*16))
        
        self.decoder1 = nn.Sequential(ConvBlock(in_channels = hidden_channels*24 , out_channels = hidden_channels*8),
                                      ConvBlock(in_channels = hidden_channels*8 , out_channels = hidden_channels*8))
            
        self.decoder2 = nn.Sequential(ConvBlock(in_channels = hidden_channels*12 , out_channels = hidden_channels*4),
                                      ConvBlock(in_channels = hidden_channels*4 , out_channels = hidden_channels*4))
        
        self.decoder3 = nn.Sequential(ConvBlock(in_channels = hidden_channels*6 , out_channels = hidden_channels*2),
                                      ConvBlock(in_channels = hidden_channels*2 , out_channels = hidden_channels*2))
        self.deconv1 = ConvTranspose(in_channels = hidden_channels*16,out_channels = hidden_channels*16)
        self.deconv2 = ConvTranspose(in_channels = hidden_channels*8,out_channels = hidden_channels*8)
        self.deconv3 = ConvTranspose(in_channels = hidden_channels*4,out_channels = hidden_channels*4)
        
        
        self.final_conv = nn.Conv3d(in_channels = hidden_channels*2, out_channels = out_channels, kernel_size = 1)
        self.pool = nn.MaxPool3d(2,2)
        
        for m in self.modules():       #模型初始化
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
          
    def forward(self,x):
        e1 = x = self.encoder1(x)
        x = self.pool(x)
        e2 = x = self.encoder2(x)
        x = self.pool(x)
        e3 = x = self.encoder3(x)
        x = self.pool(x)
        x = self.encoder4(x)
        
        #也可使用F.interpolate进行插值
        #x = F.interpolate(x,scale_factor = (2,2,2),mode = 'trilinear',align_corners = True)
        x = self.deconv1(x)
        x = torch.cat((x,e3),dim = 1)
        x = self.decoder1(x)
                          
        #x = F.interpolate(x,scale_factor = (2,2,2),mode = 'trilinear',align_corners = True)
        x = self.deconv2(x)
        x = torch.cat((x,e2),dim = 1)
        x = self.decoder2(x)
                          
        #x = F.interpolate(x,scale_factor = (2,2,2),mode = 'trilinear',align_corners = True)
        x = self.deconv3(x)
        x = torch.cat((x,e1),dim = 1)
        x = self.decoder3(x)
        
        x = self.final_conv(x)
        return x







