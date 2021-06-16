#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
class ConvBlock(nn.Module):
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
        
class ConvTranspose(nn.Module):
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
    def __init__(self, in_channels=1,out_channels = 1,hidden_channels = 16):
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(ConvBlock(in_channels = in_channels , out_channels = hidden_channels),
                                      ConvBlock(in_channels = hidden_channels , out_channels = hidden_channels*2))
        
        self.encoder2 = nn.Sequential(ConvBlock(in_channels = hidden_channels*2 , out_channels = hidden_channels*2,stride = 2),
                                      ConvBlock(in_channels = hidden_channels*2 , out_channels = hidden_channels*4))
        
        self.encoder3 = nn.Sequential(ConvBlock(in_channels = hidden_channels*4 , out_channels = hidden_channels*4,stride = 2),
                                      ConvBlock(in_channels = hidden_channels*4 , out_channels =hidden_channels*8))
        
        self.encoder4 = nn.Sequential(ConvBlock(in_channels = hidden_channels*8 , out_channels = hidden_channels*8,stride = 2),
                                      ConvBlock(in_channels = hidden_channels*8 , out_channels = hidden_channels*16))
                                      
        self.encoder5 = nn.Sequential(ConvBlock(in_channels = hidden_channels*16 , out_channels = hidden_channels*16,stride = 2),
                                      ConvBlock(in_channels = hidden_channels*16 , out_channels = hidden_channels*20))
                                      
        self.encoder6 =  nn.Sequential(ConvBlock(in_channels = hidden_channels*20 , out_channels = hidden_channels*20,stride = 2),
                                      ConvBlock(in_channels = hidden_channels*20 , out_channels = hidden_channels*20))
        
        self.decoder1 = nn.Sequential(ConvBlock(in_channels = hidden_channels*40 , out_channels = hidden_channels*16),
                                      ConvBlock(in_channels = hidden_channels*16 , out_channels = hidden_channels*16))
        
            
        self.decoder2 = nn.Sequential(ConvBlock(in_channels = hidden_channels*32 , out_channels = hidden_channels*8),
                                      ConvBlock(in_channels = hidden_channels*8 , out_channels = hidden_channels*8))
        
        self.decoder3 = nn.Sequential(ConvBlock(in_channels = hidden_channels*16 , out_channels = hidden_channels*4),
                                      ConvBlock(in_channels = hidden_channels*4 , out_channels = hidden_channels*4))
                                      
        self.decoder4 = nn.Sequential(ConvBlock(in_channels = hidden_channels*8 , out_channels = hidden_channels*2),
                                      ConvBlock(in_channels = hidden_channels*2 , out_channels = hidden_channels*2))
        
        self.decoder5 = nn.Sequential(ConvBlock(in_channels = hidden_channels*4 , out_channels = hidden_channels),
                                      ConvBlock(in_channels = hidden_channels , out_channels = hidden_channels))
                                      
        self.deconv1 = ConvTranspose(in_channels = hidden_channels*20,out_channels = hidden_channels*20)
        self.deconv2 = ConvTranspose(in_channels = hidden_channels*16,out_channels = hidden_channels*16)
        self.deconv3 = ConvTranspose(in_channels = hidden_channels*8,out_channels = hidden_channels*8)
        self.deconv4 = ConvTranspose(in_channels = hidden_channels*4,out_channels = hidden_channels*4)
        self.deconv5 = ConvTranspose(in_channels = hidden_channels*2,out_channels = hidden_channels*2)
        
        
        self.final_conv = nn.Conv3d(in_channels = hidden_channels, out_channels = out_channels, kernel_size = 1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                m.weight = nn.init.kaiming_normal_(m.weight, a=0.01)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
          
    def forward(self,x):
        e1 = x = self.encoder1(x)
        e2 = x = self.encoder2(x)
        e3 = x = self.encoder3(x)
        e4 = x = self.encoder4(x)
        e5 = x = self.encoder5(x)
        x = self.encoder6(x)
        #x = F.interpolate(x,scale_factor = (2,2,2),mode = 'trilinear',align_corners = True)
        x = self.deconv1(x)
        x = torch.cat((x,e5),dim = 1)
        x = self.decoder1(x)
                          
        #x = F.interpolate(x,scale_factor = (2,2,2),mode = 'trilinear',align_corners = True)
        x = self.deconv2(x)
        x = torch.cat((x,e4),dim = 1)
        x = self.decoder2(x)
                          
        #x = F.interpolate(x,scale_factor = (2,2,2),mode = 'trilinear',align_corners = True)
        x = self.deconv3(x)
        x = torch.cat((x,e3),dim = 1)
        x = self.decoder3(x)
        
        x = self.deconv4(x)
        x = torch.cat((x,e2),dim = 1)
        x = self.decoder4(x)
        
        x = self.deconv5(x)
        x = torch.cat((x,e1),dim = 1)
        x = self.decoder5(x)
        
        x = self.final_conv(x)
        return x







