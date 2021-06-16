#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size = 3,stride = 1,padding = 1):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                             stride = stride , padding = padding)
        self.pool = nn.MaxPool3d(2, stride=2)
        self.relu = nn.LeakyReLU(inplace = True)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.pool(x)
        out = self.relu(x)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size = 3,stride = 1,padding = 1):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                             stride = stride , padding = padding)
        self.relu = nn.LeakyReLU(inplace = True)
    
    def forward(self,x):
        x = self.conv(x)
        x = F.interpolate(x,scale_factor = (2,2,2),mode = 'trilinear',align_corners = True)
        out = self.relu(x)
        return out
    
class crfbBlock(nn.Module):
    def __init__(self, in_channels, out_channels,scale_factor,kernel_size = 3,stride = 1,padding = 1):
        super(crfbBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size,
                             stride = stride , padding = padding)
        self.relu = nn.LeakyReLU(inplace = True)
        self.factor = scale_factor
    
    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        out = F.interpolate(x,scale_factor = (self.factor,self.factor,self.factor),mode = 'trilinear',align_corners = True)
        return out

class KiUnet(nn.Module):
    def __init__(self, in_channels=1,out_channels = 1):
        super(KiUnet, self).__init__()
        n = 1
        self.encoder1 = EncoderBlock(in_channels = 1 , out_channels = n)
                                        
        self.encoder2 = EncoderBlock(in_channels = n , out_channels = n*2)
                                     
        self.encoder3 = EncoderBlock(in_channels = n*2 , out_channels = n*4)
                                       
        self.decoder1 = DecoderBlock(in_channels = n*4 , out_channels = n*2)
        
        self.decoder2 = DecoderBlock(in_channels = n*2 , out_channels = n)
        
        self.decoder3 = DecoderBlock(in_channels = n, out_channels = 1)
        
        
        self.encoderf1 = DecoderBlock(in_channels = 1 , out_channels = n)
                                        
        self.encoderf2 = DecoderBlock(in_channels = n , out_channels = n*2)
                                     
        self.encoderf3 = DecoderBlock(in_channels = n*2 , out_channels = n*2)
                                       
        self.decoderf1 = EncoderBlock(in_channels = n*2 , out_channels = n*2)
        
        self.decoderf2 = EncoderBlock(in_channels = n*2 , out_channels = n)
        
        self.decoderf3 = EncoderBlock(in_channels = n , out_channels = 1)
        
        self.crfb1_1 = crfbBlock(in_channels = n , out_channels = n,scale_factor = 1/4)
        self.crfb1_2 = crfbBlock(in_channels = n , out_channels = n,scale_factor = 4)
        
        self.crfb2_1 = crfbBlock(in_channels = n*2 , out_channels = n*2,scale_factor = 1/16)
        self.crfb2_2 = crfbBlock(in_channels = n*2 , out_channels = n*2,scale_factor = 16)
        
        self.crfb3_1 = crfbBlock(in_channels = n*2 , out_channels = n*4,scale_factor = 1/64)
        self.crfb3_2 = crfbBlock(in_channels = n*4 , out_channels = n*2,scale_factor = 64)
        
        self.crfb4_1 = crfbBlock(in_channels = n*2 , out_channels = n*2,scale_factor = 1/16)
        self.crfb4_2 = crfbBlock(in_channels = n*2 , out_channels = n*2,scale_factor = 16)
        
        self.crfb5_1 = crfbBlock(in_channels = n , out_channels = n,scale_factor = 1/4)
        self.crfb5_2 = crfbBlock(in_channels = n , out_channels = n,scale_factor = 4)
        #self.deconv1 = ConvTranspose(in_channels = hidden_channels*16,out_channels = hidden_channels*16)
        #self.deconv2 = ConvTranspose(in_channels = hidden_channels*8,out_channels = hidden_channels*8)
        #self.deconv3 = ConvTranspose(in_channels = hidden_channels*4,out_channels = hidden_channels*4)
        
        self.final_conv = nn.Conv3d(in_channels = 1, out_channels = 1, kernel_size = 1)
    def forward(self,x):
    #encoder
        out = self.encoder1(x)      #[n,32,32,32]
        out1 = self.encoderf1(x)      #[n,128,128,128]
        tmp = out
        out = torch.add(out,self.crfb1_1(out1))  #[n,32,32,32]
        out1 = torch.add(out1,self.crfb1_2(tmp))    #[n,128,128,128]
        del tmp
        
        u1 = out
        o1 = out1
        
        out = self.encoder2(out)         #[2n,16,16,16]
        out1 =self.encoderf2(out1)       #[2n,256,256,256]
        tmp = out
        out = torch.add(out,self.crfb2_1(out1))        #[2n,16,16,16]
        out1 = torch.add(out1,self.crfb2_2(tmp))       #[2n,256,256,256]
        del tmp
        
        u2 = out
        o2 = out1
        
        out = self.encoder3(out)               #[4n,8,8,8]
        out1 = self.encoderf3(out1)            #[2n,512,512,512]
        tmp = out
        out = torch.add(out,self.crfb3_1(out1))   #[4n,8,8,8]
        out1 = torch.add(out1,self.crfb3_2(tmp))  #[2n,512,512,512]
        del tmp
        
        
        #decoder
        out = self.decoder1(out)     #[2n,16,16,16]
        out1 = self.decoderf1(out1)   #[2n,256,256,256]
        tmp = out
        out = torch.add(out,self.crfb4_1(out1))   #[2n,16,16,16]
        out1 = torch.add(out1,self.crfb4_2(tmp))  #[2n,256,256,256]
        del tmp
        out = torch.add(out, u2)          
        out1 = torch.add(out1,o2)
        
        out = self.decoder2(out)      #[n,32,32,32]
        out1 = self.decoderf2(out1)    #[n,128,128,128]
        tmp = out
        out = torch.add(out,self.crfb5_1(out1)) 
        out1 = torch.add(out1,self.crfb5_2(tmp)) 
        del tmp
        out = torch.add(out, u1)
        out1 = torch.add(out1,o1)
        
        out = self.decoder3(out)   #[1,64,64,64]
        out1 = self.decoderf3(out1)   #[1,64,64,64]
        out = torch.add(out,out1)
        out = self.final_conv(out)
        
        return out






