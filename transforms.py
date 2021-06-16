#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from torchio.transforms import  ZNormalization
import random
import scipy
import torchio
class Window:

    def __init__(self, window_min, window_max):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image,label = None,training = True):
    
        image = np.clip(image, self.window_min, self.window_max)

        if training :return image,label
        return image


class MinMaxNorm:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image,label = None,training = True):
        image = (image - self.low) / (self.high - self.low)
        
        image = image * 2 - 1
        

        if training:return image,label
        return image

class RandomFlip:
    def __init__(self):
        pass
    def __call__(self,image,label):
        if random.randint(0,1):
            image,label = np.flip(image,axis = 0),np.flip(label,axis = 0)       #水平翻转
        if random.randint(0,1):
            image,label = np.flip(image,axis = 1),np.flip(label,axis = 1)     #垂直翻转
        return image,label
        
        
class addNoise:
    def __init__(self,prob=0.25,mean=0,std_factor=0.1):
        self.mean = mean
        self.std_factor = std_factor
        self.prob = prob
       
    def __call__(self,image,label):
        if random.uniform(0, 1)<self.prob:
            sigma = np.std(image) * self.std_factor              #标准差为patch标准差的0.1倍
            noise = np.random.normal(self.mean, sigma, image.shape)
            return np.add(image, noise),label
        else:return image,label
        
class RandomBlur:
    def __init__(self,prob = 0.25):
        self.prob = prob
        self.t = torchio.transforms.RandomBlur()
        
    def __call__(self,image,label):
        if random.uniform(0, 1)<self.prob:
            image = image[np.newaxis,:]
            image = self.t(image)
            image = image.squeeze(0)
        return image.copy(),label

class RandomScale:
    def __init__(self,prob=0.3,scale_range=[0.85,1.25]):
        self.prob = prob
        self.scale_range = scale_range
    def __call__(self,image,label):
        if random.uniform(0,1)<self.prob:
            scale = random.uniform(self.scale_range[0],self.scale_range[1])
            scale = np.round(scale,2)
            size = image.shape[0]
            image = scipy.ndimage.zoom(image,scale)
            label = scipy.ndimage.zoom(label,scale)
            if scale > 1:          #scale不为1时为保证图像大小不变，scale<1时使用背景的值（-200）填充，scale>1时进行center crop
                center = image.shape[0] // 2
                image = image[center-size//2:center+size//2,center-size//2:center+size//2,center-size//2:center+size//2]
                label = label[center-size//2:center+size//2,center-size//2:center+size//2,center-size//2:center+size//2]
            if scale<1:
                l_pad = (size-image.shape[0]) // 2
                r_pad = size-image.shape[0]-l_pad
                image = np.pad(image,((l_pad,r_pad),(l_pad,r_pad),(l_pad,r_pad)),'constant',constant_values = -200)
                label = np.pad(label,((l_pad,r_pad),(l_pad,r_pad),(l_pad,r_pad)),'constant')
            
        return image,label
    
class RandomRotate:
    def __init__(self,prob = 0.3):
        self.prob = prob
    def __call__(self,image,label):
        if random.uniform(0,1) < self.prob:     #随机旋转90/180/270度
            k = random.randint(1,3)
            image = np.rot90(image,k)
            label = np.rot90(label,k)
        return image,label
