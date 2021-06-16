

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_planes, planes , stride = 1):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size = 3,stride = stride,padding = 1,bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes * self.expansion,kernel_size = 3, stride = 1 , padding=1,bias = False)
        self.bn2 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(in_planes,self.expansion * planes,kernel_size = 1,stride = stride,bias = False),
                            nn.BatchNorm2d(self.expansion * planes)
                                         )
        else: self.shortcut = None
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        if self.shortcut is not None:
            x = self.shortcut(x)
        out +=x
        out = self.relu(out)
        return out
        


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self,in_planes,planes,stride = 1):
        super(Bottleneck,self).__init__()
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size = 1,bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size = 3,padding = 1,stride = stride,bias = False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,planes * self.expansion,kernel_size = 1,stride = 1,bias = False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                            nn.Conv2d(in_planes,self.expansion * planes,kernel_size = 1,stride = stride,bias = False),
                            nn.BatchNorm2d(self.expansion * planes)
                                         )
        else: self.shortcut = None
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        if self.shortcut is not None:
            x = self.shortcut(x)
        out += x
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,block,num_blocks,in_channels = 1,num_classes = 2):
        super(ResNet,self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(in_channels,64,kernel_size = 3,stride = 1, padding = 1 ,bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64  ,num_blocks[0],stride = 1)
        self.layer2 = self._make_layer(block, 128 ,num_blocks[1],stride = 2)
        self.layer3 = self._make_layer(block, 256 ,num_blocks[2],stride = 2)
        self.layer4 = self._make_layer(block, 512 ,num_blocks[3],stride = 2)
        
        self.avgpool = nn.AvgPool2d(4)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.drop = nn.Dropout(0.4)
    def _make_layer(self,block,planes,num_blocks,stride):
        layers = []
        layers.append(block(self.in_planes, planes,stride))
        self.in_planes = planes * block.expansion
        for i in range(1,num_blocks):
            layers.append(block(self.in_planes,planes,stride = 1))
        return nn.Sequential(*layers)
        
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.drop(out)
        return out



def ResNet18(in_channels, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)
    



def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)