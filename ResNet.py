import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, out_channels, strides=1, downspample = None):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downspample
        
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.downsample:
            X = self.downsample(X)
        Y += X
        Y = F.relu(Y)
        return Y 

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, input_channels=3):
        super().__init__()
        self.in_channels = 64
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.avgpool = nn.AdaptiveMaxPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 1.0)
                
        
        
    def make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if(stride!=1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, X):
        #(bs, 3, 32, 32)
        Y = self.layer0(X)  #(bs, 64, 16, 16)
        Y = self.layer1(Y)  #(bs, 64, 16, 16)
        Y = self.layer2(Y)  #(bs, 128, 8, 8)
        Y = self.layer3(Y)  #(bs, 256, 4, 4)
        Y = self.layer4(Y)  #(bs, 512, 2, 2)
        Y = self.avgpool(Y) #(bs, 512, 1, 1)
        Y = Y.view(Y.size(0), -1) #(bs, 512)
        Y = self.fc(Y) #(bs, 10)
        return Y
    

