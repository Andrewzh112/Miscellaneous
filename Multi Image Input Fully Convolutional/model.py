# https://github.com/spmallick/learnopencv/tree/master/PyTorch-Fully-Convolutional-Image-Classification
import torch
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url

from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

from torchvision import transforms
from torchsummary import summary

class FullyConvolutionalResnet18(models.ResNet):
    def __init__(self, max_input_channels, num_classes, pretrained=False, **kwargs):

        super(FullyConvolutionalResnet18, self).__init__(block=models.resnet.BasicBlock, layers=[2, 2, 2, 2], 
                         num_classes=num_classes, **kwargs)


        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)

        self.avgpool = nn.AvgPool2d((7, 7))

        self.last_conv = torch.nn.Conv2d(in_channels=self.fc.in_features, out_channels=num_classes, kernel_size=1)
        self.last_conv.weight.data.copy_(self.fc.weight.data.view(*self.fc.weight.data.shape, 1, 1))
        self.last_conv.bias.data.copy_(self.fc.bias.data)
        
        # downsample
        self.squeezer1 = nn.Conv2d(max_input_channels, out_channels=16, 
                                  kernel_size=3, padding=1)
        self.squeezer2 = nn.Conv2d(16, out_channels=3, 
                                  kernel_size=3, padding=1)
        self.sbn1 = self._norm_layer(16)
        self.sbn2 = self._norm_layer(3)


    def forward(self, x):
        x = self.squeezer1(x)
        x = self.sbn1(x)
        x = self.relu(x)
        x = self.squeezer2(x)
        x = self.sbn2(x)
        x = self.relu(x)
        

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = self.last_conv(x)
        return x