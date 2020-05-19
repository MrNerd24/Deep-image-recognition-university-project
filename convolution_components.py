import torch
import torch.nn.functional as F
import numpy as np


class BasicConv2d(torch.nn.Module):

    def __init__(self, in_channels, out_channels, device, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class AveragePooling(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, device, **kwarks):
        super(AveragePooling, self).__init__()
        del kwarks["dilation"]
        self.kwarks = kwarks
        
    def forward(self, x):
        return F.avg_pool2d(x, **self.kwarks)
    
class MaxPooling(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, device, **kwarks):
        super(MaxPooling, self).__init__()
        del kwarks["dilation"]
        self.kwarks = kwarks
        
    def forward(self, x):
        return F.max_pool2d(x, **self.kwarks)
    
    
    
class IdentityConv2d(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, device, **kwargs):
        super(IdentityConv2d, self).__init__()
        self.kwargs = kwargs
        def dimensionalize(value):
            if type(value) is tuple:
                return value
            else:
                return (value, value)
        kernelSize = dimensionalize(kwargs["kernel_size"])
        del kwargs["kernel_size"]
        self.weights = np.zeros(kernelSize)
        self.weights[kernelSize[0]//2, kernelSize[1]//2] = 1
        self.weights = torch.Tensor(self.weights).to(device)
        self.weights = self.weights.view(1, 1, kernelSize[0], kernelSize[1]).repeat(out_channels, in_channels, 1, 1)
        
    def forward(self, x):
        return F.conv2d(x, self.weights, **self.kwargs)