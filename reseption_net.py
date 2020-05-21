import torch
import math
import torch.nn.functional as F

class ReseptionNet(torch.nn.Module):
    def __init__(self, device, config):
        super(ReseptionNet, self).__init__()
        
        channels = config["inChannels"]
        
        self.inceptions = torch.nn.ModuleList([])
        dimensions = config["inDimensions"]
        for i, inception in enumerate(config["inceptions"]):
            for j in range(inception["amount"]):
                inceptionLayer = Inception(channels, dimensions, device, inception["config"])
                channels = inceptionLayer.outChannels
                dimensions = inceptionLayer.outDimensions
                self.inceptions.append(inceptionLayer)
                print("inception {} iteration {} layer output dimensions {} * {} * {} = {}".format(i+1, j+1, channels, dimensions[0], dimensions[1], channels*dimensions[0]*dimensions[1]))

                
        self.flatten = Flatten()
        self.linear = torch.nn.Linear(channels*dimensions[0]*dimensions[1], config["outputs"])
        self.sigmoid = torch.nn.Sigmoid()
        self.to(device)

    def forward(self, x):
        output = x
        for inception in self.inceptions:
            output = inception(output)
            
        output = self.flatten(output)
        output = self.linear(output)
        output = self.sigmoid(output)
        
        return output

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    
class Inception(torch.nn.Module):
    
    inceptionConfig = None
    outChannels = None
    outDimensions = None

    def __init__(self, inChannels, inDimensions, device, inceptionConfig):
        super(Inception, self).__init__()
        self.inceptionConfig = inceptionConfig
        
        self.outChannels = 0
        self.branches = torch.nn.ModuleList([])
        branchDimensions = [
            self.updateDimensions(
                inDimensions,
                self.inceptionConfig["shortcut"]["padding"],
                self.inceptionConfig["shortcut"]["dilation"],
                self.inceptionConfig["shortcut"]["kernelSize"],
                self.inceptionConfig["shortcut"]["stride"]
            )
        ]
        for branch in self.inceptionConfig["branches"]:
            blocks = torch.nn.ModuleList([])
            channels = inChannels
            dimensions = inDimensions
            for block in branch["blocks"]:
                convolution = block["convolution"]
                blocks.append(convolution(
                    channels,
                    math.ceil(channels*block["outputChannelMultiplier"]),
                    device,
                    kernel_size = block["kernelSize"],
                    padding=block["padding"],
                    stride=block["stride"],
                    dilation=block["dilation"],
                    groups=channels if block["grouping"] else 1
                ))
                channels = math.ceil(channels*block["outputChannelMultiplier"])
                dimensions = self.updateDimensions(dimensions, block["padding"], block["dilation"], block["kernelSize"], block["stride"])
            self.outChannels += channels
            self.branches.append(blocks)
            branchDimensions.append(dimensions)
            
        for dimensions in branchDimensions:
            if dimensions != branchDimensions[0]:
                print(branchDimensions)
                raise Exception("Dimensions must stay the same between all branches and shortcut in inceptions")
        
        self.outDimensions = branchDimensions[0]
            
        self.shortcut = self.inceptionConfig["shortcut"]["convolution"](
            inChannels,
            self.outChannels,
            device,
            kernel_size = self.inceptionConfig["shortcut"]["kernelSize"],
            padding = self.inceptionConfig["shortcut"]["padding"],
            stride = self.inceptionConfig["shortcut"]["stride"],
            dilation = self.inceptionConfig["shortcut"]["dilation"],
            groups=(inChannels if self.inceptionConfig["shortcut"]["grouping"] else 1)
        )

    def forward(self, x):
        outputs = []
        for branch in self.branches:
            output = x
            for block in branch:
                output = block(output)
            outputs.append(output)
        
        output = torch.cat(outputs, 1)
        shortcut = self.shortcut(x)
        output = output + shortcut
        output = F.relu(output)

        return output
    
    def updateDimensions(self, dimensions, padding, dilation, kernelSize, stride):
        def dimensionalize(value):
            if type(value) is tuple:
                return value
            else:
                return (value, value)
        padding = dimensionalize(padding)
        dilation = dimensionalize(dilation)
        kernelSize = dimensionalize(kernelSize)
        stride = dimensionalize(stride)
        
        newHeight = (dimensions[0] + 2*padding[0] - dilation[0]*(kernelSize[0]-1)-1)//(stride[0])+1
        newWidth = (dimensions[1] + 2*padding[1] - dilation[1]*(kernelSize[1]-1)-1)//(stride[1])+1
        return (newHeight, newWidth)


class Flatten(torch.nn.Module):
    
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)