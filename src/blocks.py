"""
This file will contain all the building blocks for making Generator and Discriminator(critic) models.
"""

import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, inChannels:int, outChannels:int, usePN:bool=True, *args, **kwargs) -> None:
        """
        Convolution Block used for Generators and Discriminators.

        Parameters:
        - inChannels (int): The number of input channels for the convolution block.
        - outChannels (int): The number of output channels for the convolution block.
        - usePN (bool, optional): Whether to use Pixel Normalization. Defaults to True.
        """
        super().__init__(*args, **kwargs)

        self.usePN = usePN
        self.inChannels = inChannels
        self.outChannels = outChannels
        # setting configuration 

        self.conv1 = WSConv2d(inChannels, outChannels, kernelSize=3, stride=1, padding=1)
        self.conv2 = WSConv2d(outChannels, outChannels, kernelSize=3, stride=1, padding=1)
        self.leaky = nn.LeakyReLU(0.2)
        
        if usePN:
            self.pn = PixelNorm()
        # initializing layers 

    def forward(self, x):
        """
        Passes the input through 2 convolution layers and applies pixel normalization after each convolution layer if `usePN=True`
        """
        x = self.conv1(x)
        x = self.leaky(x)

        x = self.pn(x) if self.usePN else x
        # passing the input through first convolution layer 

        x = self.conv2(x)
        x = self.leaky(x)

        x = self.pn(x) if self.usePN else x
        # passing the input through second convolution layer

        return x
    
    def __repr__(self):
        return f"ConvBlock(inChannels={self.inChannels}, outChannels={self.outChannels}, usePN={self.usePN})"

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def forward(self, x):
        meanSqrt = torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)
        return x / meanSqrt
    
    def __repr__(self):
        return f"PixelNorm(epsilon={self.epsilon})"

class WSConv2d(nn.Module):
    def __init__(self, inChannels: int, outChannels: int, kernelSize: int, stride: int, padding: int, gain: float = 2**(1/2), *args, **kwargs) -> None:
        """
        This is a weight scaled Convolution Layer.

        This layer initializes with N(0, scale). It will multiply the scale for every forward pass.

        Parameters:
        - inChannels (int): The number of input channels for the convolution layer.
        - outChannels (int): The number of output channels for the convolution layer.
        - kernelSize (int): The size of the convolution kernel.
        - stride (int): The stride of the convolution operation.
        - padding (int): The amount of padding added to the input.
        - gain (float, optional): The scaling factor for the weights. Defaults to sqrt(2).
        - args: Additional positional arguments.
        - kwargs: Additional keyword arguments.

        Returns:
        Nothing. This is a constructor method for the WSConv2d class. It initializes the convolution layer with the given parameters and sets the bias of the original convolution layer to None.
        """
        super().__init__(*args, **kwargs)

        self.convLayer = nn.Conv2d(inChannels, outChannels, kernel_size=kernelSize, stride=stride, padding=padding)
        self.bias = torch.zeros_like(self.convLayer.bias)
        self.convLayer.bias = None
        # initializes the convlayer without bias 

        inputProd = torch.prod(torch.Tensor(list(self.convLayer.weight.shape[1:])))
        self.weightScale = gain / torch.sqrt(inputProd)
        # initializing the weight scale constant 
        
        nn.init.normal_(self.convLayer.weight)
        # normalizing the weights and zeroing the bias

    def forward(self, x) -> torch.Tensor:
        """
        Passses the input through a convolution layer, multiply it with weightscale and then add the bias.
        """
        x = self.convLayer(x) * self.weightScale
        x += self.bias.view(1, self.bias.shape[0], 1, 1)

        return x
    
    def __repr__(self):
        return f"WSConv2d(weightScale = {'{: .3f}'.format(self.weightScale)})"