from src.blocks import WSConv2d, ConvBlock, PixelNorm
import torch.nn.functional as F
import torch.nn as nn
from torch import tanh

class Generator(nn.Module):
    """
    ## Generator Model

    Genertor model takes in a latent vector and generates an image.

    Generator model is made up of 2 parts:
        
    ### RGB Heads:
        
    The RGB heads are responsible for 2 things:
    1. Acting as the output head for each size the generator produce
    2. Acting as the input for the next RGB Module

    RGB heads except the last two can be dumped after training the model.

    ### Progressive Heads:
        
    Each progressive heads are used as the previous layer of the rgb head.
    """
    def __init__(self, zDim:int, channels:list[int], imageChannels:int=3, device:str="cuda", *args, **kwargs) -> None:
        """
        Initializes the generator.

        Parameters:
        zDim (int): Dimension of the latent vector
        channels (list[int]): Number of channels of each layers.
        imageChannels (int): Number of channels in the output image. Defaults to 3.
        device (str): Device to run the model on. Defaults to "cuda".

        Example:
        >>> channels = [256, 256, 256, 256, 128, 64, 32, 16, 8]
        >>> generator = Generator(128, channels, 3)
        """
        super().__init__(*args, **kwargs)
        self.device = device

        self.initialProgressiveLayer = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(zDim, channels[0], kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(channels[0], channels[0], kernelSize=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.initialRGBLayer = WSConv2d(channels[0], imageChannels, kernelSize=1, stride=1, padding=0)
        # first RGB layer 

        self.progressiveBlocks = nn.ModuleList()
        self.rgbBlocks = nn.ModuleList([self.initialRGBLayer])

        for i, channelNumber in enumerate(channels[1:-1]):
            outChannelNumber = channels[i+2]
            # getting the number of output channels 

            progressiveBlock = ConvBlock(channelNumber, outChannelNumber)
            rgbBlock = WSConv2d(outChannelNumber, imageChannels, kernelSize=1, stride=1, padding=0)
            # initializing the blocks

            self.progressiveBlocks.append(progressiveBlock)
            self.rgbBlocks.append(rgbBlock)
            # appending the blocks to the list 

        self.to(device)

    def fadeIn(self, alpha:float, upscaled, out):
        """
        Fades in the output with the upscaled image for easier training.
        
        Parameters:
        alpha (float): Alpha value for the fade-in effect
        upscaled (torch.Tensor): Upscaled image
        out (torch.Tensor): Output tensor
        """
        if alpha == 1:
            return tanh(out)
        return tanh((alpha * out) + ( (1-alpha) * upscaled ))

    def forward(self, x, steps:int, alpha:float):
        """
        Proceeds through the generator.

        Parameters:
        x (torch.Tensor): Input tensor
        steps (int): Total steps taken by the generator
        alpha (float): Alpha value for the fade-in effect
        """        
        initialProgressiveOutput = self.initialProgressiveLayer(x)

        if steps == 0:
            return self.initialRGBLayer(initialProgressiveOutput)
        
        progressiveOutput = initialProgressiveOutput
        for step in range(steps):
            upscaled = F.interpolate(progressiveOutput, scale_factor=2, mode="nearest")
            block = self.progressiveBlocks[step]
            
            progressiveOutput = block(upscaled)

        rgbUpscaled = self.rgbBlocks[steps - 1](upscaled)
        rgbOutput = self.rgbBlocks[steps](progressiveOutput)

        return self.fadeIn(alpha, rgbUpscaled, rgbOutput)