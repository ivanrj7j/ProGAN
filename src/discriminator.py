import torch.nn as nn
from src.blocks import WSConv2d, ConvBlock
from torch import std, ones, cat


class Discriminator(nn.Module):
    """
    ## Discriminator (aka Critic)

    Discriminator takes a generated image and passes it through its layers and return an single value
    Discriminator model is the exact mirror image of the `Generator` Model.

    Generator model is made up of 2 parts:

    ### RGB Heads:

    The RGB heads are responsible for 2 things:
    1. Acting as the output head for each size the generator produce
    2. Acting as the input for the next RGB Module

    RGB heads except the last two can be dumped after training the model.

    ### Progressive Heads:

    Each progressive heads are used as the previous layer of the rgb head.
    """

    def __init__(self, channels: list[int], imageChannels: int = 3, device:str="cuda", *args, **kwargs) -> None:
        """
        Initializes the discriminator.

        Parameters:
        channels (list[int]): Number of channels of each layers.
        imageChannels (int): Number of channels in the output image. Defaults to 3.
        device (str): Device to run the model on. Defaults to "cuda".

        Example:
        >>> channels = [8, 16, 32, 64, 128, 256, 256, 256, 256]
        >>> discriminator = Discriminator(channels, 3)
        """
        super().__init__(*args, **kwargs)
        self.device = device

        self.leaky = nn.LeakyReLU(0.2)
        self.avgPool = nn.AvgPool2d(kernel_size=2, stride=2)
        # initaializing leaky relu and average pool layers

        self.progressiveBlocks = nn.ModuleList()
        self.rgbBlocks = nn.ModuleList()

        for i, inChannelNumber in enumerate(channels[:-1]):
            outChannelNumber = channels[i+1]
            # getting the channel number 

            progressiveBlock = ConvBlock(inChannelNumber, outChannelNumber, False)
            rgbBlock = WSConv2d(imageChannels, inChannelNumber, 1, 1, 0)
            # initializing blocks 

            self.progressiveBlocks.append(progressiveBlock)
            self.rgbBlocks.append(rgbBlock)

        self.finaRGBBlock = WSConv2d(imageChannels, outChannelNumber, 1, 1, 0)
        self.rgbBlocks.append(self.finaRGBBlock)
        # initializing and appending the final block

        self.finalProgressiveBlock = nn.Sequential(
            WSConv2d(outChannelNumber+1, outChannelNumber, kernelSize=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(outChannelNumber, outChannelNumber, kernelSize=4, padding=0, stride=1),
            nn.LeakyReLU(0.2),
            WSConv2d(outChannelNumber, 1, kernelSize=1, padding=0, stride=1)
        )
        # final progressive block 

        self.to(device)

    def fadeIn(self, alpha:float, downscaled, out):
        """
        Fades in the downscaled image with the output image.

        Parameters:
        alpha (float): Weight for the fade in.
        downscaled (torch.Tensor): Downscaled image.
        out (torch.Tensor): Output tensor
        """
        if alpha == 1:
            return out
        
        return (alpha * out) + ( (1-alpha) * downscaled )
    
    def stdMiniBatch(self, x):
        """
        Calculates the standard deviation over the mini-batch for each channel and appends to the input.
        
        If input Shape is `(n, m)` then the output shape will be `(n, m+1)`
        """
        stdShape = list(x.shape)
        stdShape[1]  = 1
        batchStat = ones(stdShape).to(self.device) * std(x, dim=0).mean()
        # making the std layer

        return cat((x, batchStat), 1)
    
    def forward(self, x, steps:int, alpha:float):
        """
        Proceeds through the discriminator.

        Parameters:
        x (torch.Tensor): Input tensor
        steps (int): Total steps taken by the discriminator
        alpha (float): Alpha value for the fade-in effect
        """
        currentStep = len(self.progressiveBlocks) - steps

        firstRGBOut = self.rgbBlocks[currentStep](x)
        firstRGBOut = self.leaky(firstRGBOut)
        # passing through the first rgb layer 

        if steps == 0:
            out = self.stdMiniBatch(firstRGBOut)
            return self.finalProgressiveBlock(out).view(out.shape[0], -1)
        
        
        downScaled = self.leaky(self.rgbBlocks[currentStep + 1](self.avgPool(x)))
        # downscaling the image using rgb blocks
        
        
        firstProgressiveOut = self.avgPool(self.progressiveBlocks[currentStep](firstRGBOut))
        firstProgressiveOut = self.fadeIn(alpha, downScaled, firstProgressiveOut)
        # fading in the output and downscaled image
        
        progressiveOut = firstProgressiveOut
        for step in range(currentStep+1, len(self.progressiveBlocks)):
            progressiveOut = self.progressiveBlocks[step](progressiveOut)
            progressiveOut = self.avgPool(progressiveOut)
            # passing the output through the progression block and downscaling 

        out = self.stdMiniBatch(progressiveOut)

        return self.finalProgressiveBlock(out).view(out.shape[0], -1)