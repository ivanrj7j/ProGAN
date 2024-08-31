from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch import Tensor
from src import Generator, Discriminator
import torch


def writeSummary(writer:SummaryWriter, latentInput:Tensor, generatedImages:Tensor, genLoss:float, discLoss:float,epoch:int):
    """
    Writes summary statistics to TensorBoard.

    Parameters:
    writer (SummaryWriter): TensorBoard SummaryWriter object
    latentInput (torch.Tensor): Latent input tensor
    generatedImages (torch.Tensor): Generated images tensor using generator model
    genLoss (float): The loss of the generator from the given latent input
    discLoss (float): The loss of the discriminator from the given generated images
    epoch (int): The current epoch number
    """
    mat = latentInput.mean(1).view(latentInput.size(0), -1)
    generatedImages = (generatedImages / 2) + 0.5
    grid = make_grid(generatedImages)

    writer.add_scalar('Generator Loss', genLoss, global_step=epoch)
    writer.add_scalar('Discriminator Loss', discLoss, global_step=epoch)
    writer.add_image('Generated Images', grid, global_step=epoch, dataformats="CHW")
    writer.add_embedding(mat=mat, label_img=generatedImages, global_step=epoch)

def loadModels(generatorPath:str, discriminatorPath:str, zDim:int, channels:list[int], imageChannels:int=3, device:str="cuda"):
    """
    Loads the generator and discriminator from the given paths if paths are provided, else loads a simple model

    Parameters:
    generatorPath (str): Path to the saved generator model. Does not load any weights if path = ""
    discriminatorPath (str): Path to the saved discriminator model. Does not load any weights if path = ""
    zDim (int): Dimension of the latent vector
    channels (list[int]): Number of channels of each layers for the generator
    imageChannels (int): Number of channels in the output image. Defaults to 3.
    device (str): Device to run the model on. Defaults to "cuda".
    """

    generator = Generator(zDim, channels, imageChannels, device)
    discriminator = Discriminator(channels, imageChannels, device)

    if generatorPath != "":
        print(f"Loading {generatorPath}")
        genratorWeights = torch.load(generatorPath, weights_only=False)
        generator.load_state_dict(genratorWeights)
    
    if discriminatorPath != "":
        print(f"Loading {generatorPath}")
        discriminatorWeights = torch.load(discriminatorPath, weights_only=False)
        discriminator.load_state_dict(discriminatorWeights)


    return generator, discriminator

def saveModels(generator:Generator, discriminator:Discriminator, savePath:str, version:str):
    """
    Saves the generator and discriminator models to the given save path with the given version

    Parameters:
    generator (Generator): Generator model to save
    discriminator (Discriminator): Discriminator model to save
    savePath (str): Path to save the models
    version (str): Version of the models

    Example:
    >>> saveModels(generator, discriminator, "path/to/save/models", "v1")

    Saves at:
    -   path/to/save/models/gen-v1.pth
    -   path/to/save/models/dis-v1.pth
    """
    torch.save(generator.state_dict(), f"{savePath}/gen-{version}.pth")
    torch.save(discriminator.state_dict(), f"{savePath}/dis-{version}.pth")

def gradientPenalty(discriminator:Discriminator, realImage:torch.Tensor, fakeImage:torch.Tensor, trainPhase:int, alpha:float, device="cuda"):
    """
    Calculates Gradient penalty for training

    Parameters:
    discriminator (Discriminator): Discriminator model
    realImage (torch.Tensor): Real images tensor
    fakeImage (torch.Tensor): Fake images tensor
    trainPhase (int): Current training phase (1: 4x4, 2:8x8, etc)
    alpha (float): Alpha value for fade in effect
    device (str): Device to run the model on. Defaults to "cuda".
    """

    batchSize, channels, height, width = realImage.shape
    beta = torch.rand((batchSize, 1, 1, 1)).repeat(1, channels, height, width).to(device)
    interpolatedImage = (beta * realImage) + ((1 - beta) * fakeImage.detach())
    interpolatedImage.requires_grad_(True)

    mixedScore = discriminator.forward(interpolatedImage, trainPhase, alpha)

    gradient = torch.autograd.grad(inputs=interpolatedImage, outputs=mixedScore, grad_outputs=torch.ones_like(mixedScore), create_graph=True, retain_graph=True)[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradientNorm = gradient.norm(p=2, dim=1)
    return torch.mean((gradientNorm-1)**2)