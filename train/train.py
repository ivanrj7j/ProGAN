from src import Generator, Discriminator
from torch import GradScaler
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import gradientPenalty, getDataset, writeSummary, saveModels, loadModels
import os.path as path
from tqdm import tqdm
import time

def trainStep(realImages:torch.Tensor, latentNoise:torch.Tensor, generator:Generator, discriminator:Discriminator, trainPhase:int, scaler:GradScaler, alpha:float, genOpt:Adam, discOpt:Adam, device:str="cuda", lambdaGP:float|int=10):
    """
    Performs a single training step.

    Parameters:
    realImages (torch.Tensor): Tensor of real images
    latentNoise (torch.Tensor): Tensor of latent noise
    generator (Generator): Generator model
    discriminator (Discriminator): Discriminator model
    trainPhase (int): Current training phase (1: 4x4, 2:8x8, etc)
    scaler (GradScaler): Gradient scaler for stability
    alpha (float): Alpha value for the fade-in effect
    genOpt (Adam): Generator optimizer
    discOpt (Adam): Discriminator optimizer
    device (str): Device to use for training ("cuda" or "cpu"). Defaults to "cuda"
    lambdaGP (float|int): Weight for gradient penalty. Defaults to 10.
    """
    with torch.autocast(device):
        generatedImages = generator.forward(latentNoise, trainPhase, alpha)
        discRealOutput = discriminator.forward(realImages, trainPhase, alpha)
        discFakeOutput = discriminator.forward(generatedImages.detach(), trainPhase, alpha)
        # generating images and calculating adverserial loss 

        gp = gradientPenalty(discriminator, realImages, generatedImages, trainPhase, alpha, device)
        # calculating grdient penalty 

        discLoss = (torch.mean(discFakeOutput) - torch.mean(discRealOutput) + (lambdaGP * gp) + (1e-3 * torch.mean(discRealOutput**2)))
        # calculating discriminator's loss 

    discOpt.zero_grad()
    scaler.scale(discLoss).backward()
    scaler.step(discOpt)
    scaler.update()
    # backward propogation for disscriminator 

    with torch.autocast(device):
        genFake = discriminator.forward(generatedImages, trainPhase, alpha)
        genLoss = -torch.mean(genFake)
        # calculating generator's loss

    genOpt.zero_grad()
    scaler.scale(genLoss).backward()
    scaler.step(genOpt)
    scaler.update()
    # backward propogation for generator

    return torch.tensor([genLoss.item(), discLoss.item()])


def trainPhase(generator:Generator, discriminator:Discriminator, trainPhase:int, epochs:int, batchSize:int, resolution:tuple[int, int], dataPath:str, numWorkers:int, latentDimension:int, scaler:GradScaler, genOpt:Adam, discOpt:Adam, checkPointPath:str, device:str="cuda", lambdaGP:float|int=10, transitionPhaseCap:float=0.8, checkpointEvery:int=5):
    """
    Performs a single training phase.

    Parameters:
    generator (Generator): Generator model
    discriminator (Discriminator): Discriminator model
    trainPhase (int): Current training phase (1: 4x4, 2:8x8, etc)
    epochs (int): Number of epochs to train
    batchSize (int): Batch size for current training phase
    resolution (tuple[int, int]): Resolution of the images (height, width)
    dataPath (str): Path to the dataset
    numWorkers (int): Number of workers for data loading
    latentDimension (int): Dimension of the latent vector
    scaler (GradScaler): Gradient scaler for stability
    genOpt (Adam): Generator optimizer
    discOpt (Adam): Discriminator optimizer
    checkPointPath (str): Path to save the checkpoint
    device (str): Device to use for training ("cuda" or "cpu"). Defaults to "cuda"
    lambdaGP (float|int): Weight for gradient penalty. Defaults to 10.
    transitionPhaseCap (float): Determines the number of epochs(in fraction to total epochs) fade in effect is used. Defaults to 0.8.
    checkpointEvery (int): Determines after how many epochs to save the checkpoint. Defaults to 5.
    """
    loader = getDataset(dataPath, (resolution, resolution), numWorkers, batchSize)
    writer = SummaryWriter(path.join("runs", str(round(time.time())), f"Phase-{trainPhase}"))
    
    alpha = 1e-4

    for epoch in range(1, epochs+1):
        loop = tqdm(loader, f"[{epoch}/{epochs} alpha={round(alpha, 4)}]", len(loader), leave=False, unit="batch")
        losses = torch.zeros(2)
        for images in loop:
            realImages = images.to(device)
            latentNoise = torch.randn(((realImages.size(0), latentDimension, 1, 1)), device=device)

            result = trainStep(realImages, latentNoise, generator, discriminator, trainPhase, scaler, alpha, genOpt, discOpt, device, lambdaGP)
            losses += result
            resultDict = {"genLoss": float(losses[0]/epoch), "discLoss": float(losses[1]/epoch)}
            loop.set_postfix(resultDict)

        losses /= len(loader)


        with torch.no_grad():
            latentNoise = torch.randn((4, latentDimension, 1, 1), device=device)
            generatedImages = generator.forward(latentNoise, trainPhase, alpha)
            writeSummary(writer, latentNoise, generatedImages, losses[0], losses[1], epoch)
        
        alpha = min(1, epoch/round(epochs*transitionPhaseCap))

        if epoch % checkpointEvery == 0:
            saveModels(generator, discriminator, checkPointPath, f"{trainPhase}-{epoch}")

    writer.close()
    saveModels(generator, discriminator, checkPointPath, f"{trainPhase}-{epochs}-final")

    return generator, discriminator

def fitModel(epochs:list[int], channels:list[int], batchSizes:list[int], resolutions:list[int], lr:float, latentDimensions:int, device:str, numWorkers:int, datasetPath:str, checkPointPath:str, savedGeneratorPath:str="", savedDiscriminatorPath:str="", savePreviewEvery:int=5, lambdaGP:float|int=10, trainsitionPhases:float=0.8):
    """
    Fits the model to the given dataset.
    """
    assert len(epochs) == len(batchSizes) == len(resolutions), "len(channels) == len(epochs) == len(batchSizes) == len(resolutions) should be True"

    assert len(channels) - 1  == len(epochs), "There should be one more channel for number of epochs"

    scaler = GradScaler()
    generator, discriminator = loadModels(savedGeneratorPath, savedDiscriminatorPath, latentDimensions, channels, 3, device)
    genOpt = Adam(generator.parameters(), lr=lr)
    discOpt = Adam(discriminator.parameters(), lr=lr)
    # initializing generator, discriminator, scaler and optimizers 

    for phase, (epoch, batchSize, resolution) in enumerate(zip(epochs, batchSizes, resolutions)):
        generator, discriminator = trainPhase(generator, discriminator, phase, epoch, batchSize, resolution, datasetPath, numWorkers, latentDimensions, scaler, genOpt, discOpt, checkPointPath, device, lambdaGP, trainsitionPhases)