import torch

epochs = [50, 80, 100, 200, 100, 80, 50]
# number of epochs for each progressive steps 

channels = [256, 256, 256, 128, 64, 32, 16, 8] 
# number of channels for the biggest layer in each step, the channel corresponds to respective epoch 

batchSizes = [64, 64, 64, 32, 16, 8, 1] 
# batchSizes corresponds to the batchSize of each resolution 

resolutions = [4, 8, 16, 32, 64, 128, 256]
# resolutions at each epochs 

assert len(epochs) == len(batchSizes) == len(resolutions), "len(channels) == len(epochs) == len(batchSizes) == len(resolutions) should be True"

assert len(channels) - 1  == len(epochs), "There should be one more channel for number of epochs"

lr = 1e-3

latentDimension = 128
# latentDimension for initializing the input 

device = "cuda" if torch.cuda.is_available() else "cpu"

testNoise = torch.randn(8, latentDimension, 1, 1).to(device)
# noise for benchmarking model 

numWorkers = 3

savedGeneratorPath = ""
savedDiscriminatorPath = ""
# path to the saved generator and discriminator 

savePreviewEvery = 5
savePath = "checkpoints"
# determining how often and where to save the models

lambdaGP = 10
# determines the GP weight 

datasetPath = "data/train"

transitionPhases = 0.8
# total epochs in each resolution phase in which fade in is used in fraction form

startPhase = 3