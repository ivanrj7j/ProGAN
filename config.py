import torch

epochs = [50, 50, 50, 50, 40, 30, 20, 10]
# number of epochs for each progressive steps 

channels = [256, 256, 256, 128, 64, 32, 16, 8] 
# number of channels for the biggest layer in each step, the channel corresponds to respective epoch 
#generates upto 256x256 images

batchSizes = [64, 64, 64, 64, 32, 16, 8, 4] 
# batchSizes corresponds to the batchSize of each resolution 

assert len(channels) == len(epochs) == len(batchSizes), "Number of epochs should be equal to the number of channels specified and batchSizes where each epoch corresponds to respective channels and batch size"

lr = 1e-3

latentDimension = 128
# latentDimension for initializing the input 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

testNoise = torch.randn(8, latentDimension, 1, 1).to(device)
# noise for benchmarking model 

numWorkers = 3

loadModels = False

savedGeneratorPath = ""
savedDiscriminatorPath = ""
# path to the saved generator and discriminator 

savePreviewEvery = 5
savePath = ""
# determining how often and where to save the models

lambdaGP = 10
# determines the GP weight 

datasetPath = ""