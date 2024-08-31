from train import fitModel
import config


if __name__ == '__main__':
    args = (config.epochs, config.channels, config.batchSizes, config.resolutions, config.lr, config.latentDimension, config.device, config.numWorkers, config.datasetPath, config.savePath, config.savedGeneratorPath, config.savedDiscriminatorPath, config.savePreviewEvery, config.lambdaGP, config.transitionPhases)
    fitModel(*args)
    # unpacking argument instead of directly passing to make debugging easier 