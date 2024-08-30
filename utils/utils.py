from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch import Tensor


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
    grid = make_grid(generatedImages)

    writer.add_scalar('Generator Loss', genLoss, global_step=epoch)
    writer.add_scalar('Discriminator Loss', discLoss, global_step=epoch)
    writer.add_image('Generated Images', grid, global_step=epoch, dataformats="CHW")
    writer.add_embedding(mat=mat, label_img=generatedImages, global_step=epoch)