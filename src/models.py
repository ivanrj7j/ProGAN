import torch.nn as nn
from src.generator import Generator 

class ProGAN(nn.Module):
    """
    # ProGAN model

    This model loads a trained `Generator` and scraps the useless layers.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)