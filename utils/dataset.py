from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torch import Tensor
import os
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, path:str, resolution:tuple[int, int]) -> None:
        super().__init__()
        self.path = path
        self.resolution = resolution
        self.images = os.listdir(self.path)
        self.transforms = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __repr__(self) -> str:
        return f"FaceDataset({self.path} {self.resolution[0]}x{self.resolution[1]} [{len(self)}])"
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tensor:
        imageName = self.images[idx]
        imagePath = os.path.join(self.path, imageName)
        
        image = Image.open(imagePath)
        return self.transforms(image)
    
def getDataset(path:str, resolution:tuple[int, int], numWorkers:int, batchSize:int):
    """
    Returns a DataLoader for the FaceDataset.

    Parameters:
    path (str): Path to the dataset directory.
    resolution (tuple[int, int]): Desired resolution of the images.
    numWorkers (int): Number of worker processes for data loading.
    batchSize (int): Batch size for data loading.
    """
    dataset = FaceDataset(path, resolution)
    return DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=numWorkers)