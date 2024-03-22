import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class SD_Dataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))


    def __len__(self):
        return sum(len(files for dirp, dirn, files in os.walk(self.root_dir)))
    
    def __getitem__(self):
        pass