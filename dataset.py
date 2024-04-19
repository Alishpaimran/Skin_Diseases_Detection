from paths import TRAINING_FOLDER, VALIDATION_FOLDER
import torchvision.transforms.v2 as tf
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader
import os

workers = os.cpu_count()

class Dataset:
    def __init__(self, train_trans, test_trans, train_batch_size = 32, val_batch_size=31):
        self.train_ds = ImageFolder(TRAINING_FOLDER, train_trans)
        self.val_ds = ImageFolder(VALIDATION_FOLDER, test_trans)

        self.train_dl = DataLoader(self.train_ds, batch_size=train_batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        self.val_dl = DataLoader(self.val_ds, batch_size=val_batch_size, shuffle=True, num_workers=workers, pin_memory=True)

