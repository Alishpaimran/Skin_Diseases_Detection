import torch
import torch.nn as nn
from torchvision import datasets, transforms
from paths import TRAINING_FOLDER, TESTING_FOLDER
from utils import SD_Dataset
import matplotlib.pyplot as plt
from PIL import Image


pu = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = SD_Dataset(TESTING_FOLDER, None)


info = dataset.__getitem__(66)

print(info[1])
img = info[0]

img.show('img')


