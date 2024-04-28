import cv2 as cv
import torch
import numpy as np
from paths import TRAINING_FOLDER
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import random
from torchvision.transforms import transforms as tf

trans = tf.Compose([
    tf.Resize(size=(64, 64)),
    tf.ToTensor()
])

val_ds = ImageFolder(TRAINING_FOLDER, trans)

images = []

for i in range(10):
    ind = random.randint(0, len(val_ds)-1)
    img = val_ds.__getitem__(ind)[0]
    images.append(img)
    # img = img.numpy()
    # img = img.transpose((1, 2, 0)) if img.shape[0] == 3 else img

images = torch.stack(images)
print(images)

print(images.mean(dim=(-1)))

mean = np.array([207.53284205, 158.27759683, 166.7383785 ])
std = np.array([33.69236129, 39.09313952, 42.81731764])


# cv.imshow('x', img)
# cv.waitKey()
# cv.destroyAllWindows()
