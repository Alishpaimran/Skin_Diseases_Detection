import torch
import torch.nn as nn
import torchvision.transforms.v2 as tf
from paths import TRAINING_FOLDER, TESTING_FOLDER
from utils import SD_Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets 



pu = 'cuda' if torch.cuda.is_available() else 'cpu'

test_trans = tf.Compose([
    tf.Resize([256, 256]),
    tf.RandomHorizontalFlip(),
    tf.RandomVerticalFlip(),
    # tf.ToTensor()

])

# test_ds = SD_Dataset(TESTING_FOLDER, test_trans)
# train_ds = SD_Dataset(TRAINING_FOLDER, None)

test_ds = datasets.ImageFolder(TESTING_FOLDER, test_trans)



info = test_ds.__getitem__(34)

img = np.asarray(info[0])

print(img.shape)
print(test_ds.classes)
print(test_ds.extensions)
plt.imshow(img)
plt.show()


