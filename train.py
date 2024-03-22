import torch
import torchvision.transforms.v2 as tf
from paths import TRAINING_FOLDER, TESTING_FOLDER
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from nets import make_cnn

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

test_trans = tf.Compose([
    tf.Resize([256, 256]),
    tf.RandomHorizontalFlip(),
    tf.RandomVerticalFlip(),
    tf.ToImageTensor()
])



# test_ds = SD_Dataset(TESTING_FOLDER, test_trans)
# train_ds = SD_Dataset(TRAINING_FOLDER, None)

test_ds = datasets.ImageFolder(TESTING_FOLDER, test_trans)

test_dl = DataLoader(test_ds, batch_size=1, shuffle=True,
                     num_workers=4)

info = test_ds.__getitem__(5)
img = info[0]
model = make_cnn(input_shape=img.shape, num_of_classes=len(test_ds.classes), max_pool = [2, 2], hid_layers = [64])

print(img.shape)
print(model)

print(model(img))

