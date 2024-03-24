import torch
import torchvision.transforms.v2 as tf
from paths import TRAINING_FOLDER, TESTING_FOLDER
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from nets import make_cnn
from PIL import Image

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

test_trans = tf.Compose([
    tf.Resize([256, 256]),
    tf.RandomHorizontalFlip(),
    tf.RandomVerticalFlip(),
    tf.ToTensor()
])

conv_layers = [[128, 5, 1],
               [64, 3, 1],
               [32, 3, 1],
               [16, 3, 1]]

# test_ds = SD_Dataset(TESTING_FOLDER, test_trans)
# train_ds = SD_Dataset(TRAINING_FOLDER, None)

test_ds = datasets.ImageFolder(TESTING_FOLDER, test_trans)

test_dl = DataLoader(test_ds, batch_size=1, shuffle=True,
                     num_workers=4)

model = make_cnn(dataset=test_ds, conv_layers = conv_layers, 
                 max_pool = [2, 2], hid_layers = [64], pooling_after_layers=2)

img = np.expand_dims(test_ds.__getitem__(0)[0], 0)

print(model)

pred = model(torch.tensor(img, dtype=torch.float32))

print(pred)


