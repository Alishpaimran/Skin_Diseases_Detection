import torch
import torchvision.transforms.v2 as tf
from paths import TRAINING_FOLDER, TESTING_FOLDER
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from nets import make_cnn
import torch.nn.functional as F


pu = 'cuda' if torch.cuda.is_available() else 'cpu'



# model = make_cnn(dataset=test_ds, conv_layers = conv_layers, 
#                  max_pool = [2, 2], hid_layers = [64], pooling_after_layers=2).to(pu)


# print(model)
# # img = np.expand_dims(test_ds.__getitem__(0)[0], 0)

# # print(model)

# train_acc = 0.0

# for batch, (x, y) in enumerate(test_dl):
#     if batch ==0:
#         print(y)
#         print(len(y))
        # x = x.to(pu)
        # y = y.to(pu)
        # print(x.shape)
        # y_pred = model(x)
        # print(f"{y_pred =}")
        # y_pred_prob = torch.argmax(torch.softmax(y_pred, dim=-1), dim=-1)
        # print(f'{y = }')
        # print(f'{y_pred_prob = }')

        # train_acc = (y_pred_prob == y).sum().item()/len(y)
        # print(f'{train_acc = }')


