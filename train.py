import torch
import torchvision.transforms.v2 as tf
from paths import TRAINING_FOLDER, TESTING_FOLDER
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from nets import make_cnn

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'using {pu}')

def train_step(dataloader: DataLoader,
               model, loss_fn, optimizer, clip_grad=0.5):
    train_loss, train_acc = 0.0, 0.0

    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(pu), y.to(pu)
        y_pred_logits = model(x)

        loss = loss_fn(y_pred_logits, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        y_pred = torch.argmax(torch.softmax(y_pred_logits, dim=-1), dim=-1)
        train_acc += (y_pred == y).sum().item()/len(y)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc

def validate_step(dataloader: DataLoader,
               model, loss_fn):
    train_loss, train_acc = 0.0, 0.0
    with torch.no_grad():
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(pu), y.to(pu)
            y_pred_logits = model(x)

            loss = loss_fn(y_pred_logits, y)
            train_loss += loss.item()

            y_pred = torch.argmax(torch.softmax(y_pred_logits, dim=-1), dim=-1)
            train_acc += (y_pred == y).sum().item()/len(y)
    
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc



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


