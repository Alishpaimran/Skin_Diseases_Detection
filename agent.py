import torch
import torchvision.transforms.v2 as tf
from paths import TRAINING_FOLDER, TESTING_FOLDER
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from nets import make_cnn
import tqdm

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'using {pu}')

class skdi_detector:
    def __init__(self, model, optimizer, loss_fn, clip_grad):
        self.dataset = Dataset()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.clip_grad = clip_grad
        

    def train_step(self)
        
        train_loss, train_acc = 0.0, 0.0

        for _, (x, y) in enumerate(self.dataset.train_dl):
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

    def train(self, epochs):

        for epoch in tqdm(range(range(epochs))):
            train_loss, train_acc = 



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
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for _, (x, y) in enumerate(dataloader):
            x, y = x.to(pu), y.to(pu)
            y_pred_logits = model(x)

            loss = loss_fn(y_pred_logits, y)
            val_loss += loss.item()

            y_pred = torch.argmax(torch.softmax(y_pred_logits, dim=-1), dim=-1)
            val_acc += (y_pred == y).sum().item()/len(y)
    
    val_loss /= len(dataloader)
    val_acc /= len(dataloader)

    return val_loss, val_acc

def train(train_ds, val_ds, optimizer, )



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


