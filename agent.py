import torch
import torchvision.transforms.v2 as tf
from paths import TRAINING_FOLDER, TESTING_FOLDER
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from nets import make_cnn
import tqdm
from dataset import Dataset

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'using {pu}')

class skdi_detector:
    def __init__(self, model, optimizer, loss_fn, clip_grad):
        self.dataset = Dataset()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.clip_grad = clip_grad
        

    def train_step(self):
        train_loss, train_acc = 0.0, 0.0
        for _, (x, y) in enumerate(self.dataset.train_dl):
            x, y = x.to(pu), y.to(pu)
            y_pred_logits = self.model(x)

            loss = self.loss_fn(y_pred_logits, y)
            train_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

            y_pred = torch.argmax(torch.softmax(y_pred_logits, dim=-1), dim=-1)
            train_acc += (y_pred == y).sum().item()/len(y)
        
        train_loss /= len(self.dataset.train_dl)
        train_acc /= len(self.dataset.train_dl)

        return train_loss, train_acc
    
    def validate_step(self):
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for _, (x, y) in enumerate(self.dataset.val_dl):
                x, y = x.to(pu), y.to(pu)
                y_pred_logits = self.model(x)

                loss = self.loss_fn(y_pred_logits, y)
                val_loss += loss.item()

                y_pred = torch.argmax(torch.softmax(y_pred_logits, dim=-1), dim=-1)
                val_acc += (y_pred == y).sum().item()/len(y)
        
        val_loss /= len(self.dataset.val_dl)
        val_acc /= len(self.dataset.val_dl)

        return val_loss, val_acc


    def train(self, epochs):

        self.model.train()
        print(f'training for {epochs} epochs....')
        for epoch in tqdm(range(range(epochs))):
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.validate_step()

            print(f'epochs: {epoch}\t{train_loss = }\t{train_acc = }\t{val_loss = }\t{val_acc = }')



