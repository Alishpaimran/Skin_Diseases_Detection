import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchvision.transforms.v2 as tf
import tqdm
from nets import make_cnn
from dataset import Dataset
from paths import STATUS_FOLDER, PLOT_FOLDER, PARAM_FOLDER, CONFIG_FOLDER
from utils import Utils

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'using {pu}')

class skdi_detector(Utils):
    def __init__(self,name, ):
        self.dataset = Dataset()
        self.loss_fn = CrossEntropyLoss()
        self.name = name
        self.status_file = f'{STATUS_FOLDER}/{self.name}_status.txt'
        self.plot_file = f'{PLOT_FOLDER}/{self.name}_plot.txt'
        self.param_file = f'{PARAM_FOLDER}/{self.name}_param.txt'
        self.config_file = f'{CONFIG_FOLDER}/{self.name}_config.yaml'
        self.param = self.check_param_file()
        self.metric_param = metric_param
        self.clip_grad = clip_grad
        self.acc_param = False
        if self.metric_param in ['train_acc', 'val_acc']:
            self.acc_param = True

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
    
    
    def create_model(self):
        self.model = make_cnn(self.dataset.train_ds, self.hid_layers, self.act_fn,
                              self.max_pool, self.pool_after_layers, self.conv_layers)
        self.optimzer = Adam(self.model.parameters(), lr=self.lr, eps=1e-6, weight_decay=1e-5)
        print(self.model)


    def train(self, epochs):
        epoch = 0
        epoch = self.check_status_file()
        print(f'training for {epochs} epochs....')
        for ep in tqdm(range(range(epoch, epochs+1))):
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.validate_step()

            metric_param = {'train_loss': train_loss, 'train_acc': train_acc,
                            'val_loss': val_loss, 'val_acc': val_acc}
            
            print(f'epochs: {ep}\t{train_loss = :.4f}\t{train_acc = :.4f}\t{val_loss = :.4f}\t{val_acc = :.4f}')
            self.write_plot_data([train_loss, train_acc, val_loss, val_acc])
            self.save_check_interval(epoch=ep, interval=1)
            self.save_best_model(acc_param=self.acc_param, param=metric_param[self.metric_param])
        
        print('Finished Training....')




