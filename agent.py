import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchvision.transforms.v2 as tf
import tqdm
from nets import make_cnn
from dataset import Dataset
from paths import STATUS_FOLDER, PLOT_FOLDER, PARAM_FOLDER, CONFIG_FOLDER, MODEL_FOLDER
from utils import Utils
import matplotlib.pyplot as plt
import random
from fixcap import FixCapsNet

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'using {pu}')

class skdi_detector(Utils):
    def __init__(self,params):
        self.params = params
        self.dataset = Dataset(train_trans=params.train_trans, test_trans=params.test_trans,
                               train_batch_size=params.train_batch_size, val_batch_size=params.val_batch_size)
        self.loss_fn = CrossEntropyLoss()
        self.name = params.name
        self.model_file = f'{MODEL_FOLDER}/{self.name}_model.pth'
        self.status_file = f'{STATUS_FOLDER}/{self.name}_status.txt'
        self.plot_file = f'{PLOT_FOLDER}/{self.name}_plot.txt'
        self.param_file = f'{PARAM_FOLDER}/{self.name}_param.txt'
        self.config_file = f'{CONFIG_FOLDER}/{self.name}_config.yaml'
        self.param = self.check_param_file()
        self.metric_param = params.metric_param
        self.clip_grad = params.clip_grad
        self.acc_param = False
        if self.metric_param in ['train_acc', 'val_acc']:
            self.acc_param = True
    
    def to_one_hot(self, x, length):
        batch_size = x.size(0)
        x_one_hot = torch.zeros(batch_size, length)
        for i in range(batch_size):
            x_one_hot[i, x[i]] = 1.0
        return x_one_hot

    def train_step(self):
        self.model.train()
        train_loss, train_acc = 0.0, 0.0
        for _, (data, target) in enumerate(self.dataset.train_dl):
            tar_indices = target
            tar_one_hot = self.to_one_hot(tar_indices, 7)
            data, target = data.to(pu), tar_one_hot.to(pu)

            self.optim.zero_grad()
            output = self.model(data)

            v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True)) 
            pred = v_mag.data.max(1, keepdim=True)[1].cpu().squeeze()
            train_acc += (pred == tar_indices.cpu()).sum().item()/len(tar_indices)
            
            loss = self.model.loss(output, target)
            loss.backward()
            # torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optim.step()
            train_loss += loss.item()

        train_loss /= len(self.dataset.train_dl)
        train_acc /= len(self.dataset.train_dl)

        return train_loss, train_acc
    
    def validate_step(self):
        self.model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for _, (data, target) in enumerate(self.dataset.val_dl):
                tar_indices = target
                tar_one_hot = self.to_one_hot(tar_indices, 7)
                
                data, target = data.to(pu), tar_one_hot.to(pu)

                output = self.model(data)
                loss = self.model.loss(output, target)
                val_loss += loss.item()

                v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True)) 
                pred = v_mag.data.max(1, keepdim=True)[1].cpu().squeeze()
                val_acc += (pred == target.cpu()).sum().item()/len(target)
        
        val_loss /= len(self.dataset.val_dl)
        val_acc /= len(self.dataset.val_dl)
        return val_loss, val_acc
    
    def create_model(self):
        if self.params.custom_model:
            conv_outputs = 128 #128_Feature_map
            num_primary_units = 8
            primary_unit_size = 16 * 6 * 6
            output_unit_size = 16
            self.model = FixCapsNet(conv_inputs= 3,
                                conv_outputs=conv_outputs,
                                num_primary_units=num_primary_units,
                                primary_unit_size=primary_unit_size,
                                output_unit_size=output_unit_size,
                                num_classes=7,
                                init_weights=True,mode="128").to(pu)
        else:
            self.model = make_cnn(dataset=self.dataset.train_ds, hid_layers=self.params.hid_layers, act_fn=self.params.act_fn,
                                max_pool=self.params.max_pool, pooling_after_layers=self.params.pool_after_layers,
                                batch_norm=self.params.batch_norm, conv_layers=self.params.conv_layers, dropout=self.params.dropout).to(pu)
        
        self.optim = Adam(self.model.parameters(), lr=self.params.lr)
        print(f'Model: {self.model}')
        print(f'Number of classes: {len(self.dataset.train_ds.classes)}')
        print(f'Input image size: {self.dataset.train_ds.__getitem__(0)[0][0].shape}')
        print(f'total number of parameters: {sum([p.numel() for p in self.model.parameters()])}')
        
    def plot_images(self, n_imgs):
        fig, axes = plt.subplots(1, n_imgs, figsize=(10, 10))
        for i in range(n_imgs):
            ind = random.randint(0, len(self.dataset.train_ds)-1)
            img, label = self.dataset.train_ds.__getitem__(ind)
            img = img.numpy()
            img = img.transpose((1, 2, 0))
            axes.flat[i].set_title(label)
            axes.flat[i].imshow(img)
            axes.flat[i].axis('off')

    def train(self):
        epochs = self.params.epochs
        epoch = 0
        epoch = self.check_status_file()

        print(f'training for {epochs - epoch} epochs....')
        for ep in tqdm(range(epoch, epochs+1)):
            train_loss, train_acc = self.train_step()
            val_loss, val_acc = self.validate_step()

            metric_param = {'train_loss': train_loss, 'train_acc': train_acc,
                            'val_loss': val_loss, 'val_acc': val_acc}
            
            print(f'epochs: {ep}\t{train_loss = :.4f}\t{train_acc = :.4f}\t{val_loss = :.4f}\t{val_acc = :.4f}')
            self.write_plot_data([train_loss, train_acc, val_loss, val_acc])
            self.save_check_interval(epoch=ep, interval=1)
            self.save_best_model(acc_param=self.acc_param, param=metric_param[self.metric_param])
        
        print('Finished Training....')


