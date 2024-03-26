import torch
import os
from paths import CHECKPOINT_FOLDER
import yaml

def convert(**kwargs):
    return kwargs

class Utils:
    def __init__(self):
        self.model = None
        self.model_file = ''
        self.plot_file = ''
        self.min_loss = 0

    def read_file(self, path):
        file = open(path, 'r')
        file.seek(0)
        info = file.readline()
        file.close()
        return info

    def write_file(self, path, content):
        mode = 'w'
        if path == self.plot_file:
            mode = '+a'
        file = open(path, mode=mode)
        file.write(content)
        file.close()

    def create_file(self, path):
        file = open(path, 'w')
        file.close()

    def create_checkpoint_file(self, num):
        path = f'{CHECKPOINT_FOLDER}/checkpoint_{num}.pth'
        file = open(path, 'w')
        file.close()
        return path
    
    def save_config(self, args: dict):
        if not os.path.exists(self.config_file):
            self.create_file(self.config_file)
        with open(self.config_file, 'w') as file:
            yaml.safe_dump(args, file)
        file.close()


    def check_status_file(self):
        if not os.path.exists(self.status_file):
            self.create_file(self.status_file)
        checkpath = self.read_file(self.status_file)
        epoch = 0
        if checkpath != '':
            epoch = self.load_checkpoint(checkpath)
            file = open(self.plot_file, 'r')
            lines = file.readlines()
            file = open(self.plot_file, 'w')
            file.writelines(lines[:epoch+1])
            file.close()
        else:
            file = open(self.plot_file, 'w')
            file.close()
            self.write_file(self.plot_file,'Train_loss,Train_acc,Valid_loss,Valid_acc\n')
            self.model.train()
        return epoch

    def write_plot_data(self, data:list):
        str_data = ','.join(map(str, data))
        self.write_file(self.plot_file, f'{str_data}\n')

    def save_checkpoint(self, epoch, checkpath):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'epoch': epoch
        }
        file = open(self.status_file, 'w')
        file.write(checkpath)
        file.close()
        torch.save(checkpoint, checkpath)
        print('checkpoint saved..')
    
    def load_checkpoint(self, checkpath):
        print('loading checkpoint..')
        checkpoint = torch.load(checkpath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optim_state_dict'])
        self.model.train()
        print('checkpoint loaded...')
        return checkpoint['epoch']
    
    def save_check_interval(self, epoch, interval=50):
        if not(epoch % interval) and epoch > 0:
            checkpath = self.create_checkpoint_file(epoch)
            self.save_checkpoint(epoch, checkpath)
    
    def load_model(self):
        print('loading model...')
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()
        print('model loaded...')

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file)
        print('model saved...')

    def save_best_model(self, param, acc_param=True):
        if acc_param:
           param_ = max(param, self.param)
        else:
            param_ = min(param, self.param)
        self.param = param_
        self.write_file(self.param_file, f'{param_}')
        self.save_model()

    def check_param_file(self):
        if os.path.exists(self.param_file):
            param = float(self.read_file(self.param_file))
        else:
            self.create_file(self.param_file)
            param = -1000.0
            self.write_file(self.param_file, f'{param}')
        return param