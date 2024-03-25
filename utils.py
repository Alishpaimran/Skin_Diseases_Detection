import torch
import os
from paths import CHECKPOINT_FOLDER


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
        with open(self.config_file, 'w') as file:
            yaml.safe_dump(args, file)

    def check_status_file(self):
        if not os.path.exists(self.status_file):
            self.create_file(self.status_file)
        checkpath = self.read_file(self.status_file)
        epoch = 0
        if checkpath != '':
            epoch = self.load_checkpoint(checkpath) + 1
            file = open(self.plot_file, 'r')
            lines = file.readlines()
            file = open(self.plot_file, 'w')
            file.writelines(lines[:epoch+1])
            file.close()
        else:
            file = open(self.plot_file, 'w')
            file.close()
            self.write_file(self.plot_file, 'Train_loss,Train_acc,Valid_loss,Valid_acc\n')
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
        if self.net_is_shared:
            self.model.load_state_dict(torch.load(self.model_file))
            self.model.eval()
        else:
            model = torch.load(self.model_file)
            self.actor.load_state_dict(model['actor_state_dict'])
            self.critic.load_state_dict(model['critic_state_dict'])
            self.actor.eval()
            self.critic.eval()
        print('model loaded...')

    def save_model(self):
        if not self.net_is_shared:
            model = {
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict()
            }
            torch.save(model, self.model_file)
        else:
            torch.save(self.model.state_dict(), self.model_file)
        print('model saved...')

    def save_best_model(self, rewards):
        if rewards > self.max_rewards:
            self.max_rewards = rewards
            self.write_file(self.reward_file, f'{rewards}')
            self.save_model()

    def check_rewards_file(self):
        if os.path.exists(self.reward_file):
            reward = float(self.read_file(self.reward_file))
        else:
            self.create_file(self.reward_file)
            reward = -1000.0
            self.write_file(self.reward_file, f'{reward}')
        return reward