from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
import torchvision.transforms.v2 as tf
from agent import skdi_detector, pu

class Params:
    def __init__(self):
        self.name = 'fixcap_5'
        self.custom_model = True
        self.schedular = False
        self.conv_layers = [[128, 11, 2],
                            [128, 3, 2],
                            [512, 3, 2],
                            [128, 5, 1],
                            [128, 3, 1],
                            [512, 3, 1]]
                            # [512, 5, 1]]
                            # [128, 3, 1],
                            # [256, 3, 1],
                            # [512, 3, 1]]
                            # [512, 3, 1],
                            # [512, 3, 1]]
        self.max_pool = [2, 2]
        self.pool_after_layers = 3
        self.act_fn = 'relu'
        self.batch_norm = False
        self.dropout = None
        self.hid_layers = [256, 256]
        self.lr = 0.15
        self.epochs = 300
        self.clip_grad = 0.5
        self.metric_param = 'val_acc'
        self.train_batch_size = 168
        self.val_batch_size = 67
        self.test_trans = tf.Compose([
            tf.Resize((308,308)),
            tf.CenterCrop((299, 299)),
            tf.ToTensor(),
            tf.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5])
        ])
        self.train_trans = tf.Compose([
            tf.RandomResizedCrop((299, 299)),
            tf.ToTensor(),
            tf.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
        ])
params = Params()

agent = skdi_detector(params)

agent.create_model()
print(f'training dataset size: {len(agent.dataset.train_ds)}')
print(f'validation dataset size: {len(agent.dataset.val_ds)}')

agent.train()



