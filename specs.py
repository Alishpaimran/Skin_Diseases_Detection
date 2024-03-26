from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
import torchvision.transforms.v2 as tf
from utils import convert


class Params:
    def __init__(self):
        self = Params()
        self.model_specs = Params()
        self.name = 'example_model'
        self.conv_layers = [[128, 3, 1],
                            [64, 3, 1],
                            [32, 3, 1],
                            [16, 3, 1]]
        self.hid_layers = [1024, 512]
        self.max_pool = [2, 2]
        self.pool_after_layers = 2
        self.act_fn = 'relu'
        self.lr = 1e-6
        self.epochs = 150
        self.clip_grad = 0.5
        self.metric_param = 'val_acc'
        self.batch_size = 32
        self.test_trans = None
        self.train_trans = None
        self.val_rat = 0.2