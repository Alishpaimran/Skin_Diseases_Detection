import torchvision.transforms.v2 as tf
from utils import convert

params = convert(
    name = 'example_model',
    model_specs = {'conv_layers': [[128, 3, 1],
                                   [64, 3, 1],
                                   [32, 3, 1],
                                   [16, 3, 1]],
                    'hid_layers': [1024, 512], 
                    'max_pool': [2, 2],
                    'pool_after_layers': 2,
                    'act_fn': 'relu'},
    lr = 3e-6,
    epochs = 150,
    metric_param = 'val_acc',
    clip_grad = 0.3
)
