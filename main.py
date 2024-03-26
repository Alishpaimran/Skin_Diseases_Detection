from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
import torchvision.transforms.v2 as tf
from utils import convert, Params

params = Params()
params.model_specs = Params()
params.name = 'example_model'
params.model_specs.conv_layers = [[128, 3, 1],
                                  [64, 3, 1],
                                  [32, 3, 1],
                                  [16, 3, 1]]
params.model_specs.hid_layers = [1024, 512]
params.model_specs.max_pool = [2, 2]
params.model_specs.pool_after_layers = 2
params.model_specs.act_fn = 'relu'
params.lr = 1e-6
params.epochs = 150
params.clip_grad = 0.5
params.metric_param = 'val_acc'



print(params.__dict__)

# params = convert(
#     name = 'example_model',
#     model_specs = {'conv_layers': [[128, 3, 1],
#                                    [64, 3, 1],
#                                    [32, 3, 1],
#                                    [16, 3, 1]],
#                     'hid_layers': [1024, 512], 
#                     'max_pool': [2, 2],
#                     'pool_after_layers': 2,
#                     'act_fn': 'relu'},
#     lr = 3e-6,
#     epochs = 150,
#     metric_param = 'val_acc',
#     clip_grad = 0.3
# )
