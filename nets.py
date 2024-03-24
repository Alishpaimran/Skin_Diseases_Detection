import torch.nn as nn
from torchvision.datasets import ImageFolder


def make_cnn(dataset: ImageFolder, hid_layers = [64, 64],
            act_fn='relu', max_pool = None, pooling_after_layers = 2,
            conv_layers=[[32, 3, 1],
                         [16, 3, 1]]):
    
    img = dataset.__getitem__(0)[0]
    input_shape = img.shape
    num_of_classes = len(dataset.classes)
    layers = []
    activation_fun = {'relu': nn.ReLU(), 'softplus':nn.Softplus(), 'tanh':nn.Tanh(), 'elu': nn.ELU()}

    assert pooling_after_layers < len(conv_layers), 'exceeding the number conv layers..'

    in_chann, inp_h, inp_w = input_shape
    for ind, conv in enumerate(conv_layers):
        out_chann, filter_size, stride = conv
        layers.append(nn.Conv2d(in_chann, out_chann, filter_size, stride))
        layers.append(activation_fun[act_fn])

        out_h = (inp_h - filter_size)//stride + 1
        out_w = (inp_w - filter_size)//stride + 1
        inp_h = out_h
        inp_w = out_w

        if max_pool is not None and ((ind+1) % pooling_after_layers == 0 or ind == (len(conv_layers) - 1)):
            layers.append(nn.MaxPool2d(max_pool[0], max_pool[1]))
            out_h = (inp_h - max_pool[0])//max_pool[1] + 1
            out_w = (inp_w - max_pool[0])//max_pool[1] + 1
            inp_h = out_h
            inp_w = out_w
        in_chann = out_chann

    layers.append(nn.Flatten())
    layers.append(nn.Linear(inp_h*inp_w*in_chann, hid_layers[0]))
    layers.append(activation_fun[act_fn])

    
    if len(hid_layers) > 1:
        dim_pairs = zip(hid_layers[:-1], hid_layers[1:])
        for in_dim, out_dim in list(dim_pairs):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(activation_fun[act_fn])

    layers.append(nn.Linear(hid_layers[-1], num_of_classes))
    # layers.append(nn.Softmax())

    return nn.Sequential(*layers)

        