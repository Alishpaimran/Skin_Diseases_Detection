import torchvision
torchvision.disable_beta_transforms_warning()
import matplotlib.pyplot as plt
from agent import skdi_detector
from specs import Params
import numpy as np
import torch
params = Params()


agent = skdi_detector(params)

agent.create_model()
agent.plot_images()



# img = agent.dataset._ds.__getitem__(77)[0]

# img = torch.tensor(np.expand_dims(img, 0))

# pred = agent.model(img)

# print(pred)
