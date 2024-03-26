import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as tf
from warnings import warn


train_ds = tf.Compose([
    tf.Resize((128, 128)),
    tf.RandomHorizontalFlip(),
    tf.ColorJitter()
    
])


print(train_ds)
