from paths import TRAINING_FOLDER, TESTING_FOLDER
import torchvision.transforms.v2 as tf
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset

test_trans = tf.Compose([
    tf.Resize([256, 256]),
    tf.RandomHorizontalFlip(),
    tf.RandomVerticalFlip(),
    tf.ToImageTensor(),
    tf.ConvertImageDtype()
])

# test_ds = SD_Dataset(TESTING_FOLDER, test_trans)
# train_ds = SD_Dataset(TRAINING_FOLDER, None)



class Dataset:
    def __init__(self, train_trans, test_trans, train_batch_size = 32, val_rat = 0.2):
        self._ds = ImageFolder(TRAINING_FOLDER, train_trans)
        ds_size = len(self._ds)
        val_size = int(val_rat*ds_size)
        train_size = len(self._ds) - val_size

        self.train_ds, self.val_ds = Subset(self._ds, range(train_size)), Subset(self._ds, range(train_size, ds_size))


        print(len(self.train_ds))
        print(len(self.val_ds))

ds = Dataset(test_trans=test_trans, train_trans=test_trans)
