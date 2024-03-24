from paths import TRAINING_FOLDER, TESTING_FOLDER
import torchvision.transforms.v2 as tf
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, DataLoader

test_trans = tf.Compose([
    tf.Resize([256, 256]),
    tf.RandomHorizontalFlip(),
    tf.RandomVerticalFlip(),
    tf.ToImageTensor(),
    tf.ConvertImageDtype()
])


class Dataset:
    def __init__(self, train_trans, test_trans, train_batch_size = 32, val_rat = 0.2):
        self._ds = ImageFolder(TRAINING_FOLDER, train_trans)
        self.test_ds = ImageFolder(TESTING_FOLDER, test_trans)
        ds_size = len(self._ds)
        val_size = int(val_rat*ds_size)
        train_size = len(self._ds) - val_size

        self.train_ds, self.val_ds = Subset(self._ds, range(train_size)), Subset(self._ds, range(train_size, ds_size))
        self.train_dl = DataLoader(self.train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4)
        self.test_dl = DataLoader(self.test_ds, batch_size=train_batch_size, shuffle=True, num_workers=4)
        self.val_dl = DataLoader(self.val_ds, batch_size=train_batch_size, shuffle=True, num_workers=4)

