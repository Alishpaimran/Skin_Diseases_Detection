import pandas as pd
import matplotlib.pyplot as plt




data = pd.read_csv('/home/user/Skin_Diseases_Detection/plots/fixcap_5_plot.txt')

_, ax = plt.subplots(2)

ax[0].plot(data['Valid_loss'], label='Testing Loss')
ax[0].plot(data['Train_loss'], label = 'Training Loss')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(data['Valid_acc'], label='Testing Accuracy')
ax[1].plot(data['Train_acc'], label = 'Training Accuracy')
ax[1].set_ylabel('Accuracy')
ax[1].legend()


# plt.legend()
plt.xlabel('Epochs')
plt.show()

