import pandas as pd
import matplotlib.pyplot as plt




data = pd.read_csv('/home/user/Skin_Diseases_Detection/plots/model_plot.txt')


plt.plot(data['Valid_acc'], label='Validation accuracy')
plt.plot(data['Train_acc'], label = 'Training accuracy')
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('epochs')
plt.show()

