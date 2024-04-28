import cv2 as cv
from fixcap import FixCapsNet
import torch
import torchvision.transforms as tf
import numpy as np
import time

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'using {pu}')


conv_outputs = 128
num_primary_units = 8 #digicaps paper 
primary_unit_size = 16 * 6 * 6
output_unit_size = 16
test_model = FixCapsNet(conv_inputs= 3,
						conv_outputs=conv_outputs,
						num_primary_units=num_primary_units,
						primary_unit_size=primary_unit_size,
						output_unit_size=output_unit_size,
						num_classes=7,
						init_weights=True,mode="128").to(pu)
												

classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
model_path = '/home/user/Skin_Diseases_Detection/model/fixcap_3_model.pth'

test_model.load_state_dict(torch.load(model_path))
test_model.eval()

start = time.time()

img = cv.imread('/home/user/skdi_dataset/base_dir/val_dir/nv/ISIC_0029490.jpg')
img = cv.resize(img, (299, 299))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = np.asarray(img, dtype=np.uint8)
trans = tf.Compose([tf.ToTensor(), tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

img_ = trans(img).unsqueeze(0).to(pu)

output = test_model(img_)

v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True)) 
pred = v_mag.data.max(1, keepdim=True)[1].cpu().squeeze().item()

end = time.time()

print(classes[pred])

print(end - start)





