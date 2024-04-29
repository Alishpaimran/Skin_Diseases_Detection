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
model_path = '/home/user/Skin_Diseases_Detection/checkpoints/checkpoint_232.pth'

test_model.load_state_dict(torch.load(model_path)['model_state_dict'])
test_model.eval()

start = time.time()

img = cv.imread('/home/user/skdi_dataset/base_dir/val_dir/vasc/ISIC_0029439.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

print(img.shape)
size = img.shape
st_x, st_y = size[1]//2 - 151, size[0]//2 - 151
end_x, end_y = st_x+299, st_y+299
img = img[st_y:end_y, st_x:end_x]

img = (np.asarray(img)/255).astype(np.float16)
mean = 0.5
std = 0.5
img = np.expand_dims(((img - mean)/std).transpose((2, 0, 1)), 0)

# img = cv.resize(img, (299, 299))
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# img = np.asarray(img, dtype=np.uint8)
# trans = tf.Compose([tf.ToTensor(), tf.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

# img_ = trans(img).unsqueeze(0).to(pu)
img_ = torch.tensor(img, dtype=torch.float32).to(pu)

output = test_model(img_)

v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True)) 
pred = v_mag.data.max(1, keepdim=True)[1].cpu().squeeze().item()

end = time.time()

print(classes[pred])

print(end - start)





