from flask import Flask, send_from_directory, Response, jsonify
import cv2 as cv
import base64
import numpy as np
import torch
from fixcap import FixCapsNet
import time

pu = 'cuda' if torch.cuda.is_available() else 'cpu'

cam = None
cam = cv.VideoCapture(0)
_, img = cam.read()
cam.release()
size = img.shape
st_x, st_y = size[1]//2 - 151, size[0]//2 - 151
end_x, end_y = st_x+299, st_y+299

color = (0, 255, 0)
thickness = 5
mean = 0.5
std = 0.5

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

classes = ['Actnic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevus', 'Vacular Lesion']
model_path = '/home/user/Skin_Diseases_Detection/checkpoints/checkpoint_232.pth'

test_model.load_state_dict(torch.load(model_path)['model_state_dict'])
test_model.eval()

app = Flask(__name__)
curr_frame = None
@app.route('/')
def index():
    global cam
    cam = cv.VideoCapture(0)
    return send_from_directory('', 'index.html')

def gen_frames():
    while True:
        # Capture frame-by-frame
        success, frame = cam.read()
        if not success:
            break
        else:
            frame = cv.rectangle(frame, (st_x, st_y), (end_x, end_y), color, thickness)
            ret, buff = cv.imencode('.jpg', frame)
            frame=buff.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
            frame = None
            
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['GET'])
def capture_frame():
    global cam, mean, std
    _, frame = cam.read()
    frame = frame[st_y:end_y, st_x:end_x]
    _, frame_ = cv.imencode('.jpg', frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame = (np.asarray(frame)/255).astype(np.float16)
    frame = np.expand_dims(((frame - mean)/std).transpose((2, 0, 1)), 0)
    img_ = torch.tensor(frame, dtype=torch.float32).to(pu)
    output = test_model(img_)

    v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True)) 
    pred = v_mag.data.max(1, keepdim=True)[1].cpu().squeeze().item()

    result = classes[pred]
    img_data = base64.b64encode(frame_).decode()
    content = f"The Detected Disease is: {result}"
    cam.release()
    return jsonify(string=content, img=img_data)

@app.route('/result')
def processing():
    return send_from_directory('', 'processing.html')

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000)
    