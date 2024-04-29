from flask import Flask, send_from_directory, Response, jsonify
import cv2 as cv
import base64
import time

cam = None
cam = cv.VideoCapture(0)
_, img = cam.read()
size = img.shape
st_x, st_y = size[1]//2 - 151, size[0]//2 - 151
end_x, end_y = st_x+300, st_y+300
color = (0, 255, 0)
thickness = 5
cam.release()


app = Flask(__name__)
curr_frame = None
@app.route('/')
def index():
    global cam
    cam = cv.VideoCapture(0)
    return send_from_directory('', 'index.html')

def gen_frames():
    global curr_frame
    while True:
        # Capture frame-by-frame
        success, frame = cam.read()
        if not success:
            break
        else:
            curr_frame = frame
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
    _, frame = cam.read()
    frame = frame[st_y:end_y, st_x:end_x]
    _, frame = cv.imencode('.jpg', frame)
    img_data = base64.b64encode(frame).decode()
    content = "The Detected Disease is: "
    cam.release()
    return jsonify(string=content, img=img_data)

@app.route('/result')
def processing():
    return send_from_directory('', 'processing.html')

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000)
    