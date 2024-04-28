from flask import Flask, send_from_directory, Response
import cv2 as cv


app = Flask(__name__)
cam = cv.VideoCapture(0)
curr_frame = None
@app.route('/')
def index():
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
            ret, buff = cv.imencode('.jpg', frame)
            frame=buff.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n')
            
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['GET'])
def capture_frame():
    global curr_frame
    cv.imwrite('cap_frame.jpg', curr_frame)
    return 'Frame Captured'

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000)
    