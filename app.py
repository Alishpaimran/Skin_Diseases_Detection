from flask import Flask, send_from_directory, Response, render_template_string, jsonify
import cv2 as cv
import time


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
            frame = cv.resize(frame, dsize=(400, 400))
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
    time.sleep(10)

    return 

@app.route('/result')
def processing():
    content = """<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Processing...</title>
            <style>
                body, html {
                    height: 100%;
                    margin: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    background-color: #f0f0f0; /* Optional background color */
                }

                #processingtext {
                    font-size: 48px;
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div id="processingtext"></div>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                var textContainer = document.getElementById('processingtext');
                textContainer.textContent = 'Processing...';
                });
            </script>
        </body>
        </html>"""
    return render_template_string(content)

if __name__ == '__main__':
    app.run('0.0.0.0', port=8000)
    