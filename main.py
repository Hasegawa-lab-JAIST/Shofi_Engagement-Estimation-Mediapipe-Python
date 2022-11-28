# from _typeshed import OpenTextModeUpdating
from flask import Flask, render_template, Response, request
from camera_media import VideoCamera
import os
import cv2
os.environ['KMP_DUPLICATE_LIB_OK']='True'

global capture, rec_frame, rec, out, switch
capture=0
switch=0


camera = cv2.VideoCapture(0)

app = Flask(__name__, template_folder='./templates')

@app.route('/')
def index(): #rendering webpage
    return render_template('index.html')

def gen(camera_media): ##activate VideoCamera feed
    global switch
    while switch==1: #get camera webpage
        frame = camera_media.get_frame() #get the feed frame by frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def gen_frames():  # generate frame by frame from camera
    global switch
    while switch==0:
        # Capture frame-by-frame
        _, frame = camera.read()  # read the camera frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed') #get the prediction from the VideoCamera class
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')#the prediction back to the web interface

@app.route('/video_feed0') #get the prediction from the VideoCamera class
def video_feed0():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to the buttons
@app.route('/requests', methods=['POST','GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('stop') == 'Stop':
            switch=0
            cv2.destroyAllWindows()
        elif request.form.get('start') == 'Start':
            switch=1
            cv2.destroyAllWindows()
    
    elif request.method== 'GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__': #defining server ip address and port
    app.run(host='0.0.0.0', port='5050', debug=True) #the app is running at localhost. the default port is 5000
   

''' 
ref for the buttons:
https://towardsdatascience.com/camera-app-with-flask-and-opencv-bd147f6c0eec
'''