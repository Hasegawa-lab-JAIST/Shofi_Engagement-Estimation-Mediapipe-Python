# from _typeshed import OpenTextModeUpdating
from flask import Flask, render_template, Response, request
from flask_cors import CORS
from camera_media import VideoCamera
from camera_media_api import get_frame_api
import os
import cv2
import pickle
import csv
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore") # Trying to unpickle estimator Pipeline from version 0.24.0 when using version 0.24.2
os.environ['KMP_DUPLICATE_LIB_OK']='True'

global capture, rec_frame, rec, out, switch
capture=0
switch=0

camera = cv2.VideoCapture(0)

app = Flask(__name__, template_folder='./templates')
CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/')
def index(): #rendering webpage
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    # Sol2
    # landmark_from_js = request.get_json(force=True)
    # predict_from_py = get_pred(landmark_from_js)
    # Sol 3
    payload = request.get_json()
    if (payload.get('timestamp') is None):
        return {"class": 0, "prob": 0} # Error happens, return default value
    return get_frame_api(payload["encodedImage"], payload["timestamp"]) #from camera_media_api
    

def get_pred(landmark_from_js):
    with open('engagement.pkl', 'rb') as f:
        model = pickle.load(f)

    try:
        # Export coordinates
        pose = landmark_from_js["landmark_from_js"]["poseLandmarks"]
        pose_row = list(np.array([[landmark["x"], landmark["y"], landmark["z"], landmark["visibility"]] for landmark in pose]).flatten())
        face = landmark_from_js["landmark_from_js"]["faceLandmarks"]
        face_row = list(np.array([[landmark["x"], landmark["y"], landmark["z"]] for landmark in face]).flatten())

        # Concatenate rows
        row = pose_row+face_row

        #Predict images with the model
        X = pd.DataFrame([row])
        body_language_class = model.predict(X)[0]
        body_language_prob = model.predict_proba(X)[0]

        # Class
        pred = body_language_class.split(' ')[0]
        # Probability Confidence
        confi = body_language_prob[np.argmax(body_language_prob)]
        confi = " {:.1f}%".format(confi*100)
        confi = str(confi)
        print("Test", pred)
        return {"class": pred, "prob": confi}
    except Exception as e: 
        print('error', e)
        pass
    return {"class": 0, "prob": 0} # Exception occurred

# Data Collection process
# =======================================================================
    # class_name = "VeryEngaged"

    ## create header row for csv
    ## Run the following lines only once to create the dataset header
    ## =========================================================================================
    # pose = landmark_from_js["landmark_from_js"]["poseLandmarks"]
    # face = landmark_from_js["landmark_from_js"]["faceLandmarks"]
    # num_coords = len(pose)+len(face)
    # row = ['class']
    # for val in range(1,num_coords+1):
    #    row += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    # with open('engagement_from_js_2.csv', mode='w', newline='') as f:
    #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     csv_writer.writerow(row)
    ## =========================================================================================

    # try:
    #     # Export coordinates
    #     pose = landmark_from_js["landmark_from_js"]["poseLandmarks"]
    #     pose_row = list(np.array([[landmark["x"], landmark["y"], landmark["z"], landmark["visibility"]] for landmark in pose]).flatten())
    #     face = landmark_from_js["landmark_from_js"]["faceLandmarks"]
    #     face_row = list(np.array([[landmark["x"], landmark["y"], landmark["z"]] for landmark in face]).flatten())

    #     # Concatenate rows
    #     row = pose_row+face_row

    #     # # Uncomment the following lines to collect the landmarks in the dataset
    #     # # ============================================================================
    #     row.insert(0, class_name) # Append class name

    #     with open('engagement_from_js_2.csv', mode='a', newline='') as f:
    #         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #         csv_writer.writerow(row)
    
    # except:
    #     pass
    # =========================================================================

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
    app.run(host='0.0.0.0', port='5050', debug=True, threaded=True) #the app is running at localhost. the default port is 5000
   

''' 
ref for the buttons:
https://towardsdatascience.com/camera-app-with-flask-and-opencv-bd147f6c0eec
'''