# from _typeshed import OpenTextModeUpdating
from flask import Flask, render_template, Response, request
from flask_cors import CORS
from camera_media import VideoCamera
import os
import cv2
import pickle
import csv
import numpy as np
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'

global capture, rec_frame, rec, out, switch
capture=0
switch=0


camera = cv2.VideoCapture(1)

app = Flask(__name__, template_folder='./templates')
CORS(app)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/')
def index(): #rendering webpage
    return render_template('index.html')

@app.route('/api', methods=['POST'])
def api():
    landmark_from_js = request.get_json(force=True)

    class_name = "NormalEngaged"

    ## create header row for csv
    ## Run the following lines only once to create the dataset header
    ## =========================================================================================
    # pose = landmark_from_js["landmark_from_js"]["poseLandmarks"]
    # face = landmark_from_js["landmark_from_js"]["faceLandmarks"]
    # num_coords = len(pose)+len(face)
    # row = ['class']
    # for val in range(1,num_coords+1):
    #    row += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
    # with open('engagement_from_js.csv', mode='w', newline='') as f:
    #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     csv_writer.writerow(row)
    ## =========================================================================================

    try:
        # Export coordinates
        pose = landmark_from_js["landmark_from_js"]["poseLandmarks"]
        pose_row = list(np.array([[landmark["x"], landmark["y"], landmark["z"], landmark["visibility"]] for landmark in pose]).flatten())
        face = landmark_from_js["landmark_from_js"]["faceLandmarks"]
        face_row = list(np.array([[landmark["x"], landmark["y"], landmark["z"]] for landmark in face]).flatten())

        # Concatenate rows
        row = pose_row+face_row

        # Append class name
        row.insert(0, class_name)

        # Export to CSV
        with open('engagement_from_js.csv', mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row)
    
    except:
        pass

#     predict_from_py = get_pred(landmark_from_js)
#     return predict_from_py

# def get_pred(landmark_from_js):
#     with open('engagement.pkl', 'rb') as f:
#         model = pickle.load(f)
#     pose = landmark_from_js["landmark_from_js"]["poseLandmarks"]
#     pose_row = list(np.array([[landmark["x"], landmark["y"], landmark["z"], landmark["visibility"]] for landmark in pose]).flatten())
#     face = landmark_from_js["landmark_from_js"]["faceLandmarks"]
#     face_row = list(np.array([[landmark["x"], landmark["y"], landmark["z"]] for landmark in face]).flatten())            
#     row = pose_row+face_row
#     X = pd.DataFrame([row])
#     body_language_class = model.predict(X)[0]
#     body_language_prob = model.predict_proba(X)[0]
#     return {"class": body_language_class, "prob": body_language_prob}

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