"""
Shofiyati Nur Karimah 

Capture image from webcam and predict the engagement state
"""
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import csv
from sklearn.metrics import accuracy_score  # accuracy metrics
import pickle
from datetime import datetime, time
import base64
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
font = cv2.FONT_HERSHEY_SIMPLEX
start_time = datetime.now()
time_format = "{:%H:%M:%S}"
extension = "csv"
prefix = "log/log_engagement"
filename_format = "{:s}-{:%Y%m%d_%H%M}.{:s}"
filename = filename_format.format(prefix, start_time, extension)
header = ["Time", "States", "Probability"]
with open("models/engagement.pkl", "rb") as f:
    model = pickle.load(f)
def get_frame_api(encodedData, timeStamp):  # extracting frames
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        try:
            # Write image
            imageName = "image/" + timeStamp + ".png"
            if not encodedData:
                return {"class": 0, "prob": 0} # If error happens
            with open(imageName, "wb") as fh:
                fh.write(base64.b64decode(encodedData))
            # Read image
            src = cv2.imread(imageName)
            if src is None:
                return {"class": 0, "prob": 0} # If error happens
            image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                            for landmark in pose]).flatten())
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                            for landmark in face]).flatten())
            row = pose_row+face_row
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            confi = body_language_prob[np.argmax(body_language_prob)]
            confi = " {:.1f}%".format(confi*100)
            confi = str(confi)
            return {"class": body_language_class, "prob": confi}
        except Exception as e: 
            print('error', e)
            return {"class": 0, "prob": 0} # Exception occurred
