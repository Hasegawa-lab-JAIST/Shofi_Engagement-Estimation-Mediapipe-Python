'''
Shofiyati Nur Karimah 

Capture image from webcam and predict the engagement state
'''

import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import csv
from sklearn.metrics import accuracy_score #accuracy metrics
import pickle
from datetime import datetime, time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
font = cv2.FONT_HERSHEY_SIMPLEX

start_time = datetime.now()
time_format = "{:%H:%M:%S}"
extension = "csv"
prefix = 'log/log_engagement'
filename_format = "{:s}-{:%Y%m%d_%H%M}.{:s}"
filename = filename_format.format(prefix, start_time, extension)
header = ["Time", "States", "Probability"]

# Create Header
# ========================================================================================
with open(filename, mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(header)

# Load model
with open('engagement.pkl', 'rb') as f:
    model = pickle.load(f)

class VideoCamera(object):
    def __init__(self): #capturing video        
        self.video = cv2.VideoCapture(1) #the source of video

    def __del__(self): #releasing camera
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self): #extracting frames
        with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
            while self.video.isOpened():
                success , image = self.video.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                # Make detections
                results = holistic.process(image)
                # print(results.face_landmarks)

                # Convert back to BGR
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 1. Draw face landmarks
                mp_drawing.draw_landmarks(
                    image, 
                    results.face_landmarks, 
                    mp_holistic.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0,200,0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0,200,0), thickness=1, circle_radius=1))

                # 2. Right hand
                mp_drawing.draw_landmarks(
                    image=image, 
                    landmark_list=results.right_hand_landmarks, 
                    connections=mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))

                # 3. Left Hand
                mp_drawing.draw_landmarks(
                    image, 
                    results.left_hand_landmarks, 
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=2, circle_radius=2))

                # 4. Pose Detector
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_holistic.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,0,200), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,0,200), thickness=2, circle_radius=2))
                
                # Export coordinate and estimate
                try:
                    # Extract pose landmark
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                            for landmark in pose]).flatten())
                    
                    # Extract face landmark
                    face = results.face_landmarks.landmark
                    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                            for landmark in face]).flatten())
                    
                    # # Concatenate rows
                    row = pose_row+face_row
                    
                    #Predict images with the model
                    X = pd.DataFrame([row])
                    body_language_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]
                    # print(body_language_class, body_language_prob)
                    
                    # Get status box
                    cv2.rectangle(image, (0,0), (400,100), (117,117,16), -1)
                    
                    #Display class
                    pred = body_language_class.split(' ')[0]
                    cv2.putText(image, 'CLASS',
                                (150,35), font, 1, (0,0,255),2,cv2.LINE_AA)
                    cv2.putText(image, pred,
                                (150,75), font, 1, (255,255,255),2,cv2.LINE_AA)
                    
                    #Display probability
                    confi = body_language_prob[np.argmax(body_language_prob)]
                    confi = " {:.1f}%".format(confi*100)
                    confi = str(confi)
                    cv2.putText(image, 'PROB',
                                (15,35), font, 1, (0,0,255),2, cv2.LINE_AA)
                    cv2.putText(image, confi, 
                                (0,75), font, 1, (255,255,255),2, cv2.LINE_AA)
                    
                    # # Export to CSV
                    # # ==============================================================================
                    tic = datetime.now()
                    tic_format = str(time_format.format(tic))
                    with open(filename, mode='a', newline='') as f:
                        fieldnames = ['Time', 'State', 'Confidence']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        tic = datetime.now()
                        tic_format = time_format.format(tic)
                        writer.writerow({'Time':str(tic_format), 'State':pred, 'Confidence':confi})

                except:
                    pass

                _, jpeg = cv2.imencode('.jpg', image)
                return jpeg.tobytes()