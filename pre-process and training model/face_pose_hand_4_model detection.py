import mediapipe as mp
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score #accuracy metrics
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Load model
with open('engagement.pkl', 'rb') as f:
    model = pickle.load(f)

# Holistic Model (output: pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks)
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        # by setting writeable to False, it prevent copying the image data but able to use the same image for rendering
        image.flags.writeable = False 

        # Make detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        # print(results.pose_landmarks)

        # # pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks
        # Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(
            image, 
            results.face_landmarks, 
            mp_holistic.FACE_CONNECTIONS,
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
        
        # Export coordinates
        try:
            # Extract pose landmark
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                    for landmark in pose]).flatten())
            # print(pose)
            
            # Extract face landmark
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                    for landmark in face]).flatten())
            
            # Concatenate rows
            row = pose_row+face_row
            # print(row)
            
            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_proba(X)[0]
            # print(body_language_class, body_language_prob)
            
            # # Grab ear coords # 640,480 is the camera frame size
            # .flatten() is a function to collapse an array into a single dimension. e.g. [[1,2],[3,4]] becomes [1,2,3,4]
            # ===========================================================================================================
            # coords = tuple(np.multiply(
            #                 np.array(
            #                     (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
            #                      results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
            #             , [640,480]).astype(int))
            
            # #Display the label on left ear
            # cv2.rectangle(image, 
            #               (coords[0], coords[1]+5), 
            #               (coords[0]+len(body_language_class)*20, coords[1]-30),
            #               (245,117,16), -1)
            # cv2.putText(image, body_language_class, coords,
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            # =============================================================================================================

            # Get status box
            cv2.rectangle(image, (0,0), (250,60), (245,117,16), -1)
            
            #Display class
            cv2.putText(image, 'CLASS',
                        (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1,cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0],
                        (95,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),2,cv2.LINE_AA)
            
            #Display probabilitu
            cv2.putText(image, 'PROB',
                        (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2)), 
                        (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255),2, cv2.LINE_AA)
        
        except:
            pass

        cv2.imshow('Raw Webcam Feed', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

