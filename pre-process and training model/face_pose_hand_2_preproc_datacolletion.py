# Data Collection. Output: dataset.csv with class names, pose, and face landmark parameter.
# =========================================================================================

import mediapipe as mp
import cv2
import csv
import os
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Change the class name to collect the data with specific class name
# ==================================================================
class_name = "NotEngaged"

# Holistic Model (output: pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks)
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()

        # Recolor feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Make detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        # print(results.pose_landmarks)

        num_coords = len(results.face_landmarks.landmark)+len(results.pose_landmarks.landmark)

        # # create header row for csv
        # # Run the following lines only once to create the dataset header
        # =========================================================================================
        # row = ['class']
        # for val in range(1,num_coords+1):
        #     row += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        # with open('engagement.csv', mode='w', newline='') as f:
        #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     csv_writer.writerow(row)
        # =========================================================================================

        # # pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks
        # Recolor image back to BGR for rendering
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

        ## Export coordinates
        try:
            # Extract pose landmark
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                    for landmark in pose]).flatten())
            
            # Extract face landmark
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                    for landmark in face]).flatten())
            
            # Concatenate rows
            row = pose_row+face_row
            
            # Append class name
            row.insert(0, class_name)
            
            # Export to CSV
            with open('engagement.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
        
        except:
            pass

        cv2.imshow('Raw Webcam Feed', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


