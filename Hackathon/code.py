import cv2
import mediapipe as mp
import numpy as np
import math


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


zoom_out_factor = 1  

while cap.isOpened():
    ret, frame = cap.read(5)
    if not ret:
        break

    
    height, width, _ = frame.shape

    
    new_width = int(width * zoom_out_factor)
    new_height = int(height * zoom_out_factor)

    
    zoomed_out_frame = cv2.resize(frame, (new_width, new_height))

    
    start_x = (new_width - width) // 2
    start_y = (new_height - height) // 2
    frame = zoomed_out_frame[start_y:start_y + height, start_x:start_x + width]

    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = results.pose_landmarks.landmark

        
        def get_pixel_coordinates(landmark, width, height):
            return np.array([landmark.x * width, landmark.y * height])

        nose = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.NOSE], width, height)
        left_shoulder = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], width, height)
        right_shoulder = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], width, height)
        left_hip = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.LEFT_HIP], width, height)
        right_hip = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_HIP], width, height)
        right_knee = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE], width, height)
        right_ankle = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE], width, height)
        right_heel = get_pixel_coordinates(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL], width, height)
        
        shoulder_center = (left_shoulder + right_shoulder) / 2
        hip_center = (left_hip + right_hip) / 2

        neck = np.linalg.norm(nose - shoulder_center)
        body = np.linalg.norm(shoulder_center - hip_center)
        thigh = np.linalg.norm(right_hip - right_knee)
        leg_calf = np.linalg.norm(right_knee - right_ankle)
        foot = np.linalg.norm(right_ankle - right_heel)

        h_p = neck + body + thigh + leg_calf + foot

        nose_landmark = landmarks[mp_pose.PoseLandmark.NOSE]

        z_coordinate = nose_landmark.z
        M = z_coordinate * -1
        
        h_T = 0.5

        f = 200

        h_R_m = (h_p * h_T) / f * (h_T / M)
        
        h_R_cm = h_R_m * 100
 
        cv2.putText(frame, f'Height: {h_R_cm:.2f} cm', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 150), 2)

    cv2.imshow('MediaPipe Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord("e"):
        break

cap.release()
cv2.destroyAllWindows()
