import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def hand_gesture(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


    # live video
cap = cv2.VideoCapture(0)  # default webcam

# set up media pose instance
with mp_pose.Pose(min_detection_confidence=0.55, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()  # getting frames (img) from video feed

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # reorder colour array to rgb for mediapipe
        image.flags.writeable = False  # save memory by setting to not writeable

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # reorder to bgr to put it back into opencv

        # extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            # landmark format in landmarks
            # landmarks[index] (index represents a point)
            # landmarks[0].x (gets x coordinate of nose)
            # x: 0.638852
            # y: 0.671197
            # z: 0.129959
            # visibility: 0.9999997615814209 # % likely of it being visible (basically checking if mediapipe can but a dot on it)
            #print(landmarks)
        except:
            pass

        # draw detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  # change colours of dots and lines (connections) in bgr format
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):  # press q to quit
            #print(landmarks[21].visibility)
            break

cap.release()
cv2.destroyAllWindows()

