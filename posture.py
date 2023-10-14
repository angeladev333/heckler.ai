import cv2
import mediapipe as mp
import numpy as np
import time
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def vector_change(frames):
    angle_changes = []

    for i in range(1, len(frames)):
        v1 = frames[i - 1]
        v2 = frames[i]
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        angle_changes.append(angle)

    threshold = 0.15
    significant_changes = [i for i, angle in enumerate(angle_changes) if angle > threshold]  # gets index of frame for gesture
    return significant_changes

    # live video
cap = cv2.VideoCapture(0)  # default webcam

frames = []
left_wrist_frames = []
right_wrist_frames = []
start_time = 0

# set up media pose instance
with mp_pose.Pose(min_detection_confidence=0.50, min_tracking_confidence=0.5) as pose:
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()  # getting frames (img) from video feed
        frames.append(frame) # adding frame to list of frames

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

            wrist_left = [round(landmarks[15].x, 2), round(landmarks[15].y, 2), round(landmarks[15].z, 2)]
            left_wrist_frames.append(wrist_left)

            wrist_right = [round(landmarks[16].x, 2), round(landmarks[16].y, 2), round(landmarks[16].z, 2)]
            right_wrist_frames.append(wrist_right)

            cv2.putText(image, str(wrist_left),
                        tuple(np.multiply([wrist_left[0], wrist_left[1]], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(wrist_right),
                        tuple(np.multiply([wrist_right[0], wrist_right[1]], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2, cv2.LINE_AA
                        )

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

end_time = time.time()
elapse_time = end_time - start_time
time_per_frame = elapse_time / len(frames)

print(elapse_time)
print(time_per_frame)
cap.release()
cv2.destroyAllWindows()

gesture_index = vector_change(left_wrist_frames) + vector_change(right_wrist_frames)
gesture_index = sorted(gesture_index)

print(gesture_index)

# list of a frame index where there are no gestures for a long period of time
absent_gesture = []

# i = index of gesture in frame array (aka frame number)
for i in gesture_index:
    cv2.imshow("asdf", frames[i])
    cv2.waitKey(0)

# finding where are no gestures
if len(gesture_index) == 1 and elapse_time > 13:
    absent_gesture.append(int(gesture_index[0] / 2))  # code is a bit scuffed
else:
    for i in range(1, len(gesture_index)):
        delta_time = (gesture_index[i] - gesture_index[i - 1]) * time_per_frame
        if delta_time > 7: # if there is more than 7 seconds in between each hand gesture, we append an image to the absent_gesture list
            absent_gesture.append(int((gesture_index[i] + gesture_index[i - 1]) / 2))

for i in absent_gesture:
    cv2.imshow("no gesture", frames[i])
    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
