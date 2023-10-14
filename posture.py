import cv2
import mediapipe as mp
import numpy as np
import time
import pickle
import pandas as pd

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

def isFacingForward(a, b):
    if ((a < 0 and b > 0) or (a > 0 and b < 0)):
        # print("FACING DIAGONAL!")
        return False
    elif (a > 0 and b > 0) or (a < 0 and b < 0):
        if abs(a - b) > 0.37:
            # print("FACING DIAGONAL!")
            return False
        else:
            # print("FACING FORWARD")
            return True

    return True

# getting ml face model
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# live video
cap = cv2.VideoCapture(0)  # default webcam

frames = []
left_wrist_frames = []
right_wrist_frames = []
front_slouch_frames = []
start_time = 0

# set up media pose instance
with mp_pose.Pose(min_detection_confidence=0.50, min_tracking_confidence=0.5) as pose, mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    start_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()  # getting frames (img) from video feed
        frames.append(frame) # adding frame to list of frames

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # reorder colour array to rgb for mediapipe
        image.flags.writeable = False  # save memory by setting to not writeable

        # Make detection
        results_pose = pose.process(image)
        results_face = holistic.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # reorder to bgr to put it back into opencv



        # extract landmarks
        try:
            landmarks = results_pose.pose_landmarks.landmark
            # landmark format in landmarks
            # landmarks[index] (index represents a point)
            # landmarks[0].x (gets x coordinate of nose)
            # x: 0.638852
            # y: 0.671197
            # z: 0.129959
            # visibility: 0.9999997615814209 # % likely of it being visible (basically checking if mediapipe can but a dot on it)
            #print(landmarks)

            # calculate bend of torso/waist
            # left_knee = [landmarks[25].x, landmarks[25].y]
            left_waist = [landmarks[23].x, landmarks[23].y]
            shoulder_left = [landmarks[11].x, landmarks[11].y]

            right_shoulder = [landmarks[12].x, landmarks[12].y]
            right_waist = [landmarks[24].x, landmarks[24].y]
            # right_knee = [landmarks[26].x, landmarks[26].y]

            wrist_left = [round(landmarks[15].x, 2), round(landmarks[15].y, 2), round(landmarks[15].z, 2)]
            left_wrist_frames.append(wrist_left)

            wrist_right = [round(landmarks[16].x, 2), round(landmarks[16].y, 2), round(landmarks[16].z, 2)]
            right_wrist_frames.append(wrist_right)

            # display text on image
            cv2.putText(image, str(wrist_left),
                        tuple(np.multiply([wrist_left[0], wrist_left[1]], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(wrist_right),
                        tuple(np.multiply([wrist_right[0], wrist_right[1]], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2, cv2.LINE_AA
                        )

            shoulder = (landmarks[11].z + landmarks[12].z) / 2  # getting average shoulder z value
            # print("shoulder =", shoulder)

            waist = (landmarks[23].z + landmarks[24].z) / 2  # getting average waist z value
            # print("waist =", waist)

            cv2.putText(image, str(shoulder),
                        tuple(np.multiply([shoulder_left[0], shoulder_left[1]], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2, cv2.LINE_AA
                        )

            cv2.putText(image, str(waist),
                        tuple(np.multiply([left_waist[0], left_waist[1]], [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2, cv2.LINE_AA
                        )

            if isFacingForward(landmarks[12].z, landmarks[11].z):
                shoulder = (landmarks[11].z + landmarks[12].z) / 2
                print("shoulder =", shoulder)

                waist = (landmarks[23].z + landmarks[24].z) / 2
                print("waist =", waist)

                knee = (landmarks[25].z + landmarks[26].z) / 2
                print("knee =", knee)

                if abs(shoulder) / abs(waist) > 750:
                    print("shoulders slouched! SLOUCHER! FOUND THE SLOUCHER!")
                    front_slouch_frames.append(frame)

            # pose detection (Angela)
            if results_face.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results_face.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # face detection
            if results_face.pose_landmarks and results_face.face_landmarks:
                num_coords = len(results_face.pose_landmarks.landmark) + \
                             len(results_face.face_landmarks.landmark)
                landmarks = ['class']

                for val in range(1, num_coords + 1):
                    landmarks += ['x{}'.format(val), 'y{}'.format(val),
                                  'z{}'.format(val), 'v{}'.format(val)]

            # angela's pose stuff
            pose_marks = results_face.pose_landmarks.landmark
            pose_row = list(np.array(
                [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_marks]).flatten())

            # Extract face landmarks
            face = results_face.face_landmarks.landmark
            face_row = list(np.array(
                [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # Concate rows
            row = pose_row + face_row

            # Make face detections
            X = pd.DataFrame([row])
            body_language_class = model.predict(X.values)[0]
            body_language_prob = model.predict_proba(X.values)[0]
            # print(body_language_class, body_language_prob)

            # Grab ear coords
            coords = tuple(
                np.multiply(np.array((results_face.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                                      results_face.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)),
                            [640, 480]).astype(int))

            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(image, 'CLASS', (95, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Display Probability
            cv2.putText(image, 'PROB', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except:
            pass

        # draw face landmarks
        # mp_drawing.draw_landmarks(image, results_face.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        #                           mp_drawing.DrawingSpec(color=(
        #                               80, 110, 0), thickness=1, circle_radius=1),
        #                           mp_drawing.DrawingSpec(color=(100, 100, 10), thickness=1, circle_radius=1))

        # draw pose landmarks
        mp_drawing.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  # change colours of dots and lines (connections) in bgr format
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

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

# # i = index of gesture in frame array (aka frame number)
# for i in gesture_index:
#     cv2.imshow("asdf", frames[i])
#     cv2.waitKey(0)

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

for f in front_slouch_frames:
    cv2.imshow("shouching", f)
    cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
