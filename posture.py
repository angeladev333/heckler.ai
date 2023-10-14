import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def get_angle(first, mid, end):
    first = np.array(first)  # First
    mid = np.array(mid)  # Mid
    end = np.array(end)  # End

    radians = np.arctan2(end[1] - mid[1], end[0] - mid[0]) - np.arctan2(first[1] - mid[1], first[0] - mid[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


    # live video
cap = cv2.VideoCapture(0)  # default webcam

counter = 0

# set up media pose instance
with mp_pose.Pose(min_detection_confidence=0.50, min_tracking_confidence=0.5) as pose:
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

            # calculate bend of right elbow
            shoulder_left = [landmarks[11].x, landmarks[11].y]
            elbow_left = [landmarks[13].x, landmarks[13].y]
            wrist_left = [landmarks[15].x, landmarks[15].y]

            left_elbow_angle = get_angle(shoulder_left, elbow_left, wrist_left)

            # display angle
            # color in bgr
            cv2.putText(image, str(left_elbow_angle),
                        tuple(np.multiply(elbow_left, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 250), 2, cv2.LINE_AA
                        )

            # hand gesture logic
            if left_elbow_angle > 150:
                stage = "down"
            if left_elbow_angle < 60 and stage == 'down':
                stage = "up"
                counter += 1
                print(counter)

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

