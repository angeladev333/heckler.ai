import mediapipe as mp
import cv2
import csv
import numpy as np
import pickle
import pandas as pd


# with open('body_language.pkl', 'rb') as f:
#     model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)


cap = cv2.VideoCapture(0)

# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detections
        results = holistic.process(image)

        # Recolor image backto BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
        # if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(
            80, 110, 0), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(100, 100, 10), thickness=1, circle_radius=1))

        # Pose detection
        # if results.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # get landmark coordinates
        # if results.pose_landmarks and results.face_landmarks:
        #     num_coords = len(results.pose_landmarks.landmark) + \
        #         len(results.face_landmarks.landmark)
        #     landmarks = ['class']
        #
        #     for val in range(1, num_coords + 1):
        #         landmarks += ['x{}'.format(val), 'y{}'.format(val),
        #                       'z{}'.format(val), 'v{}'.format(val)]

        if results.face_landmarks:
            num_coords = len(results.face_landmarks.landmark)
            landmarks = ['class']

            for val in range(1, num_coords + 1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val),
                              'z{}'.format(val), 'v{}'.format(val)]


        # ----------------------------------------------------------------------------------
        # with open('coords.csv', mode='w', newline='') as f:
        #     csv_writer = csv.writer(
        #         f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     csv_writer.writerow(landmarks)
        # ----------------------------------------------------------------------------------

        class_name = "Disengaged"

        # Export coordinates
        try:
            # Extract pose landmarks
            # pose = results.pose_landmarks.landmark
            # pose_row = list(np.array(
            #     [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # Extract face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array(
                [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # Concate rows
            # row = pose_row + face_row
            row = face_row

            # ----------------------------------------------------------------------------------
            # Append class name
            row.insert(0, class_name)

            # Export to CSV
            with open('coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(
                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)
            # ----------------------------------------------------------------------------------

            # ----------------------------------------------------------------------------------
            # # Make Detections
            # X = pd.DataFrame([row])
            # body_language_class = model.predict(X.values)[0]
            # body_language_prob = model.predict_proba(X.values)[0]
            # # print(body_language_class, body_language_prob)

            # # Grab ear coords
            # coords = tuple(np.multiply(np.array((results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
            #                results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)), [640, 480]).astype(int))

            # # cv2.rectangle(image, (coords[0], coords[1]+5), (coords[0]+len(
            # #     body_language_class)*20, coords[1]-30), (245, 117, 16), -1)
            # # cv2.putText(image, body_language_class, coords,
            # #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # # Get status box
            # cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            # # Display Class
            # cv2.putText(image, 'CLASS', (95, 12),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, body_language_class.split(' ')[0], (90, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # # Display Probability
            # cv2.putText(image, 'PROB', (15, 12),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10, 40),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # ----------------------------------------------------------------------------------

        except:
            pass

        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
