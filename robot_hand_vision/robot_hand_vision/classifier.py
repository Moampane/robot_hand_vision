import pickle
import cv2
import mediapipe as mp
import numpy as np
from helpers import get_max_dim, get_min_dim

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model_dict = pickle.load(open("robot_hand_vision/robot_hand_vision/model.pickle", "rb"))
model = model_dict["model"]

gesture_threshold = 0.1

cap = cv2.VideoCapture(0)

labels_dict = {
    0: "Tog-Teleop",
    1: "Forward",
    2: "Backwards",
    3: "Right",
    4: "Left",
    5: "Stop",
    6: "Spin",
    7: "P-Following",
    8: "Gojo",
}

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                x_min = get_min_dim(hand_landmarks, "x")
                y_min = get_min_dim(hand_landmarks, "y")
                x_max = get_max_dim(hand_landmarks, "x")
                y_max = get_max_dim(hand_landmarks, "y")

                for i in range(len(hand_landmarks.landmark)):
                    x = (hand_landmarks.landmark[i].x - x_min) / (x_max - x_min)
                    y = (hand_landmarks.landmark[i].y - y_min) / (y_max - y_min)
                    data_aux.append(x)
                    data_aux.append(y)

            x1 = int(x_min * W)
            y1 = int(y_min * H)

            x2 = int(x_max * W)
            y2 = int(y_max * H)

            try:
                prediction = list(model.predict_proba([np.asarray(data_aux)])[0])
                max_percent = max(prediction)

                if max_percent > gesture_threshold:
                    max_idx = prediction.index(max_percent)
                    predicted_gesture = labels_dict[max_idx]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(
                        frame,
                        f"{predicted_gesture} {max_percent*100}%",
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )

                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(
                        frame,
                        f"N/A {max_percent*100}%",
                        (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 0, 0),
                        3,
                        cv2.LINE_AA,
                    )

            except ValueError:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    frame,
                    "Gojo",
                    (x1, y1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

        cv2.imshow("Hand Gesture Classifier", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
