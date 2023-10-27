import pickle
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

labels_dict = {0: 'Gojo', 1: 'Mahito', 2: 'Thumbs Up!'}

with mp_hands.Hands(min_detection_confidence = 0.8, min_tracking_confidence = 0.5) as hands:
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
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)
                        x_.append(x)
                        y_.append(y)

                x1 = int(min(x_) * W)
                y1 = int(min(y_) * H)
                
                x2 = int(max(x_) * W)
                y2 = int(max(y_) * H)

                try:
                    prediction = model.predict([np.asarray(data_aux)])

                    predicted_gesture = labels_dict[int(prediction[0])]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_gesture, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                except ValueError:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, 'Gojo', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow('Hand Gesture Classifier', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()