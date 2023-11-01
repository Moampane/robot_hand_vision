import mediapipe as mp
import cv2
import os
import pickle
import matplotlib.pyplot as plt

from helpers import get_max_dim, get_min_dim

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "./data"

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # for hand_landmarks in results.multi_hand_landmarks: # for multiple hands
            hand_landmarks = results.multi_hand_landmarks[0]  # for one hand

            x_min = get_min_dim(hand_landmarks, "x")
            y_min = get_min_dim(hand_landmarks, "y")
            x_max = get_max_dim(hand_landmarks, "x")
            y_max = get_max_dim(hand_landmarks, "y")

            for i in range(len(hand_landmarks.landmark)):
                x = (hand_landmarks.landmark[i].x - x_min) / (x_max - x_min)
                y = (hand_landmarks.landmark[i].y - y_min) / (y_max - y_min)
                data_aux.append(x)
                data_aux.append(y)

            data.append(data_aux)
            labels.append(dir_)

f = open("new_data.pickle", "wb")
pickle.dump({"data": data, "labels": labels}, f)
f.close()
