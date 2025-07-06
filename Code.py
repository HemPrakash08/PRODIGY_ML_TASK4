import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

import numpy as np
import pandas as pd

def extract_landmarks(results):
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()
    return None

data = []
labels = []

cap = cv2.VideoCapture(0)
gesture = "thumbs_up"  # change for each gesture
while len(data) < 500:  # 500 samples per gesture
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    landmarks = extract_landmarks(results)
    if landmarks is not None:
        data.append(landmarks)
        labels.append(gesture)
    
    # Optional: Show frame
    cv2.imshow("Collecting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))

joblib.dump(model, "gesture_model.pkl")

model = joblib.load("gesture_model.pkl")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    landmarks = extract_landmarks(results)
    if landmarks is not None:
        prediction = model.predict([landmarks])[0]
        cv2.putText(frame, f"Gesture: {prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

