import os
import sys
import logging
import warnings
import cv2
import mediapipe as mp
import time
import joblib
import jaconv
import math
import numpy as np
from flask import Flask, render_template, Response

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

app = Flask(__name__)

def flatten_landmarks_1d(landmarks):
    return [coord for landmark in landmarks for coord in [landmark.x, landmark.y, landmark.z]]

def normalize_landmarks(landmarks):
    if not landmarks or len(landmarks) == 0:
        return []
    wrist = landmarks[:3]
    centered_landmarks = [landmarks[i] - wrist[i % 3] for i in range(len(landmarks))]
    max_dist = max(math.sqrt(sum([centered_landmarks[i+j]**2 for j in range(3)])) for i in range(0, len(centered_landmarks), 3))
    return [coord / max_dist for coord in centered_landmarks]

def detect_hand_landmarks(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3) as hands:
        results = hands.process(image_rgb)
        all_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2))
                landmarks = flatten_landmarks_1d(hand_landmarks.landmark)
                all_landmarks.append((landmarks, hand_label))
        return image, all_landmarks

model_path = 'svm_model_10_19.pkl'
svm_model = joblib.load(model_path)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

last_detection_time = time.time()
detection_interval = 0.01
last_prediction = "待機中..."

def generate_frames():
    global last_detection_time, last_prediction
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            current_time = time.time()
            if current_time - last_detection_time >= detection_interval:
                result_image, all_landmarks = detect_hand_landmarks(frame)
                if all_landmarks:
                    for landmarks, hand_label in all_landmarks:
                        normalized_landmarks = normalize_landmarks(landmarks)
                        X_test = [np.array(normalized_landmarks)]
                        y_pred = svm_model.predict(X_test)
                        last_prediction = jaconv.alphabet2kana(y_pred[0])
                else:
                    last_prediction = "検出なし"
                last_detection_time = current_time
            else:
                result_image = frame

            ret, buffer = cv2.imencode('.jpg', result_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return last_prediction

if __name__ == '__main__':
    app.run(debug=True, port=8080)