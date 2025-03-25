
import cv2 
import mediapipe as mp
import joblib
from preprocess_landmark import preprocess_landmarks

model_path = "models/rf_model.pkl"
model = joblib.load(model_path)  

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

capture = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not capture.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


            landmark_vector = preprocess_landmarks(hand_landmarks.landmark)
            if landmark_vector is None:
                continue  

            prediction = model.predict([landmark_vector])[0]

            cv2.putText(image, f"Prediction: {prediction}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Landmarks", image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()