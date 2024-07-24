import cv2
import mediapipe as mp

# Initialize Mediapipe Hands and Drawing utilities
hands = mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Function to count fingers
def count_fingers(hand_landmarks, hand_label):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_label == 'Right':
        fingers.append(hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x)
    else:
        fingers.append(hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x)

    # Other four fingers
    fingers.extend(hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y for i in range(1, 5))
    return fingers.count(True)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            fingers_count = count_fingers(hand_landmarks, hand_label)
            cv2.putText(frame, f'{hand_label} Hand Fingers: {fingers_count}', (10, 30 + idx*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Finger Counter', frame)
    if cv2.waitKey(10) == 27:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
