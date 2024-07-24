import cv2
import mediapipe as mp
import pyautogui
from math import sqrt, atan2, degrees
from pynput.mouse import Controller

# Initialize the mouse controller and screen size
mouse = Controller()
screen_width, screen_height = pyautogui.size()

# Initialize Mediapipe Hands solution
hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

def get_distance(point1, point2):
    return sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def get_angle(a, b, c):
    angle = degrees(atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]))
    return angle + 360 if angle < 0 else angle

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks = processed.multi_hand_landmarks[0]
        return (
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
        )
    return None, None, None

def move_mouse(index_finger_tip):
    if index_finger_tip:
        x, y = int(index_finger_tip.x * screen_width), int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(x, y)

def is_left_click(thumb_tip, index_finger_tip):
    return get_distance((thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y)) < 0.05

def is_right_click(landmark_list, thumb_tip, index_finger_tip):
    thumb_index_dist = get_distance((thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y))
    return (
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) > 90 and
        thumb_index_dist > 0.05
    )

def is_double_click(landmark_list, thumb_tip, index_finger_tip):
    thumb_index_dist = get_distance((thumb_tip.x, thumb_tip.y), (index_finger_tip.x, index_finger_tip.y))
    return (
        get_angle(landmark_list[5], landmark_list[6], landmark_list[8]) < 50 and
        get_angle(landmark_list[9], landmark_list[10], landmark_list[12]) < 50 and
        thumb_index_dist > 0.05
    )

def is_scroll_down(index_finger_tip, middle_finger_tip):
    return (
        get_distance((index_finger_tip.x, index_finger_tip.y), (middle_finger_tip.x, middle_finger_tip.y)) < 0.05 and
        index_finger_tip.y > middle_finger_tip.y
    )

def is_scroll_up(index_finger_tip, middle_finger_tip):
    return (
        get_distance((index_finger_tip.x, index_finger_tip.y), (middle_finger_tip.x, middle_finger_tip.y)) < 0.05 and
        index_finger_tip.y < middle_finger_tip.y
    )

def detect_gesture(frame, landmark_list, processed):
    if len(landmark_list) >= 21:
        index_finger_tip, thumb_tip, middle_finger_tip = find_finger_tip(processed)

        if is_left_click(thumb_tip, index_finger_tip):
            pyautogui.click()
            cv2.putText(frame, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if is_right_click(landmark_list, thumb_tip, index_finger_tip):
            pyautogui.click(button='right')
            cv2.putText(frame, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if is_double_click(landmark_list, thumb_tip, index_finger_tip):
            pyautogui.doubleClick()
            cv2.putText(frame, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        if is_scroll_down(index_finger_tip, middle_finger_tip):
            pyautogui.scroll(-10)
            cv2.putText(frame, "Scroll Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if is_scroll_up(index_finger_tip, middle_finger_tip):
            pyautogui.scroll(10)
            cv2.putText(frame, "Scroll Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        move_mouse(index_finger_tip)

def main():
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            landmark_list = []
            if processed.multi_hand_landmarks:
                hand_landmarks = processed.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                for lm in hand_landmarks.landmark:
                    landmark_list.append((lm.x, lm.y))

            detect_gesture(frame, landmark_list, processed)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
