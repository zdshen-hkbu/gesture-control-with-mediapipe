import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import ctypes
import screen_brightness_control as sbc

# Initialize MediaPipe
mp_hands = mp.solutions.hands.Hands(max_num_hands=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
recognition_result = None
last_timestamp_ms = 0


def process_recognition_result(result: mp.tasks.vision.GestureRecognizerResult, output_image: mp.Image,
                               timestamp_ms: int):
    global recognition_result
    recognition_result = result


# Load gesture recognizer model
try:
    with open('gesture_recognizer.task', 'rb') as f:
        model = f.read()
    base_options = mp.tasks.BaseOptions(model_asset_buffer=model)
    options = mp.tasks.vision.GestureRecognizerOptions(base_options=base_options,
                                                       running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
                                                       result_callback=process_recognition_result)  # Callback provided
    recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(options)
except RuntimeError as e:
    print(f"Error loading gesture recognizer model: {e}")
    exit()

# Screen resolution
user32 = ctypes.windll.user32
screen_width = user32.GetSystemMetrics(0)
screen_height = user32.GetSystemMetrics(1)

# Webcam setup
cap = cv2.VideoCapture(0)

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands and Gesture Recognizer
    results = mp_hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            # Gesture Recognition
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            recognizer.recognize_async(mp_image, time.time_ns() // 1000000)  # Use corrected timestamp
            if recognition_result and recognition_result.gestures:  # Check if result is available
                gesture = recognition_result.gestures[0][0].category_name
                cv2.putText(image, gesture, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # Hand Laterality Check
                hand_label = results.multi_handedness[0].classification[0].label

                if gesture == "Victory":
                    x = int(hand_landmarks.landmark[
                                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].x * screen_width)
                    y = int(hand_landmarks.landmark[
                                mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP].y * screen_height)
                    pyautogui.moveTo(x, y)
                elif gesture == "Pointing_Up":
                    pyautogui.click(button='left')
                elif gesture == "ILoveYou":
                    pyautogui.click(button='right')
                elif gesture == "Thumb_Up":
                    if hand_label == "Right":
                        pyautogui.press("volumeup")
                    else:
                        sbc.set_brightness("+1")
                elif gesture == "Thumb_Down":
                    if hand_label == "Right":
                        pyautogui.press("volumedown")
                    else:
                        sbc.set_brightness("-1")

    # Convert the image back to BGR.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
