import cv2
import os
import mediapipe as mp

# script for collecting training images from webcam
dataset_path = "dataset"
classes = {
    '1': 'thumbs_up',
    '2': 'thumbs_down',
    '3': 'peace',
    '4': 'open_palm',
    '5': 'no_hand'
}

for class_name in classes.values():
    os.makedirs(os.path.join(dataset_path, class_name), exist_ok=True)

# setup mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
counters = {name: len(os.listdir(os.path.join(dataset_path, name))) for name in classes.values()}

print("Press 1-5 to save image for each class, q to quit")
print("1: thumbs_up, 2: thumbs_down, 3: peace, 4: open_palm, 5: no_hand")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # extract hand bounding box with 20px padding
    hand_crop = None
    if results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0]
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]

        x_min = int(min(x_coords) * w) - 20
        x_max = int(max(x_coords) * w) + 20
        y_min = int(min(y_coords) * h) - 20
        y_max = int(max(y_coords) * h) + 20

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        hand_crop = frame[y_min:y_max, x_min:x_max]

    y_pos = 30
    for key, name in classes.items():
        text = f"{key}: {name} ({counters[name]})"
        cv2.putText(display, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25

    if hand_crop is None:
        cv2.putText(display, "No hand detected", (10, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Collect Data', display)

    key = cv2.waitKey(1) & 0xFF

    # save image when 1-5 pressed, q to quit
    if key == ord('q'):
        break
    elif chr(key) in classes:
        class_name = classes[chr(key)]
        if class_name == 'no_hand':
            filename = f"{class_name}_{counters[class_name]}.jpg"
            filepath = os.path.join(dataset_path, class_name, filename)
            cv2.imwrite(filepath, frame)
            counters[class_name] += 1
            print(f"Saved {filepath}")
        elif hand_crop is not None:
            filename = f"{class_name}_{counters[class_name]}.jpg"
            filepath = os.path.join(dataset_path, class_name, filename)
            cv2.imwrite(filepath, hand_crop)
            counters[class_name] += 1
            print(f"Saved {filepath}")
        else:
            print("No hand detected, not saving")

cap.release()
cv2.destroyAllWindows()
hands.close()
