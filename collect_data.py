import cv2
import os

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

cap = cv2.VideoCapture(0)
counters = {name: len(os.listdir(os.path.join(dataset_path, name))) for name in classes.values()}

print("Press 1-5 to save image for each class, q to quit")
print("1: thumbs_up, 2: thumbs_down, 3: peace, 4: open_palm, 5: no_hand")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    y_pos = 30
    for key, name in classes.items():
        text = f"{key}: {name} ({counters[name]})"
        cv2.putText(display, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_pos += 25

    cv2.imshow('Collect Data', display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif chr(key) in classes:
        class_name = classes[chr(key)]
        filename = f"{class_name}_{counters[class_name]}.jpg"
        filepath = os.path.join(dataset_path, class_name, filename)
        cv2.imwrite(filepath, frame)
        counters[class_name] += 1
        print(f"Saved {filepath}")

cap.release()
cv2.destroyAllWindows()
