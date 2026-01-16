import torch
import cv2
import numpy as np
import mediapipe as mp
import os
from model import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['thumbs_up', 'thumbs_down', 'peace', 'open_palm', 'no_hand']

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "model_all_augmentations.pth")


def load_model(model_path=DEFAULT_MODEL_PATH):
    model = SimpleCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def preprocess_frame(frame):
    image = cv2.resize(frame, (64, 64))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image).unsqueeze(0).to(device)


def predict(model, frame):
    input_tensor = preprocess_frame(frame)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction = probabilities.max(1)
    return class_names[prediction.item()], confidence.item()


def run_live_demo(model_path=DEFAULT_MODEL_PATH):
    model = load_model(model_path)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Live demo started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture = "no_hand"
        confidence = 1.0

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

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            hand_crop = frame[y_min:y_max, x_min:x_max]
            if hand_crop.size > 0:
                gesture, confidence = predict(model, hand_crop)

        label = f"{gesture} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Hand Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH
    print(f"Loading model: {model_path}")
    run_live_demo(model_path)
