import cv2
import numpy as np
import random

def horizontal_flip(image):
    return cv2.flip(image, 1)

def rotate(image, angle=None):
    if angle is None:
        angle = random.uniform(-30, 30)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def adjust_brightness(image, factor=None):
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def gaussian_blur(image, kernel_size=None):
    if kernel_size is None:
        kernel_size = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def color_jitter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def zoom(image, factor=None):
    if factor is None:
        factor = random.uniform(1.0, 1.3)
    h, w = image.shape[:2]
    new_h, new_w = int(h * factor), int(w * factor)
    resized = cv2.resize(image, (new_w, new_h))
    start_x = (new_w - w) // 2
    start_y = (new_h - h) // 2
    return resized[start_y:start_y + h, start_x:start_x + w]

def apply_augmentations(image, augment_list):
    result = image.copy()
    for aug_name in augment_list:
        if aug_name == 'flip':
            result = horizontal_flip(result)
        elif aug_name == 'rotate':
            result = rotate(result)
        elif aug_name == 'brightness':
            result = adjust_brightness(result)
        elif aug_name == 'blur':
            result = gaussian_blur(result)
        elif aug_name == 'color':
            result = color_jitter(result)
        elif aug_name == 'zoom':
            result = zoom(result)
    return result
