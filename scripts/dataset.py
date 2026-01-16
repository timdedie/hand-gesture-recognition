import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from augmentations import apply_augmentations
import random

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_PATH = os.path.join(SCRIPT_DIR, "..", "dataset")

class GestureDataset(Dataset):
    def __init__(self, images, labels, augment_list=None, augment_prob=0.5):
        self.images = images
        self.labels = labels
        self.augment_list = augment_list
        self.augment_prob = augment_prob

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].copy()
        label = self.labels[idx]

        if self.augment_list and random.random() < self.augment_prob:
            image = apply_augmentations(image, self.augment_list)

        image = cv2.resize(image, (64, 64))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        return torch.tensor(image), torch.tensor(label)


def load_dataset(dataset_path=None, max_per_class=None):
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH
    classes = ['thumbs_up', 'thumbs_down', 'peace', 'open_palm', 'no_hand']
    images = []
    labels = []

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            continue

        files = os.listdir(class_path)
        if max_per_class:
            files = files[:max_per_class]

        for filename in files:
            filepath = os.path.join(class_path, filename)
            image = cv2.imread(filepath)
            if image is not None:
                images.append(image)
                labels.append(class_idx)

    return images, np.array(labels)


def get_data_loaders(dataset_path=None, batch_size=32, augment_list=None, max_per_class=None):
    if dataset_path is None:
        dataset_path = DEFAULT_DATASET_PATH
    images, labels = load_dataset(dataset_path, max_per_class)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = GestureDataset(X_train, y_train, augment_list)
    test_dataset = GestureDataset(X_test, y_test, augment_list=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
