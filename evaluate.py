import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import SimpleCNN
from dataset import get_data_loaders

DEFAULT_MODEL_PATH = "models/model_all_augmentations.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['thumbs_up', 'thumbs_down', 'peace', 'open_palm', 'no_hand']


# loads a model and runs it on the test set
def evaluate_model(model_path=DEFAULT_MODEL_PATH):
    model = SimpleCNN(num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _, test_loader = get_data_loaders()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_preds, all_labels


# prints accuracy, f1 scores, and per-class metrics
def print_metrics(y_true, y_pred):
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)

    overall_acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {overall_acc:.4f}")

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print(f"F1-Score (weighted): {f1_weighted:.4f}")

    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        mask = y_true == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == i).sum() / mask.sum()
            print(f"  {class_name}: {class_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))


# generates and saves confusion matrix heatmap
def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"\nConfusion matrix saved to {save_path}")


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_PATH

    print(f"Evaluating model: {model_path}")

    y_pred, y_true = evaluate_model(model_path)
    print_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred)
