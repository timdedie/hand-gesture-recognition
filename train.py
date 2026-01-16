import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN
from dataset import get_data_loaders
import json
import os

MODELS_DIR = "models"
RESULTS_PATH = "results.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(train_loader, test_loader, epochs=20):
    model = SimpleCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total
        train_loss = train_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = correct / total
        test_loss = test_loss / len(test_loader)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    return model, history


def run_augmentation_experiments():
    augment_configs = [
        ('no_augmentation', None),
        ('flip_only', ['flip']),
        ('rotate_only', ['rotate']),
        ('brightness_only', ['brightness']),
        ('blur_only', ['blur']),
        ('flip_rotate', ['flip', 'rotate']),
        ('all_augmentations', ['flip', 'rotate', 'brightness', 'blur']),
    ]

    results = {}

    for name, aug_list in augment_configs:
        print(f"\n{'='*50}")
        print(f"Experiment: {name}")
        print('='*50)

        train_loader, test_loader = get_data_loaders(augment_list=aug_list)
        model, history = train_model(train_loader, test_loader)

        results[name] = {
            'final_train_acc': history['train_acc'][-1],
            'final_test_acc': history['test_acc'][-1],
            'history': history
        }

        torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"model_{name}.pth"))

    return results


def run_dataset_size_experiments():
    sizes = [50, 100, 200, None]
    results = {}

    for with_aug in [False, True]:
        aug_list = ['flip', 'rotate', 'brightness'] if with_aug else None
        aug_name = "with_aug" if with_aug else "no_aug"

        for size in sizes:
            size_name = str(size) if size else "full"
            exp_name = f"{size_name}_{aug_name}"

            print(f"\n{'='*50}")
            print(f"Experiment: {exp_name}")
            print('='*50)

            train_loader, test_loader = get_data_loaders(
                augment_list=aug_list,
                max_per_class=size
            )
            model, history = train_model(train_loader, test_loader)

            results[exp_name] = {
                'final_train_acc': history['train_acc'][-1],
                'final_test_acc': history['test_acc'][-1]
            }

    return results


if __name__ == "__main__":
    print("Running augmentation experiments...")
    aug_results = run_augmentation_experiments()

    print("\n\nRunning dataset size experiments...")
    size_results = run_dataset_size_experiments()

    all_results = {
        'augmentation_experiments': aug_results,
        'dataset_size_experiments': size_results
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: x if not isinstance(x, list) else x)

    print("\n\nAll experiments completed!")
    print("\nAugmentation Results:")
    for name, res in aug_results.items():
        print(f"  {name}: Test Acc = {res['final_test_acc']:.4f}")

    print("\nDataset Size Results:")
    for name, res in size_results.items():
        print(f"  {name}: Test Acc = {res['final_test_acc']:.4f}")
