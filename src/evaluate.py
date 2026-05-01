import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from model import get_resnet18_model
from utils import get_device, create_dir


# =========================
# Configuration
# =========================

TEST_DIR = "dataset/test"
MODEL_PATH = "models/best_model.pth"
RESULTS_DIR = "results/figures"
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")

BATCH_SIZE = 32
NUM_CLASSES = 2


# =========================
# Data Transform
# =========================

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def main():
    create_dir(RESULTS_DIR)

    device = get_device()
    print(f"Using device: {device}")

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=test_transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Class mapping:")
    print(test_dataset.class_to_idx)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = get_resnet18_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_features=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = (all_labels == all_predictions).mean()

    print("\nTest Accuracy:")
    print(f"{accuracy:.4f}")

    class_names = list(test_dataset.class_to_idx.keys())

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=class_names
        )
    )

    cm = confusion_matrix(all_labels, all_predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - Cat vs Dog Classification")
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()

    print(f"\nConfusion matrix saved to: {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()