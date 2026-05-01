import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_resnet18_model
from utils import get_device, create_dir, save_training_plot


# =========================
# Configuration
# =========================

DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_SAVE_DIR = "models"
RESULTS_DIR = "results/figures"

MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "best_model.pth")
PLOT_SAVE_PATH = os.path.join(RESULTS_DIR, "accuracy_loss_curve.png")

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 2


# =========================
# Data Transforms
# =========================

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# Training Function
# =========================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    loop = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)

        running_loss += loss.item() * images.size(0)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


# =========================
# Validation Function
# =========================

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            running_loss += loss.item() * images.size(0)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


# =========================
# Main
# =========================

def main():
    create_dir(MODEL_SAVE_DIR)
    create_dir(RESULTS_DIR)

    device = get_device()
    print(f"Using device: {device}")

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transforms
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print("Class mapping:")
    print(train_dataset.class_to_idx)

    model = get_resnet18_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_features=True
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Accuracy  : {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "val_accuracy": best_val_accuracy
            }, MODEL_SAVE_PATH)

            print(f"Best model saved with validation accuracy: {best_val_accuracy:.4f}")

    save_training_plot(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        PLOT_SAVE_PATH
    )

    print("\nTraining completed.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    print(f"Training plot saved to: {PLOT_SAVE_PATH}")


if __name__ == "__main__":
    main()