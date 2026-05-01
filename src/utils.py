import os
import torch
import matplotlib.pyplot as plt


def get_device():
    """
    GPU varsa CUDA, yoksa CPU döndürür.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dir(path):
    """
    Klasör yoksa oluşturur.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_training_plot(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    Eğitim ve doğrulama loss/accuracy grafiklerini kaydeder.
    """

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Training Accuracy")
    plt.plot(epochs, val_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()