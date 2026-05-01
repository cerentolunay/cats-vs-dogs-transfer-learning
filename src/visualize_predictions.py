import os
import random
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import datasets, transforms

from model import get_resnet18_model
from utils import get_device, create_dir


# =========================
# Configuration
# =========================

TEST_DIR = "dataset/test"
MODEL_PATH = "models/best_model.pth"
OUTPUT_DIR = "results/predictions"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "sample_predictions.png")

NUM_CLASSES = 2
NUM_SAMPLES = 8
RANDOM_SEED = 42


# =========================
# Image Transform
# =========================

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(device):
    """
    Eğitilmiş ResNet18 modelini yükler.
    """

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = get_resnet18_model(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_features=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {value: key for key, value in class_to_idx.items()}

    return model, idx_to_class


def predict_single_image(model, image_path, device):
    """
    Tek bir görüntü için tahmin ve güven skorunu döndürür.
    """

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return predicted_idx.item(), confidence.item() * 100


def main():
    random.seed(RANDOM_SEED)

    create_dir(OUTPUT_DIR)

    device = get_device()
    print(f"Using device: {device}")

    dataset = datasets.ImageFolder(root=TEST_DIR)

    model, idx_to_class = load_model(device)

    sample_indices = random.sample(range(len(dataset.samples)), NUM_SAMPLES)

    plt.figure(figsize=(16, 8))

    for i, sample_idx in enumerate(sample_indices):
        image_path, true_label_idx = dataset.samples[sample_idx]

        predicted_idx, confidence = predict_single_image(
            model,
            image_path,
            device
        )

        true_class = idx_to_class[true_label_idx]
        predicted_class = idx_to_class[predicted_idx]

        image = Image.open(image_path).convert("RGB")

        is_correct = true_class == predicted_class
        result_text = "Correct" if is_correct else "Wrong"

        plt.subplot(2, 4, i + 1)
        plt.imshow(image)
        plt.axis("off")
        plt.title(
            f"True: {true_class}\n"
            f"Pred: {predicted_class}\n"
            f"Conf: {confidence:.2f}% | {result_text}",
            fontsize=10
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()

    print(f"Sample predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()