import argparse
import torch
from PIL import Image
from torchvision import transforms

from model import get_resnet18_model
from utils import get_device


# =========================
# Configuration
# =========================

MODEL_PATH = "models/best_model.pth"
NUM_CLASSES = 2


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
    Eğitilmiş modeli yükler.
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


def predict_image(image_path):
    """
    Tek bir görüntü için sınıf tahmini yapar.
    """

    device = get_device()
    print(f"Using device: {device}")

    model, idx_to_class = load_model(device)

    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = idx_to_class[predicted_idx.item()]
    confidence_score = confidence.item() * 100

    print("\nPrediction Result")
    print("-----------------")
    print(f"Image Path      : {image_path}")
    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence_score:.2f}%")

    return predicted_class, confidence_score


def main():
    parser = argparse.ArgumentParser(
        description="Cat vs Dog image prediction using trained ResNet18 model."
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image."
    )

    args = parser.parse_args()

    predict_image(args.image)


if __name__ == "__main__":
    main()