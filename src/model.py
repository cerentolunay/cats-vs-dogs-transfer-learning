import torch.nn as nn
from torchvision import models


def get_resnet18_model(num_classes=2, pretrained=True, freeze_features=True):
    """
    ResNet18 tabanlı transfer learning modeli oluşturur.

    Args:
        num_classes (int): Çıkış sınıf sayısı. Cat/Dog için 2.
        pretrained (bool): ImageNet üzerinde eğitilmiş ağırlıkları kullanır.
        freeze_features (bool): Feature extraction katmanlarını dondurur.

    Returns:
        model: Düzenlenmiş ResNet18 modeli.
    """

    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet18(weights=weights)

    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )

    return model