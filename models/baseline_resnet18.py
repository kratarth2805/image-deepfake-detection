import torch
import torch.nn as nn
from torchvision import models

def build_resnet18_baseline(num_classes=2, freeze_features=True):
    # ImageNet pretrained weights load karna
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # Feature extraction layers ko freeze karna
    if freeze_features:
        for param in model.parameters():
            param.requires_grad = False

    # Aakhiri classification layer ko 2 classes (REAL / FAKE) ke liye replace karna
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

if __name__ == "__main__":
    # Architecture verify karne ke liye dummy tensor check
    model = build_resnet18_baseline()
    dummy_x = torch.randn(2, 3, 224, 224)
    out = model(dummy_x)
    print("Baseline model output shape:", out.shape)  # Expected: [2, 2]