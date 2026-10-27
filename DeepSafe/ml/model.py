
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn


def build_model(num_classes=2):
    # Load pretrained ResNet18.
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze the feature extractor.
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer.
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Make the new classifier trainable.
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

