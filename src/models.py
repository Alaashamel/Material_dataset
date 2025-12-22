import torch.nn as nn
from torchvision import models


def build_model(name: str, num_classes: int, pretrained: bool = True):
    name = name.lower()

    if name == "resnet50":
        model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
        return model

    if name == "vgg16":
        model = models.vgg16(
            weights=models.VGG16_Weights.DEFAULT if pretrained else None
        )
        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features, num_classes
        )
        return model

    if name == "inception_v3":
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT if pretrained else None,
            aux_logits=True
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model.AuxLogits.fc = nn.Linear(
            model.AuxLogits.fc.in_features, num_classes
        )
        return model

    raise ValueError(f"Unsupported model: {name}")
