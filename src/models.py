import torch.nn as nn
from torchvision import models

def _replace_classifier(model, in_features, num_classes):
    return nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

def build_model(name: str, num_classes: int, pretrained: bool = True):
    name = name.lower()
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m
    if name == "inception_v3":
        m = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT if pretrained else None, aux_logits=True)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        if m.aux_logits:
            m.AuxLogits.fc = nn.Linear(m.AuxLogits.fc.in_features, num_classes)
        return m
    if name == "vgg16":
        m = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        in_features = m.classifier[6].in_features
        m.classifier[6] = nn.Linear(in_features, num_classes)
        return m
    raise ValueError("Unsupported model")
