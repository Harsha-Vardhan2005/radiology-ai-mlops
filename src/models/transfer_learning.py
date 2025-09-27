import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ViT_B_16_Weights

# =====================================================
# Transfer Learning Models (ResNet50, ViT)
# =====================================================

def get_resnet50(num_classes: int, lr: float = 1e-4, freeze_backbone: bool = True, device: str = "cpu"):
    """
    Load pretrained ResNet50 and adapt for pneumonia classification.
    """
    resnet = models.resnet50(weights="IMAGENET1K_V1")  # pretrained weights
    if freeze_backbone:
        for param in resnet.parameters():
            param.requires_grad = False
    
    resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
    resnet = resnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(resnet.fc.parameters(), lr=lr)

    return resnet, criterion, optimizer


def get_vit(num_classes: int, lr: float = 1e-4, freeze_backbone: bool = True, device: str = "cpu"):
    """
    Load pretrained ViT-B/16 and adapt for pneumonia classification.
    """
    vit_model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    if freeze_backbone:
        for param in vit_model.parameters():
            param.requires_grad = False
    
    vit_model.heads.head = nn.Linear(vit_model.heads.head.in_features, num_classes)
    vit_model = vit_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vit_model.heads.head.parameters(), lr=lr)

    return vit_model, criterion, optimizer
