# src/models/baseline_cnn.py

import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    """
    Baseline CNN for Chest X-ray Classification (Normal vs Pneumonia)
    """

    def __init__(self, num_classes=2, img_size=224):
        super(BaselineCNN, self).__init__()
        # Feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (img_size // 8) * (img_size // 8), 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_baseline_cnn(num_classes=2, img_size=224, device='cpu'):
    """
    Helper function to create and move model to device.
    """
    model = BaselineCNN(num_classes=num_classes, img_size=img_size)
    return model.to(device)
