"""
Model 1 -- Temel CNN (LeNet-5 Benzeri).

CIFAR-10 (3x32x32) girdi boyutuna uyarlanmis LeNet-5 mimarisi.
Katman yapisi:
  Conv2d(3,6,5) -> ReLU -> MaxPool2d(2,2)
  Conv2d(6,16,5) -> ReLU -> MaxPool2d(2,2)
  Flatten
  Linear(16*5*5, 120) -> ReLU
  Linear(120, 84) -> ReLU
  Linear(84, 10)

Secim gerekceleri:
  - LeNet-5, CNN literaturunun temel referans mimarisidir (LeCun et al., 1998).
  - Sade yapisi sayesinde Model 2 ile kontrollü kiyaslama saglar.
  - CIFAR-10 32x32 boyutu, iki adet 5x5 conv + 2x2 pool sonrasi 5x5 feature map uretir;
    bu da LeNet-5 boyut hesaplamasiyla birebir uyumludur.
"""

import torch.nn as nn


class BasicCNN(nn.Module):
    """LeNet-5 benzeri temel CNN sinifi.

    Mimari parametreleri sabittir ve Model 2 ile birebir ayni tutulmustur
    (BatchNorm/Dropout haric).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),    # 32->28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # 28->14

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),   # 14->10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # 10->5
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # Flatten
        x = self.classifier(x)
        return x

    def get_features(self, x):
        """Son FC katmanina girmeden onceki ozellik vektorunu dondurur.

        Model 4 (Hibrit) icin feature extractor olarak kullanilir.
        Cikti boyutu: (batch_size, 84).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier[0](x)   # Linear(400, 120)
        x = self.classifier[1](x)   # ReLU
        x = self.classifier[2](x)   # Linear(120, 84)
        x = self.classifier[3](x)   # ReLU
        return x
