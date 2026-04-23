"""
Model 2 -- Iyilestirilmis CNN (BatchNorm + Dropout).

Model 1 ile ayni konvolusyon ve FC katmanlari, ayni hiperparametreler.
Farklar:
  - Her Conv2d sonrasina BatchNorm2d eklenmistir.
    BatchNorm, ic kovaryans kaymasini (internal covariate shift) azaltarak
    egitimi hizlandirir ve regularizasyon etkisi saglar (Ioffe & Szegedy, 2015).
  - FC katmanlari arasina Dropout(p=0.5) eklenmistir.
    Dropout, rastgele noron devre disi birakarak asiri ogrenmeyi (overfitting)
    engeller (Srivastava et al., 2014).
"""

import torch.nn as nn


class ImprovedCNN(nn.Module):
    """BatchNorm ve Dropout eklemeli iyilestirilmis CNN sinifi."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_features(self, x):
        """Son FC katmani oncesindeki ozellik vektorunu dondurur.

        Cikti boyutu: (batch_size, 84).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier[0](x)   # Linear(400, 120)
        x = self.classifier[1](x)   # ReLU
        # Dropout atlanir (inference modunda zaten devre disi)
        x = self.classifier[3](x)   # Linear(120, 84)
        x = self.classifier[4](x)   # ReLU
        return x
