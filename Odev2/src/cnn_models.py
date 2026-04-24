import torch.nn as nn
from torchvision import models

class BasicCNN(nn.Module):
    """
    LeNet-5 tabanli, sifirdan yazilmis temel CNN mimarisi.
    """
    def __init__(self, num_classes=10):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ImprovedCNN(nn.Module):
    """
    Model 1 uzerine BatchNorm ve Dropout eklenmis iyilestirilmis CNN.
    """
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_vgg16_transfer(num_classes=10):
    """
    Transfer learning icin on egitimli VGG-16 modelini hazirlar.
    Ozellik cikarici katmanlar dondurulur.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    
    # Mevcut katmanlari dondur
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Sadece son siniflandiriciyi degistir
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)
    
    return model
