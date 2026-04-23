"""
Model 3 -- Transfer Learning (VGG-16).

torchvision.models.vgg16 kullanilarak olusturulan transfer ogrenme modeli.
  - Pretrained=True: ImageNet uzerinde on-egitilmis agirliklar yuklenir.
  - Ozellik cikarici katmanlar (features) dondurulur (requires_grad=False).
  - Siniflandirici katmanin son katmani 10 sinif ciktisina uyarlanir.
  - CIFAR-10 goruntulerinin 224x224 boyutuna resize edilmesi gerekir.

VGG-16 secim gerekceleri:
  - Odev sartlarinda AlexNet veya VGG onerilmistir.
  - VGG-16, derin ama duzgun (uniform) mimarisi ile transfer learning
    uygulamalarinda referans model olarak yaygin kullanilir (Simonyan & Zisserman, 2014).
  - RTX 4050 GPU ile 224x224 boyutunda egitim suresi makuldur.
"""

import torch.nn as nn
import torchvision.models as models


def get_vgg16_transfer(num_classes: int = 10, freeze_features: bool = True):
    """Pretrained VGG-16 modelini CIFAR-10 icin yapilandirir.

    Args:
        num_classes: Cikti sinif sayisi.
        freeze_features: True ise ozellik cikarici katmanlar dondurulur.

    Returns:
        Yapilandirilmis VGG-16 modeli.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    # Ozellik cikarici katmanlari dondur
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Siniflandirici katmanin son katmanini degistir
    # Orijinal: Linear(4096, 1000) -> Yeni: Linear(4096, 10)
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model


def get_vgg16_feature_extractor(model):
    """Egitilmis VGG-16 modelinden feature extractor olusturur.

    Classifier'in son katmani oncesindeki 4096 boyutlu ozellik
    vektorunu cikarir.

    Args:
        model: Egitilmis VGG-16 modeli.

    Returns:
        Feature extractor fonksiyonu (nn.Sequential).
    """
    # features + avgpool + classifier[:-1]
    feature_extractor = nn.Sequential(
        model.features,
        model.avgpool,
        nn.Flatten(),
        *list(model.classifier.children())[:-1]   # Son Linear haric
    )
    return feature_extractor
