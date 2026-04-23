"""
Yardimci fonksiyonlar: Seed ayarlama, parametre sayisi hesabi.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Tekrarlanabilirlik icin tum rastgelelik kaynaklarini sabitler."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Modeldeki toplam ve egitilabilir parametre sayisini dondurur."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_model_summary(model, model_name="Model"):
    """Model parametre ozetini yazdirir."""
    total, trainable = count_parameters(model)
    frozen = total - trainable
    print(f"\n  {model_name} Parametre Ozeti:")
    print(f"    Toplam:       {total:>10,}")
    print(f"    Egitilabilir: {trainable:>10,}")
    print(f"    Dondurulan:   {frozen:>10,}")
