"""
CIFAR-10 veri seti yukleme, on isleme ve DataLoader olusturma modulu.

On isleme adimlari:
  - ToTensor ile [0,1] araligina donusum
  - CIFAR-10 kanal ortalama ve standart sapmalari ile normalizasyon
  - Egitim seti icin RandomHorizontalFlip ve RandomCrop (veri artirma)
  - VGG-16 icin ayri pipeline: Resize(224) eklenir
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# CIFAR-10 kanal istatistikleri (literaturden)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def get_transforms(train: bool = True, resize_224: bool = False):
    """Egitim veya test icin transform pipeline dondurur.

    Args:
        train: True ise veri artirma (augmentation) uygulanir.
        resize_224: True ise goruntuleri 224x224 boyutuna buyutur (VGG-16 icin).

    Returns:
        torchvision.transforms.Compose nesnesi.
    """
    transform_list = []

    if resize_224:
        transform_list.append(transforms.Resize(224))

    if train:
        pad = 4 if not resize_224 else 16
        crop_size = 32 if not resize_224 else 224
        transform_list.extend([
            transforms.RandomCrop(crop_size, padding=pad),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return transforms.Compose(transform_list)


def get_cifar10_loaders(batch_size: int = 64, data_dir: str = None,
                        resize_224: bool = False, num_workers: int = 2):
    """CIFAR-10 train ve test DataLoader nesnelerini dondurur.

    Veri seti yoksa otomatik indirilir.

    Args:
        batch_size: Mini-batch boyutu.
        data_dir: Veri setinin kaydedilecegi dizin. None ise Odev2/data kullanilir.
        resize_224: True ise 224x224 boyutuna resize yapilir (VGG-16 icin).
        num_workers: Veri yukleme icin paralel isci sayisi.

    Returns:
        (train_loader, test_loader) tuple.
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    train_transform = get_transforms(train=True, resize_224=resize_224)
    test_transform = get_transforms(train=False, resize_224=resize_224)

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin
    )

    return train_loader, test_loader


def show_sample_images(loader, n_images: int = 16, title: str = "Ornek Goruntular"):
    """DataLoader'dan alinmis bir batch'ten ornek goruntuleri gorsellestirir.

    Normalizasyonu geri alarak orijinal renkleri gosterir.

    Args:
        loader: DataLoader nesnesi.
        n_images: Gosterilecek goruntu sayisi.
        title: Grafik basligi.
    """
    images, labels = next(iter(loader))
    images = images[:n_images]
    labels = labels[:n_images]

    # Normalizasyonu geri al
    mean = np.array(CIFAR10_MEAN).reshape(1, 3, 1, 1)
    std = np.array(CIFAR10_STD).reshape(1, 3, 1, 1)
    images_np = images.numpy() * std + mean
    images_np = np.clip(images_np, 0, 1)

    cols = 4
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flat):
        if i < n_images:
            ax.imshow(images_np[i].transpose(1, 2, 0))
            ax.set_title(CIFAR10_CLASSES[labels[i]], fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    return fig


def show_class_distribution(loader, dataset_name: str = "Egitim"):
    """Veri setindeki sinif dagilimini cubuk grafigi ile gosterir.

    Args:
        loader: DataLoader nesnesi.
        dataset_name: Grafik basliginda kullanilacak veri seti adi.

    Returns:
        matplotlib Figure nesnesi.
    """
    all_labels = []
    for _, labels in loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)

    counts = np.bincount(all_labels, minlength=10)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(10), counts, color=plt.cm.tab10(range(10)), edgecolor="black")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_ylabel("Ornek Sayisi")
    ax.set_title(f"{dataset_name} Seti -- Sinif Dagilimi", fontsize=13, fontweight="bold")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig
