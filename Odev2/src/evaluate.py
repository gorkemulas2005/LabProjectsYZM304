"""
Degerlendirme ve gorsellestirme modulu.

Confusion matrix, classification report, per-class accuracy ve
egitim egrisi grafikleri uretir.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score
)


@torch.no_grad()
def get_predictions(model, loader, device):
    """Modelin tum test seti uzerindeki tahminlerini toplar.

    Args:
        model: PyTorch modeli.
        loader: Test DataLoader'i.
        device: torch.device.

    Returns:
        (all_preds, all_labels) numpy dizileri.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, class_names, title="Karmasiklik Matrisi",
                          save_path=None):
    """Confusion matrix isi haritasini cizer.

    Args:
        y_true: Gercek etiketler.
        y_pred: Tahmin edilen etiketler.
        class_names: Sinif isimleri listesi.
        title: Grafik basligi.
        save_path: Kaydedilecek dosya yolu (None ise kaydedilmez).

    Returns:
        matplotlib Figure nesnesi.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Tahmin", fontsize=12)
    ax.set_ylabel("Gercek", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_training_curves(history, model_name="Model", save_path=None):
    """Egitim ve test loss/accuracy grafiklerini cizer.

    Args:
        history: train_model fonksiyonundan donen dict.
        model_name: Grafik basliginda kullanilacak model adi.
        save_path: Kaydedilecek dosya yolu.

    Returns:
        matplotlib Figure nesnesi.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss grafigi
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=3, label="Egitim Loss")
    ax1.plot(epochs, history["test_loss"], "r-s", markersize=3, label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (CrossEntropy)")
    ax1.set_title(f"{model_name} -- Kayip Egrisi", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy grafigi
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=3, label="Egitim Accuracy")
    ax2.plot(epochs, history["test_acc"], "r-s", markersize=3, label="Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"{model_name} -- Dogruluk Egrisi", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_per_class_accuracy(y_true, y_pred, class_names,
                            title="Sinif Bazli Dogruluk", save_path=None):
    """Her sinif icin ayri accuracy cubuk grafigi cizer.

    Args:
        y_true: Gercek etiketler.
        y_pred: Tahmin edilen etiketler.
        class_names: Sinif isimleri listesi.
        title: Grafik basligi.
        save_path: Kaydedilecek dosya yolu.

    Returns:
        matplotlib Figure nesnesi.
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(class_names)), per_class_acc * 100,
                  color=plt.cm.tab10(range(len(class_names))), edgecolor="black")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)

    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc * 100:.1f}%", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(results_dict, metric="accuracy",
                          title="Model Karsilastirmasi", save_path=None):
    """Birden fazla modelin performans metrigini cubuk grafiginde kiyaslar.

    Args:
        results_dict: {model_adi: deger} sozlugu.
        metric: Gosterilecek metrik adi (grafik etiketi icin).
        title: Grafik basligi.
        save_path: Kaydedilecek dosya yolu.

    Returns:
        matplotlib Figure nesnesi.
    """
    names = list(results_dict.keys())
    values = list(results_dict.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, values, color=colors, edgecolor="black", width=0.6)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_ylabel(f"{metric} (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(values) + 10)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_comparison_training_curves(histories_dict, save_path=None):
    """Birden fazla modelin egitim egrilerini ayni grafik uzerinde gosterir.

    Args:
        histories_dict: {model_adi: history_dict} sozlugu.
        save_path: Kaydedilecek dosya yolu.

    Returns:
        matplotlib Figure nesnesi.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(histories_dict)))

    for (name, hist), color in zip(histories_dict.items(), colors):
        epochs = range(1, len(hist["test_loss"]) + 1)
        ax1.plot(epochs, hist["test_loss"], "-", color=color, label=name, linewidth=2)
        ax2.plot(epochs, hist["test_acc"], "-", color=color, label=name, linewidth=2)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Loss")
    ax1.set_title("Test Kayip Egrileri Karsilastirmasi", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy (%)")
    ax2.set_title("Test Dogruluk Egrileri Karsilastirmasi", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def print_classification_report(y_true, y_pred, class_names, model_name="Model"):
    """Siniflandirma raporunu formatli sekilde yazdirir.

    Args:
        y_true: Gercek etiketler.
        y_pred: Tahmin edilen etiketler.
        class_names: Sinif isimleri listesi.
        model_name: Model adi.
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   zero_division=0)
    print(f"\n{'='*60}")
    print(f"  {model_name} -- Siniflandirma Raporu")
    print(f"  Genel Dogruluk: {acc * 100:.2f}%")
    print(f"{'='*60}")
    print(report)
    return acc, report
