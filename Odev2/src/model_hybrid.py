"""
Model 4 ve Model 5 -- Hibrit yaklasim.

Model 4:
  Bir CNN modeli (Model 2) feature extractor olarak kullanilir.
  Cikarilan ozellikler ve etiketler .npy dosyalarina kaydedilir.
  Bu ozellikler SVM ve Random Forest ile siniflandirilir.

Model 5:
  Ayni veri seti uzerinde tam CNN modeli egitilip hibrit sonuclarla kiyaslanir.
  (Kisayol: Model 1 veya Model 2 ayni veri setinde egitildiyse tekrar kullanilabilir.)
"""

import os
import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


@torch.no_grad()
def extract_features(model, loader, device):
    """CNN modelinden ozellik vektorlerini cikarir.

    Args:
        model: get_features() metodu olan bir CNN modeli.
        loader: DataLoader nesnesi.
        device: torch.device.

    Returns:
        (features, labels) numpy dizileri.
    """
    model.eval()
    all_features = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        features = model.get_features(images)
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


def save_features(features, labels, save_dir, prefix=""):
    """Ozellik ve etiket dizilerini .npy dosyalarina kaydeder.

    Dosya boyutu ve uzunlugu ekrana yazdirilir (odev gereksinimi).

    Args:
        features: Ozellik matrisi (N, D).
        labels: Etiket vektoru (N,).
        save_dir: Kayit dizini.
        prefix: Dosya adi oneki ("train" veya "test").
    """
    os.makedirs(save_dir, exist_ok=True)

    feat_path = os.path.join(save_dir, f"{prefix}_features.npy")
    label_path = os.path.join(save_dir, f"{prefix}_labels.npy")

    np.save(feat_path, features)
    np.save(label_path, labels)

    feat_size = os.path.getsize(feat_path) / (1024 * 1024)
    label_size = os.path.getsize(label_path) / (1024 * 1024)

    print(f"  {prefix.upper()} Ozellikler: shape={features.shape}, "
          f"dosya boyutu={feat_size:.2f} MB")
    print(f"  {prefix.upper()} Etiketler:  shape={labels.shape}, "
          f"dosya boyutu={label_size:.4f} MB")
    print(f"  Toplam ornek sayisi: {len(labels)}")


def load_features(save_dir, prefix=""):
    """Kaydedilmis .npy dosyalarindan ozellikleri ve etiketleri yukler.

    Args:
        save_dir: Kayit dizini.
        prefix: Dosya adi oneki.

    Returns:
        (features, labels) numpy dizileri.
    """
    feat_path = os.path.join(save_dir, f"{prefix}_features.npy")
    label_path = os.path.join(save_dir, f"{prefix}_labels.npy")

    features = np.load(feat_path)
    labels = np.load(label_path)

    print(f"  {prefix.upper()} Ozellikler yuklendi: shape={features.shape}")
    print(f"  {prefix.upper()} Etiketler yuklendi:  shape={labels.shape}")

    return features, labels


def train_svm(train_features, train_labels, test_features, test_labels,
              class_names=None):
    """SVM siniflandirici egitir ve degerlendirir.

    Kernel: RBF (varsayilan). C: 10.
    C=10 secimi: Varsayilan C=1'e kiyasla daha siki sinir olusturur;
    ozellik uzayindaki CNN temsillerinin ayirt edici gucunu daha iyi kullanir.

    Args:
        train_features, train_labels: Egitim verisi.
        test_features, test_labels: Test verisi.
        class_names: Sinif isimleri listesi.

    Returns:
        (accuracy, classification_report_str, predictions) tuple.
    """
    print("\n  SVM egitiliyor (kernel=rbf, C=10)...")
    svm = SVC(kernel="rbf", C=10, random_state=42)
    svm.fit(train_features, train_labels)

    predictions = svm.predict(test_features)
    acc = accuracy_score(test_labels, predictions)
    report = classification_report(
        test_labels, predictions, target_names=class_names, zero_division=0
    )

    print(f"  SVM Test Accuracy: {acc * 100:.2f}%")
    return acc, report, predictions


def train_random_forest(train_features, train_labels, test_features,
                        test_labels, class_names=None):
    """Random Forest siniflandirici egitir ve degerlendirir.

    n_estimators=200: Yeterli agac sayisi ile varyans dusurulur.

    Args:
        train_features, train_labels: Egitim verisi.
        test_features, test_labels: Test verisi.
        class_names: Sinif isimleri listesi.

    Returns:
        (accuracy, classification_report_str, predictions) tuple.
    """
    print("\n  Random Forest egitiliyor (n_estimators=200)...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)
    acc = accuracy_score(test_labels, predictions)
    report = classification_report(
        test_labels, predictions, target_names=class_names, zero_division=0
    )

    print(f"  Random Forest Test Accuracy: {acc * 100:.2f}%")
    return acc, report, predictions
