"""
Genel egitim ve degerlendirme dongusu.

Tum modeller (Model 1-5) icin ortak egitim altyapisi saglar.
  - GPU/CPU otomatik tespiti
  - Epoch bazli loss ve accuracy kaydi (train + test)
  - Egitim suresi olcumu
  - Model agirliklarini diske kaydetme

Hiperparametre tercihleri:
  - Adam optimizer: Adaptif ogrenme orani + momentum birlesimi;
    CNN egitiminde SGD'ye kiyasla daha hizli yakinsamasi kanitlanmistir
    (Kingma & Ba, 2014).
  - CrossEntropyLoss: Cok sinifli siniflandirma icin standart kayip
    fonksiyonu; softmax + NLL bilesimini tek adimda hesaplar.
  - Learning rate = 0.001: Adam icin onerilen varsayilan deger.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam


def get_device():
    """Mevcut cihazi (GPU veya CPU) tespit eder ve dondurur."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"Cihaz: {torch.cuda.get_device_name(0)} (CUDA)")
    else:
        print("Cihaz: CPU")
    return device


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Tek bir epoch egitim dongusu.

    Args:
        model: Egitilecek PyTorch modeli.
        loader: Egitim DataLoader'i.
        criterion: Kayip fonksiyonu.
        optimizer: Optimizer nesnesi.
        device: torch.device.

    Returns:
        (ortalama_loss, accuracy) tuple.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Model performansini test/validation seti uzerinde olcer.

    Args:
        model: Degerlendirilecek PyTorch modeli.
        loader: Test/validation DataLoader'i.
        criterion: Kayip fonksiyonu.
        device: torch.device.

    Returns:
        (ortalama_loss, accuracy) tuple.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, epochs: int = 20,
                lr: float = 0.001, device=None, model_name: str = "model",
                save_dir: str = None):
    """Tam egitim pipeline'i: egitim + degerlendirme + kayit.

    Args:
        model: PyTorch modeli.
        train_loader: Egitim DataLoader'i.
        test_loader: Test DataLoader'i.
        epochs: Toplam epoch sayisi.
        lr: Ogrenme orani.
        device: torch.device. None ise otomatik tespit edilir.
        model_name: Kayit ve log icin model adi.
        save_dir: Model agirliklarinin kaydedilecegi dizin.

    Returns:
        dict: Egitim gecmisi (train_loss, train_acc, test_loss, test_acc listleri
              ve toplam egitim suresi).
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [],
        "test_loss": [], "test_acc": [],
    }

    print(f"\n{'='*60}")
    print(f"  {model_name} -- Egitim Basliyor")
    print(f"  Epochs: {epochs} | LR: {lr} | Optimizer: Adam")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"  Epoch [{epoch:3d}/{epochs}]  "
              f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%  |  "
              f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.2f}%")

    elapsed = time.time() - start_time
    history["elapsed_time"] = elapsed

    print(f"\n  Egitim suresi: {elapsed:.1f} saniye")
    print(f"  En iyi test accuracy: {max(history['test_acc']):.2f}% "
          f"(Epoch {history['test_acc'].index(max(history['test_acc'])) + 1})")

    # Model agirliklarini kaydet
    if save_dir is None:
        save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models"
        )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"  Model kaydedildi: {save_path}")

    return history
