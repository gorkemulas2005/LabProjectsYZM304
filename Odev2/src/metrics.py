import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def plot_training_curves(history, title="Egitim ve Test Egrileri"):
    """
    Modelin egitim ve validation kayip/dogruluk egrilerini cizer.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history and history['val_loss']:
        ax1.plot(epochs, history['val_loss'], 'r-', label='Test Loss')
    ax1.set_title(f'Kayip (Loss) - {title}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    if 'val_acc' in history and history['val_acc']:
        ax2.plot(epochs, history['val_acc'], 'r-', label='Test Accuracy')
    ax2.set_title(f'Dogruluk (Accuracy) - {title}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_and_plot_cm(y_true, y_pred, classes, title="Confusion Matrix"):
    """
    Classification report yazdirir ve Confusion Matrix cizer.
    """
    print(f"\n{title} - Siniflandirma Raporu:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Gercek Sinif')
    plt.xlabel('Tahmin Edilen Sinif')
    plt.tight_layout()
    plt.show()
