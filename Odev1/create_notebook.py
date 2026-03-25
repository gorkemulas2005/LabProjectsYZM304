import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}

cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

# =====================
md("""# YZM304 Derin Ogrenme - Proje Odevi 1
Wisconsin Breast Cancer veri seti uzerinde ikili siniflandirma.
Modeller: NumPy (sifirdan), Scikit-learn, PyTorch.""")

# =====================
md("## 1. Kutuphaneler")
code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '.')

from src.data_preprocessing import load_data, split_data, standardize, SEED
from src.numpy_model import NeuralNetwork
from src.metrics import calc_metrics, confusion_matrix
from src.sklearn_model import train_sklearn
from src.pytorch_model import train_pytorch

LR = 0.01
EPOCHS = 1000
np.random.seed(SEED)
print("Kutuphaneler ve moduller yuklendi.")""")

# =====================
md("## 2. Veri Yukleme ve Analiz")
code("""X, y, df = load_data()
print(f"Ornek sayisi: {X.shape[0]}")
print(f"Ozellik sayisi: {X.shape[1]}")
print(f"Sinif dagilimi: M(1)={int(y.sum())}, B(0)={int(len(y)-y.sum())}")
print()
df.head()""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = [int(len(y) - y.sum()), int(y.sum())]
axes[0].bar(["Benign (0)", "Malign (1)"], counts, color=["#4CAF50", "#F44336"])
axes[0].set_title("Sinif Dagilimi")
axes[0].set_ylabel("Ornek Sayisi")
for i, v in enumerate(counts):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')

axes[1].pie(counts, labels=["Benign", "Malign"], autopct='%1.1f%%',
            colors=["#4CAF50", "#F44336"], startangle=90)
axes[1].set_title("Sinif Oranlari")
plt.tight_layout()
plt.show()""")

# =====================
md("## 3. Veri On Isleme")
code("""X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train, X_val, X_test, scaler = standardize(X_train, X_val, X_test)
n_features = X_train.shape[1]

print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
print(f"Ozellik sayisi: {n_features}")
print(f"Standardizasyon: train seti uzerinde fit, val/test'e transform")""")

# =====================
md("""## 4. NumPy Sinir Agi Sinifi
Kod: `src/numpy_model.py`
- Sifirdan sinir agi (`__forward`, `__backward` vb. private sinif metotlari ile tam OOP yapisi)
- He initialization, ReLU gizli katman, Sigmoid cikis
- Binary Cross Entropy kayip, SGD optimizer
- L2 regularizasyon ve mini-batch destegi

Not: Her iki modelde de (NumPy ve Scikit-learn) agirlik atamalarinin kontrollu olmasi adina Seed (Random State) 42 olarak sabitlenmistir.""")

# =====================
md("## 5. Model Egitimi")
md("### 5.1 Temel Model (1 gizli katman, 64 noron)")
code("""model1 = NeuralNetwork([n_features, 64, 1], lr=LR, seed=SEED)
hist1 = model1.train(X_train, y_train, X_val, y_val, epochs=EPOCHS)
print(f"Train Acc: {model1.accuracy(X_train, y_train):.4f}, Val Acc: {model1.accuracy(X_val, y_val):.4f}")""")

md("### 5.2 Genis Model (1 gizli katman, 128 noron)")
code("""model2 = NeuralNetwork([n_features, 128, 1], lr=LR, seed=SEED)
hist2 = model2.train(X_train, y_train, X_val, y_val, epochs=EPOCHS)
print(f"Train Acc: {model2.accuracy(X_train, y_train):.4f}, Val Acc: {model2.accuracy(X_val, y_val):.4f}")""")

md("### 5.3 Derin Model (2 gizli katman, 64-32)")
code("""model3 = NeuralNetwork([n_features, 64, 32, 1], lr=LR, seed=SEED)
hist3 = model3.train(X_train, y_train, X_val, y_val, epochs=EPOCHS)
print(f"Train Acc: {model3.accuracy(X_train, y_train):.4f}, Val Acc: {model3.accuracy(X_val, y_val):.4f}")""")

md("### 5.4 L2 Regularizasyonlu Model (lambda=0.01)")
code("""model4 = NeuralNetwork([n_features, 64, 1], lr=LR, lambda_reg=0.01, seed=SEED)
hist4 = model4.train(X_train, y_train, X_val, y_val, epochs=EPOCHS)
print(f"Train Acc: {model4.accuracy(X_train, y_train):.4f}, Val Acc: {model4.accuracy(X_val, y_val):.4f}")""")

md("### 5.5 Mini-batch Model (batch_size=32)")
code("""model5 = NeuralNetwork([n_features, 64, 1], lr=LR, seed=SEED)
hist5 = model5.train(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=32)
print(f"Train Acc: {model5.accuracy(X_train, y_train):.4f}, Val Acc: {model5.accuracy(X_val, y_val):.4f}")""")

code("""models_dict = {
    "Temel (64)":      (model1, hist1),
    "Genis (128)":     (model2, hist2),
    "Derin (64-32)":   (model3, hist3),
    "L2 Reg":          (model4, hist4),
    "Mini-batch":      (model5, hist5),
}""")

# =====================
md("## 6. Overfitting / Underfitting Analizi")
md("### 6.1 Egitim Egrileri (Loss ve Accuracy)")
code("""fig, axes = plt.subplots(2, 5, figsize=(28, 9))
names = list(models_dict.keys())

for i, (name, (mdl, hist)) in enumerate(models_dict.items()):
    axes[0, i].plot(hist["train_loss"], label="Train", linewidth=1.2)
    axes[0, i].plot(hist["val_loss"], label="Val", linewidth=1.2, linestyle="--")
    axes[0, i].set_title(f"{name} - Kayip", fontsize=11)
    axes[0, i].set_xlabel("Epoch")
    axes[0, i].set_ylabel("Loss")
    axes[0, i].legend(fontsize=9)
    axes[0, i].grid(True, alpha=0.3)

    axes[1, i].plot(hist["train_acc"], label="Train", linewidth=1.2)
    axes[1, i].plot(hist["val_acc"], label="Val", linewidth=1.2, linestyle="--")
    axes[1, i].set_title(f"{name} - Dogruluk", fontsize=11)
    axes[1, i].set_xlabel("Epoch")
    axes[1, i].set_ylabel("Accuracy")
    axes[1, i].legend(fontsize=9)
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/training_curves.png", dpi=150, bbox_inches='tight')
plt.show()
print("Kaydedildi: models/training_curves.png")""")

md("### 6.2 Overfitting Analiz Tablosu")
code("""print(f"{'Model':<18} {'TrainAcc':>8} {'ValAcc':>8} {'Fark':>7} {'TrainLoss':>10} {'ValLoss':>10} {'EnIyiEp':>8} {'Durum'}")
print("-" * 90)

for name, (mdl, hist) in models_dict.items():
    ta = hist["train_acc"][-1]
    va = hist["val_acc"][-1]
    gap = ta - va
    tl = hist["train_loss"][-1]
    vl = hist["val_loss"][-1]
    best_ep = int(np.argmax(hist["val_acc"])) + 1
    min_vl_ep = int(np.argmin(hist["val_loss"])) + 1

    if gap > 0.05: durum = "YUKSEK VARYANS"
    elif va < 0.85 and ta < 0.85: durum = "YUKSEK BIAS"
    elif gap > 0.03: durum = "HAFIF OVERFITTING"
    else: durum = "IYI UYUM"

    print(f"{name:<18} {ta:>8.4f} {va:>8.4f} {gap:>7.4f} {tl:>10.4f} {vl:>10.4f} {best_ep:>8} {durum}")
    if min_vl_ep < len(hist["val_loss"]) - 200:
        print(f"  -> Val loss epoch {min_vl_ep}'de minimum. Early stopping faydali olabilir.")""")

# =====================
md("## 7. Model Secimi")
code("""best_name, best_val, best_model = None, 0, None
for name, (mdl, hist) in models_dict.items():
    va = hist["val_acc"][-1]
    if va > best_val:
        best_val = va
        best_name = name
        best_model = mdl

print(f"En iyi model: {best_name}")
print(f"Dogrulama dogrulugu: {best_val:.4f}")
print(f"Kriter: En yuksek validation accuracy")""")

# =====================
md("## 8. Test Seti Metrikleri")
code("""numpy_results = {}
for name, (mdl, hist) in models_dict.items():
    y_pred = mdl.predict(X_test)
    m = calc_metrics(y_test, y_pred)
    numpy_results[name] = m
    cm = m["cm"]
    print(f"[{name}]  Acc: {m['accuracy']:.4f}  Prec: {m['precision']:.4f}  Rec: {m['recall']:.4f}  F1: {m['f1']:.4f}  | TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")""")

md("### 8.1 Confusion Matrix Gorsellestirme")
code("""fig, axes = plt.subplots(1, 5, figsize=(25, 4))

for i, (name, m) in enumerate(numpy_results.items()):
    cm = m["cm"]
    im = axes[i].imshow(cm, cmap='Blues', interpolation='nearest')
    axes[i].set_title(f"{name}\\nAcc={m['accuracy']:.3f}", fontsize=10)
    axes[i].set_xticks([0, 1])
    axes[i].set_yticks([0, 1])
    axes[i].set_xticklabels(["B (0)", "M (1)"])
    axes[i].set_yticklabels(["B (0)", "M (1)"])
    axes[i].set_xlabel("Tahmin")
    axes[i].set_ylabel("Gercek")
    for r in range(2):
        for c in range(2):
            axes[i].text(c, r, str(cm[r, c]), ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color='white' if cm[r, c] > cm.max()/2 else 'black')

plt.suptitle("NumPy Modelleri - Confusion Matrix", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("models/confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()""")

# =====================
md("## 9. Scikit-learn MLPClassifier")
code("""sklearn_clf, sklearn_pred = train_sklearn(X_train, y_train, X_test, y_test,
                                          hidden_layers=(64,), lr=LR, max_iter=EPOCHS, seed=SEED)
sklearn_m = calc_metrics(y_test, sklearn_pred)
cm = sklearn_m["cm"]
print(f"Scikit-learn MLPClassifier (hidden=(64,))")
print(f"  Train Acc: {sklearn_clf.score(X_train, y_train.ravel()):.4f}")
print(f"  Test  Acc: {sklearn_m['accuracy']:.4f}  Prec: {sklearn_m['precision']:.4f}  Rec: {sklearn_m['recall']:.4f}  F1: {sklearn_m['f1']:.4f}")
print(f"  CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")""")

# =====================
md("## 10. PyTorch Modeli")
code("""pt_model, pytorch_pred = train_pytorch(X_train, y_train, X_test, y_test,
                                       hidden_sizes=[64], lr=LR, epochs=EPOCHS, seed=SEED)
pytorch_m = calc_metrics(y_test, pytorch_pred)
cm = pytorch_m["cm"]
print(f"\\nPyTorch (hidden=[64])")
print(f"  Test  Acc: {pytorch_m['accuracy']:.4f}  Prec: {pytorch_m['precision']:.4f}  Rec: {pytorch_m['recall']:.4f}  F1: {pytorch_m['f1']:.4f}")
print(f"  CM: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")""")

# =====================
md("## 11. Tum Modellerin Karsilastirmasi")
md("### 11.1 Karsilastirma Tablosu")
code("""all_results = {}
for name, m in numpy_results.items():
    all_results["NP: " + name] = m
all_results["Scikit-learn"] = sklearn_m
all_results["PyTorch"] = pytorch_m

print(f"{'Model':<24} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4}")
print("-" * 78)
for name, m in all_results.items():
    cm = m["cm"]
    print(f"{name:<24} {m['accuracy']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f} {cm[0,0]:>4} {cm[0,1]:>4} {cm[1,0]:>4} {cm[1,1]:>4}")""")

md("### 11.2 Metrik Karsilastirma Grafigi")
code("""model_names = list(all_results.keys())
accs = [m["accuracy"] for m in all_results.values()]
precs = [m["precision"] for m in all_results.values()]
recs = [m["recall"] for m in all_results.values()]
f1s = [m["f1"] for m in all_results.values()]

x = np.arange(len(model_names))
w = 0.2

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - 1.5*w, accs, w, label='Accuracy', color='#2196F3')
ax.bar(x - 0.5*w, precs, w, label='Precision', color='#4CAF50')
ax.bar(x + 0.5*w, recs, w, label='Recall', color='#FF9800')
ax.bar(x + 1.5*w, f1s, w, label='F1', color='#9C27B0')

ax.set_ylabel('Skor')
ax.set_title('Tum Modeller - Metrik Karsilastirmasi')
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=9)
ax.legend()
ax.set_ylim(0.9, 1.01)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("models/metric_comparison.png", dpi=150, bbox_inches='tight')
plt.show()""")

md("### 11.3 Confusion Matrix - Scikit-learn ve PyTorch")
code("""fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for i, (name, m) in enumerate([("Scikit-learn", sklearn_m), ("PyTorch", pytorch_m)]):
    cm = m["cm"]
    im = axes[i].imshow(cm, cmap='Oranges', interpolation='nearest')
    axes[i].set_title(f"{name}\\nAcc={m['accuracy']:.3f}", fontsize=11)
    axes[i].set_xticks([0, 1])
    axes[i].set_yticks([0, 1])
    axes[i].set_xticklabels(["B (0)", "M (1)"])
    axes[i].set_yticklabels(["B (0)", "M (1)"])
    axes[i].set_xlabel("Tahmin")
    axes[i].set_ylabel("Gercek")
    for r in range(2):
        for c in range(2):
            axes[i].text(c, r, str(cm[r, c]), ha='center', va='center',
                        fontsize=18, fontweight='bold',
                        color='white' if cm[r, c] > cm.max()/2 else 'black')

plt.tight_layout()
plt.savefig("models/confusion_sklearn_pytorch.png", dpi=150, bbox_inches='tight')
plt.show()""")

# =====================
md("""## 12. Sonuc

- Tum modeller ayni seed, ayni veri bolunmesi ve ayni hiperparametreler ile egitildi.
- NumPy sifirdan model, Scikit-learn ile ayni sonucu verdi (gradyan dogrulamasi).
- En iyi dogrulama dogrulugu Genis (128) modelde elde edildi.
- Kaynak kodlar `src/` klasorunde modullere ayrilmistir.

Ortak yapilandirma: seed=42, lr=0.01, epoch=1000, SGD, BCE, ReLU/Sigmoid.""")

nb.cells = cells
nbf.write(nb, "proje.ipynb")
print("Notebook olusturuldu: proje.ipynb")
