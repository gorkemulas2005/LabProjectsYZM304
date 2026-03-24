# YZM304 Derin Ogrenme - Proje Odevi 1

Wisconsin Breast Cancer veri seti uzerinde ikili siniflandirma yapan sinir agi modelleri.
Modeller NumPy (sifirdan), Scikit-learn MLPClassifier ve PyTorch ile uygulanmistir.

## Giris

Bu projede, meme kanseri teshisi icin ikili siniflandirma (Malign / Benign) problemi ele alinmistir.
Veri seti 569 ornekten ve 30 sayisal ozellikten olusmaktadir.
Amac, farkli sinir agi yapilarini karsilastirmak ve en iyi modeli belirlemektir.

Calisma kapsaminda:
- NumPy ile sifirdan sinir agi uygulanmistir (kutuphane kullanilmadan).
- Overfitting/underfitting analizi yapilmistir.
- Farkli mimari, regularizasyon ve mini-batch ile model iyilestirmeleri denenmistir.
- Ayni mimari Scikit-learn MLPClassifier ve PyTorch ile tekrar yazilmistir.
- Tum modeller ayni hiperparametreler ve veri bolunmesi ile egitilmistir.

## Yontemler

### Veri Seti
- Kaynak: Wisconsin Breast Cancer (data.csv)
- Ornek sayisi: 569 (212 Malign, 357 Benign)
- Ozellik sayisi: 30
- Hedef degisken: diagnosis (M=1, B=0)

### On Isleme
- id sutunu cikarilmistir.
- Etiket kodlama: M=1, B=0
- Bolunme: %70 egitim, %15 dogrulama, %15 test (stratified, random_state=42)
- StandardScaler yalnizca egitim seti uzerinde fit edilmistir.

### Hiperparametreler

| Parametre | Deger |
|-----------|-------|
| Rastgele tohum (seed) | 42 |
| Ogrenme orani | 0.01 |
| Epoch sayisi | 1000 |
| Optimizer | SGD |
| Kayip fonksiyonu | Binary Cross Entropy |
| Gizli katman aktivasyonu | ReLU |
| Cikis aktivasyonu | Sigmoid |
| Agirlik baslatma | He Initialization |
| Batch boyutu | Tam batch (varsayilan) / 32 (mini-batch) |
| L2 lambda | 0.0 (varsayilan) / 0.01 (regularizasyonlu) |

### Model Yapilari

| Model | Mimari |
|-------|--------|
| Temel | Giris(30) -> Dense(64, ReLU) -> Dense(1, Sigmoid) |
| Genis | Giris(30) -> Dense(128, ReLU) -> Dense(1, Sigmoid) |
| Derin | Giris(30) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid) |
| L2 Regularizasyonlu | Temel ile ayni, lambda=0.01 |
| Mini-batch | Temel ile ayni, batch_size=32 |

### Overfitting Analizi
- Egitim ve dogrulama dogruluk/kayip farki incelenmistir.
- En iyi dogrulama epoch'u ve minimum dogrulama kaybi epoch'u raporlanmistir.
- Early stopping gerekliligi degerlendirilmistir.

### Model Secimi
- En yuksek dogrulama dogruluguna sahip model secilmistir.
- Tum modeller 1000 epoch ile egitildigi icin, genelleme performansi belirleyici olmustur.

### Degerlendirme Metrikleri
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix (TN, FP, FN, TP)

## Sonuclar

Tum modellerin test seti sonuclari asagidaki tabloda verilmistir.
Detayli ciktilar `proje.ipynb` dosyasinda mevcuttur.

| Model | Test Acc | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| NumPy Temel (64) | ~0.988 | ~1.000 | ~0.969 | ~0.984 |
| NumPy Genis (128) | ~0.988 | ~1.000 | ~0.969 | ~0.984 |
| NumPy Derin (64-32) | ~0.988 | ~1.000 | ~0.969 | ~0.984 |
| NumPy L2 Reg | ~0.988 | ~1.000 | ~0.969 | ~0.984 |
| NumPy Mini-batch | ~0.977 | ~1.000 | ~0.938 | ~0.968 |
| Scikit-learn MLP | ~0.988 | ~1.000 | ~0.969 | ~0.984 |
| PyTorch | ~0.977 | ~1.000 | ~0.938 | ~0.968 |

## Tartisma

- Veri seti iyi ayrilabilir yapidadir. Basit mimariler bile yuksek dogruluk saglamaktadir.
- NumPy ile sifirdan yazilan model, Scikit-learn ile ayni sonuclari vermistir. Bu, geri yayilim ve gradyan hesaplamalarinin dogru uygulandigini gostermektedir.
- Genis model (128 noron) en iyi dogrulama dogrulugunu elde etmis, asiri ogrenmeden kacinmistir.
- Mini-batch egitiminde dogrulama kaybi 204. epoch'ta minimuma ulasmis, devam eden egitim hafif overfitting'e neden olmustur. Early stopping uygulanabilir.
- L2 regularizasyon bu veri setinde belirgin bir fark yaratmamistir.
- Gelecek calismalarda PCA, cross-validation ve dropout gibi yontemler denenebilir.

## Proje Yapisi

```
Odev1/
  src/
    __init__.py
    data_preprocessing.py    # veri yukleme, bolme, standardizasyon
    numpy_model.py           # sifirdan sinir agi (NumPy)
    metrics.py               # accuracy, precision, recall, f1, confusion matrix
    sklearn_model.py         # MLPClassifier
    pytorch_model.py         # nn.Module + SGD
  data/
    data.csv
  models/
    training_curves.png      # egitim egrileri
    confusion_matrices.png   # confusion matrix gorselleri
    metric_comparison.png    # metrik karsilastirma grafigi
  proje.ipynb                # ana notebook (ciktilar dahil)
  requirements.txt
  README.md
```

## Kurulum ve Calistirma

### Sanal ortam olusturma
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### Bagimliliklari yukleme
```bash
pip install -r requirements.txt
```

### Calistirma
Jupyter Notebook uzerinden:
```bash
jupyter notebook proje.ipynb
```

## Tekrarlanabilirlik

Tum rastgele tohumlar 42 olarak sabitlenmistir.
Ayni veri bolunmesi, agirlik baslatma ve hiperparametreler kullanilmistir.
Kod tekrar calistirildiginda ayni sonuclar elde edilecektir.


## Ekler
Veri Seti:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
