# YZM304 Derin Ogrenme - Proje Odevi 2

CIFAR-10 goruntu veri seti uzerinde Evrisimli Sinir Aglari (CNN) kullanilarak nesne siniflandirma islemleri gerceklestirilmistir. Projede kurgulanan ag mimarileri, laboratuvar standartlarina bagli kalinarak PyTorch kütüphanesi kullanimiyla nesne yonelimli programlama (OOP) mantigi uzerinden insa edilmis; performans ciktilari, Transfer Learning (VGG-16) ve Klasik Makine Ogrenmesi hibritleri (SVM, Random Forest) ile analiz edilerek kiyaslanmistir.

## 1. Giris

Bilgisayarli gorme (Computer Vision) problemlerinde nesne tanima dogruluk oranini artirmak amaciyla cesitli CNN mimarileri tasarlanmistir. Bu calismada; temel bir CNN mimarisinin ogrenme kapasitesi, dropout ve batch normalization gibi optimizasyon tekniklerinin model uzerindeki etkisi, VGG-16 gibi onceden egitilmis (pretrained) derin aglarin basarisi ve destek vektor makineleri (SVM) gibi klasik algoritmalarin CNN özellik cikarimiyla nasil hibritlenebilecegi incelenmistir. Problem cercevesinde her bir mimarinin dogruluk oranlari ve egitim maliyetleri objektif olarak analiz edilmistir.

## 2. Yontemler

### 2.1 Veri Seti Tanitimi ve On Isleme
Veri seti 10 farkli nesne sinifina ait (Ucak, Otomobil, Kus, Kedi, Geyik, Kopek, Kurbaga, At, Gemi, Kamyon), `32x32` piksel boyutlarinda toplam 60.000 adet RGB formunda renkli goruntuden olusmaktadir. Veri seti %83.3 Egitim (50.000 ornek) ve %16.7 Test (10.000 ornek) olacak sekilde bolunmustur. Veri sizintisini engellemek amaciyla setler kesin sinirlarla ayrilmistir. Egitim asamasinda aşırı öğrenmeyi (overfitting) onlemek ve agin genellenebilirligini artirmak amaciyla egitim verilerine yatay cevirme (`RandomHorizontalFlip`) ve kirpma (`RandomCrop(32, padding=4)`) teknikleriyle veri artirma (Data Augmentation) uygulanmistir. Modelin stabil yakinsamasi amaciyla pikseller, kanal bazlı `(0.4914, 0.4822, 0.4465)` ortalama ve `(0.2470, 0.2435, 0.2616)` standart sapma degerleri ile normalize edilmistir.

### 2.2 Nesne Yonelimli Programlama (OOP) Modulasyonu
Odevin kod mimarisi, kapsulleme (encapsulation) kurallarina uygun olarak moduler olarak tasarlanmistir:
- **CNNClassifier:** PyTorch modellerini (BasicCNN, ImprovedCNN vb.) alarak egitim, degerlendirme ve tahmin dongulerini kendi icinde isleyen bagimsiz sarmalayici (wrapper) siniftir.
- **HybridClassifier:** Egitilmis bir Evrisimli Sinir Agini özellik cikarici (Feature Extractor) olarak kullanip, bu özellikleri boyutlandirarak SVM ve Random Forest sistemlerine besleyen özel bir siniftir. Ayrica cıkarılan bu özellikleri ve ilgili etiketleri `.npy` uzantili dosyalar olarak sisteme kaydeder.

### 2.3 Deney Tasarimi ve Mimariler
Agi optimize etmek uzere asagidaki 4 farkli varyasyon test edilmistir. Bütün egitim sureclerinde kayıp fonksiyonu olarak **Cross Entropy Loss**, optimizasyon algoritmasi olarak **Adam Optimizer** kullanilmistir. Adam optimizasyon algoritması, hızlı ve dengeli öğrenme karakteristiği nedeniyle tercih edilmistir. Egitim esnasında model hiperparametreleri varsayılan olarak `Epoch: 10`, `Batch Size: 128` ve `Learning Rate: 0.001` değerlerinde sabitlenmiştir.

- **Model 1 (Temel CNN):** 2 Evrisim (Conv2d) + ReLU + MaxPool ve 3 Tam Baglantili (Linear) Katmandan olusmaktadir.
- **Model 2 (Iyilestirilmis CNN):** Model 1 ile tamamen ayni hiperparametre ve noron sayilarina sahiptir; ancak agin daha duzenli ogrenmesini saglamak uzere katman aralarina `BatchNorm2d` ve `Dropout(p=0.5)` eklenmistir. (Odevin 5. model karsiligi olarak varsayılmaktadır).
- **Model 3 (VGG-16 Transfer):** Orijinal torchvision mimarisi uzerinden cagrilan ve egitilmis (pretrained) agirliklar kullanan derin agdir. Girdi boyutu `224x224` piksele ayarlanmis, sadece son siniflandirma katmani 10 sinif ciktisi verecek sekilde guncellenmistir.
- **Model 4 (Hibrit CNN + ML):** Egitimi bitmis Model 2'nin cikarim mekanizmasi kullanilarak elde edilen (Egitim: `50.000 x 84`, Test: `10.000 x 84` boyutlarındaki) özellik matrislerinin ve etiket kümelerinin `.npy` dosyalarına yazdırılıp, ardından klasik SVM (RBF Kernel) ve Random Forest sistemleri uzerinde egitilip test edilmesiyle olusturulmustur.

## 3. Sonuclar

Tüm hesaplamalar `10.000` orneklik bagimsiz Test seti uzerinde gerceklestirilmistir. Sonuclara ait sayisal veriler ve egitim periyotlari asagida tablolanmistir.

| Model | Test Dogrulugu (Accuracy) | Parametre Sayisi | Egitim Suresi |
|-------|-------------------------|------------------|----------------|
| Model 1 (Temel CNN) | %65.75 | 62,006 | ~328.8s |
| Model 2 (Iyilestirilmis CNN)| %61.60 | 62,006 | ~335.2s |
| Model 3 (VGG-16 Transfer) | %87.64 | 134,301,514 | ~3761.7s |
| Model 4 (Hibrit - SVM) | %61.95 | - | - |
| Model 4 (Hibrit - Random Forest)| %61.68 | - | - |

*(Not: Egitim sureleri donanimsal islem birimlerine (CPU/GPU) gore spesifik degisiklik gosterebilir, referans amaciyla sunulmustur.)*

## 4. Cikarimlar

**4.1. Model Kapasitesi ve Optimizasyon Katmanlarinin Etkisi**
Tablolanan sonuclara gore Model 2 (%61.60), Model 1'in (%65.75) gerisinde kalmistir. Bu istatistiksel dususun temel sebebi, katman sayisi kismen az olan bir sinir aginda yuksek Dropout orani (`p=0.5`) kullanilmasinin egitim periyodunu yavaslatmasidir. Dropout'un getirdigi regülarizasyon baskısı nedeniyle ağ, 10 epoch icerisinde asil ogrenme surecini tamamlayamamaktadir. CIFAR-10 gibi kismen kompleks bir veri setinde iyilestirilmis modelin kapasitesini gosterebilmesi icin Epoch sayisinin artirilmasi veya Dropout degerinin dusurulmesi gerektigi sonucuna varilmistir. PyTorch icerisindeki CrossEntropyLoss fonksiyonunun secimi ise veri seti ve coklu siniflandirma probleminin dogasiyla tutarlidir.

**4.2. Transfer Learning (VGG-16) Performansi**
Onceden egitilmis VGG-16 modeli deneylerde yaklasik %87.64 gibi yuksek bir dogruluk oranina ulasmistir. ImageNet veri kumeleri uzerinde onceden optimize edilmis olan bu derin agin sadece son siniflandirma katmaninin problem uzayina uyarlanmasiyla gösterdiği basari, derin aglarin nesne bazlı cikarim (feature extraction) konusundaki kapasitelerini dogrulamaktadir. Pre-trained mimarilerin kucuk epoch boyutlariyla bile genelleme problemini optimum duzeyde asabildigi gozlemlenmistir.

**4.3. Klasik Makine Ogrenmesi ve Hibrit Modeller**
Model 2 uzerinden vektorize edilen verilerle egitilen Hibrit SVM (%61.95) ve Random Forest (%61.68) modelleri, saf Evrisimli Sinir Agi karsiligi olan Model 2 ile buyuk oranda tutarli performans sergilemistir. Elde edilen sonuclardan yola cikilarak; iyi egitilmis bir CNN mimarisinin ayrik olarak cikarici islevini yurutebildigi, uretilen özellik matrislerinin (`.npy` dosyalariyla disariya aktarildiktan sonra) kanonik makine ogrenmesi algoritmalarina dogrudan beslenip orijinal aga denk bir siniflandirma haritasi elde edilebilecegi kanitlanmistir. 

## 5. Ekler ve Referanslar

**Veri Seti:**
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (Ağın eğitimi ve performans ölçümü için kullanılmıştır.)

**Kullanilan Kütüphaneler:**
- [PyTorch (torch, torchvision)](https://pytorch.org/docs/stable/index.html) (Derin öğrenme modellerinin nesne yönelimli yapılanması `nn.Module`, veri augmentasyon işlemi, kayıp hesaplamaları ve gradyan optimizasyonları için kullanılmıştır.)
- [Scikit-learn](https://scikit-learn.org/stable/) (Hibrit modellerde CNN'in özellik çıkarıcı çıktılarının kullanılarak Destek Vektör Makineleri `SVC` ve Rastgele Orman `RandomForestClassifier` algoritmalarının eğitilmesi ve test edilmesi için kullanılmıştır.)
- [NumPy](https://numpy.org/doc/stable/) (Hibrit modellerde özellik ve etiket setlerinin `.npy` formatında belleğe alınması, matris boyutlandırmaları ve istatistik işlemleri için kullanılmıştır.)
- [Matplotlib / Seaborn](https://matplotlib.org/) (Model öğrenme eğrilerinin ve karmaşıklık matrislerinin ısı haritaları halinde görselleştirilmesi için kullanılmıştır.)

## 6. Proje Yapisi ve Calistirma

Projenin bilgisayar bagimsiz bir sekilde tamamen tekrar edilebilmesi amaciyla olusturulan dosya hiyerarsisi asagidaki sekildedir:

```text
Odev2/
  src/
    __init__.py
    data_preprocessing.py   # Veri yukleme ve Augmentation islemleri
    cnn_models.py           # nn.Module tabanli mimari tasarimlari (BasicCNN, ImprovedCNN, vgg16)
    classifier.py           # CNNClassifier (OOP sarmalayicisi)
    hybrid.py               # HybridClassifier (.npy cikarimi ve SVM/RF egitimi)
    metrics.py              # Istatistiksel cizim metotlari
    utils.py                # Rastgelelik sabitleyiciler ve parametre hesablamalari
  proje.ipynb               # Ciktilari basan otonom test notebook dosyasi
  requirements.txt          # Gerekli kutuphaneler
```

**Sanal Ortam ve Calistirma Yonergesi:**
```bash
python -m venv venv
venv\Scripts\activate      # Windows ortamlarinda
# source venv/bin/activate # Linux/Mac ortamlarinda

pip install -r requirements.txt
jupyter notebook proje.ipynb
```
