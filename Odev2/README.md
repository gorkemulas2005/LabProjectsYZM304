# Odev2: CIFAR-10 Goruntu Siniflandirma (CNN & Hibrit Modeller)

Bu proje kapsaminda CIFAR-10 veri seti kullanilarak 5 farkli sinir agi ve makine ogrenmesi yaklasimi (Sifirdan CNN, Gelistirilmis CNN, Transfer Learning, CNN + SVM ve CNN + Random Forest) uzerinde siniflandirma performanslari kiyaslanmistir. Tüm modeller PyTorch kütüphanesi ve Nesne Yonelimli Programlama (OOP) prensipleri dogrultusunda kurgulanmistir.

## 1. Giris
Evrisimli Sinir Aglari (CNN), goruntu isleme ve bilisavarlarla gorme alaninda standart kabul edilen mimarilerdir. Ancak bir agin sifirdan tasarlanmasi, hiperparametre optimizasyonu ve mevcut on-egitilmis agirliklarin (Transfer Learning) bu sureclere entegrasyonu performansi dogrudan etkileyen faktorlerdir. 
Bu calismanin amaci, ayni veri seti uzerinde tamamen ayni sartlarda calisan temel bir CNN mimarisi ile, BatchNorm ve Dropout gibi duzenleme (regularization) teknikleri uygulanmis bir varyasyonunu kiyaslamaktir. Ek olarak, derin bir literatur agi olan VGG-16'nin transfer ogrenme gucu ve ayrica bir CNN'in yalnizca ozellik cikarici (feature extractor) olarak kullanilip klasik ML (SVM, Random Forest) modelleriyle hibritlendigi senaryolar degerlendirilmistir.

## 2. Metotlar

### 2.1 Veri Seti ve On Isleme
Veri seti olarak `CIFAR-10` secilmistir (10 sinif, 60.000 adet 32x32 boyutlarinda RGB goruntu).
* **Veri Artirma (Data Augmentation)**: Modelin asiri ogrenmesini engellemek ve genellenebilirligini artirmak amaciyla egitim setine yatay cevirme (`RandomHorizontalFlip`) ve kirpma (`RandomCrop(32, padding=4)`) islemleri uygulanmistir.
* **Normalizasyon**: Pikseller `[0,1]` araligina alindiktan sonra CIFAR-10 kanal ortalamalari `(0.4914, 0.4822, 0.4465)` ve standart sapmalari `(0.2470, 0.2435, 0.2616)` kullanilarak standardize edilmistir. 
* **VGG-16 Duzenlemesi**: Model 3 (Transfer Learning) calistirilirken VGG-16'nin girdi boyutu beklentisi nedeniyle goruntuler `Resize(224)` ile buyutulmustur.

### 2.2 Model Mimarileri ve Kullanilan Parametreler
Secilen tum modeller icin ortak hiperparametreler su sekildedir:
* **Optimizer**: `Adam`. Adaptif ogrenme orani ve momentum birlesimi sayesinde hizli yakinsama hedeflenmistir. 
* **Learning Rate**: `0.001`. Adam optimizasyonu icin stabil varsayilan deger secilmistir.
* **Loss Fonksiyonu**: `CrossEntropyLoss`. Cok sinifli siniflandirma problemleri icin standart kayip metrigidir.
* **Batch Size**: `64`. GPU bellek (NVIDIA RTX 4050) ve islem hizi optimum oranda tutulmustur.

**Gelistirilen 5 Modelin Detaylari:**
1. **Model 1 (Temel CNN):** LeNet-5 mimarisinin CIFAR-10 (3x32x32) formatina uyarlanmis halidir. 2 Convolutional (Conv2d) + MaxPool katmani ardindan 3 Tam Baglantili (Linear) katman icermektedir. 20 epoch egitilmistir.
2. **Model 2 (Gelistirilmis CNN):** Model 1 ile mimarisi ve epoch sayisi (20 epoch) tamamen aynidir. Ancak hizli ogrenim ve over-fitting kontrolu amaciyla her `Conv2d` sonrasi `BatchNorm2d`, FC katmanlari arasina ise `Dropout(p=0.5)` eklenmistir.
3. **Model 3 (VGG-16 Transfer Learning):** Orijinal `torchvision.models` modulunden onceden egitilmis (Pretrained) VGG-16 cekilmistir. Ozellik cikarici (`features`) katmanlari dondurulmus (freeze), yalnizca son FC katmani 10 sinifa yeniden duzenlenerek ag 10 epoch boyunca egitilmistir.
4. **Model 4 (Hibrit CNN + ML):** Egitilmis Model 2, sadece "ozellik cikarici" olarak konumlandirilmistir. Son FC katmanindan onceki 84 boyutlu neron ciktilari yakalanarak `.npy` dosyalari olarak (Train: 16 MB, Test: 3.2 MB) diske kaydedilmistir. Bu cikarilan matrisler, RBF kernelli DVM (SVM, C=10) ve Random Forest (n_estimators=200) uzerinde egitilmistir.
5. **Model 5 (Hibrit Kiyaslama):** Odev sartlarinda belirtildigi uzere ayni test setini kullanan klasik tam CNN (Model 2) ile Hibrit (Model 4) sonuclari birbirine karsi mukayese edilmistir.

## 3. Sonuclar

Model kiyaslamalarina ait Accuracy metrikleri ve egitim sureleri (RTX 4050 uzerinde) asagidaki tabloda verilmistir:

| Model | Test Accuracy | Egitim Suresi |
| :--- | :--- | :--- |
| Model 1 (Temel CNN) | **65.75%** | 328.8s |
| Model 2 (Iyilestirilmis CNN) | 61.60% | 335.2s |
| Model 3 (VGG-16 Transfer) | **87.64%** | 3761.7s |
| Model 4 (Hibrit - SVM) | 61.95% | -- |
| Model 4 (Hibrit - Random Forest) | 61.68% | -- |

*Grafik ciktilari (Loss, Confusion Matrix vb.) kod klasorundeki `outputs` altinda bulunmaktadir.*

## 4. Cikarimlar ve Analiz

Elde edilen sonuclara gore modellerin performanslari ve veri isleme surecleri uzerine su cikarimlar yapilmistir:

**1. Sifirdan Kurulan Aglar: Model 1 ve Model 2 Karsilastirmasi**
Ayni mimariye (LeNet-5 tabanli) sahip olmalarina ragmen, temel Model 1 (%65.75) ve iyilestirilmis Model 2 (%61.60) arasinda sasirtici bir sonuc elde edilmistir. 
- *Neden Model 1 Daha Iyi?* Model 2'ye eklenen `Dropout(p=0.5)` katmani, agdaki neronlarin yarisini rastgele devredisi birakarak kati bir ceza (regularization) uygulamaktadir. 20 epoch'luk kisa bir egitim suresinde, Dropout'un ogrenmeyi yavaslatma etkisi baskin cikmis ve ag tam kapasitesine ulasamamistir. 
- *Egitim Suresi:* Her iki model de 330 saniye bandinda egitilmistir (Model 1: 328.8s, Model 2: 335.2s). BatchNorm katmaninin getirdigi ufak hesaplama yuku goz ardi edilebilir seviyededir.

**2. On Egitimli (Pretrained) Ag: Model 3 (VGG-16)**
Model 3, yalnizca 10 epoch egitilmesine ragmen **%87.64** test dogrulugu ile en basarili model olmustur. 
- *Cikarim:* ImageNet veri setiyle onceden egitilen VGG-16, kenar ve doku gibi temel ozellikleri halihazirda taniyabilmektedir. CIFAR-10 goruntuleri `224x224` boyutuna cikarilip bu aga verildiginde, ag sadece son siniflandirma katmanini guncelleyerek inanilmaz bir isabet oranina ulasmistir. Ancak bu islem, `3761.7 saniye` (yaklasik 1 saat) surmus ve devasa (134 milyon parametreli) mimarinin islemsel maliyetini gozler onune sermistir.

**3. Klasik Makine Ogrenmesi ve CNN Hibritlemesi: Model 4**
Model 2'nin son FC katmanindan hemen onceki `84 boyutlu` tensörler cikarilmis (Egitim: 50.000 x 84 matris) ve klasik makine ogrenmesi algoritmalarina verilmistir.
- *SVM (%61.95)* ve *Random Forest (%61.68)* modelleri, onlara ozellik cikarici olarak hizmet eden Model 2 (%61.60) ile neredeysa birebir ayni performansi gostermistir.
- *Cikarim:* Klasik bir ML modeli (SVM/RF), girdi olarak ham pikseller yerine **iyi ayiklanmis neron ciktilarini (feature maps)** aldiginda, derindeki bir yapay sinir agi (CNN) kadar basarili olabilmektedir. Bu durum, CNN'lerin asil gucunun "siniflandirmadan" ziyade "otomatik ozellik cikarimi" yapmalarinda yattigini matematiksel olarak ispatlamaktadir.

## 5. Sonuc
CIFAR-10 veri seti gibi kompleks (dusuk cozunurluklu fakat cesitli nesneler barindiran) problemlerde:
1. Donanim (GPU) gucu elverdigi surece **Transfer Learning (VGG-16 vb.)** tartismasiz en iyi performansi (%87.64) sunmaktadir.
2. Sifirdan model egitilecekse `Dropout` oranlari, epoch sayisina gore (kisa sureli egitimler icin p=0.2 gibi) optimize edilmelidir.
3. Disk uzerinde `.npy` formatinda ozellik tasiyarak kurulan Hibrit sistemler, klasik ML algoritmalarina derin ogrenme kabiliyeti kazandirabilen gecerli bir alternatiftir.

## 6. Ekler ve Referanslar

1. CIFAR-10 Dataset: Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*.
2. VGG-16 Mimarisi: Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*.
3. Adam Optimizer: Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*.
4. PyTorch ve Scikit-Learn Resmi Dokumantasyonlari.
**Veri Seti:**
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html) (Ağın eğitimi ve performans ölçümü için kullanılmıştır.)

**Kullanilan Kütüphaneler:**
- [PyTorch (torch, torchvision)](https://pytorch.org/docs/stable/index.html) (Derin öğrenme modellerinin nesne yönelimli yapılanması `nn.Module`, veri augmentasyon işlemi, kayıp hesaplamaları ve gradyan optimizasyonları için kullanılmıştır.)
- [Scikit-learn](https://scikit-learn.org/stable/) (Hibrit modellerde CNN'in özellik çıkarıcı çıktılarının kullanılarak Destek Vektör Makineleri `SVC` ve Rastgele Orman `RandomForestClassifier` algoritmalarının eğitilmesi ve test edilmesi için kullanılmıştır.)
- [NumPy](https://numpy.org/doc/stable/) (Hibrit modellerde özellik ve etiket setlerinin `.npy` formatında belleğe alınması, matris boyutlandırmaları ve istatistik işlemleri için kullanılmıştır.)
- [Matplotlib / Seaborn](https://matplotlib.org/) (Model öğrenme eğrilerinin ve karmaşıklık matrislerinin ısı haritaları halinde görselleştirilmesi için kullanılmıştır.)
