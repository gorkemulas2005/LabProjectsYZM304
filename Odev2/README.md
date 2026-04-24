# Odev 2: Nesne Yönelimli Derin Öğrenme ile CIFAR-10 Sınıflandırma

Bu proje kapsamında CIFAR-10 veri seti kullanılarak görüntü sınıflandırma problemi çözülmüştür. Çalışmada beş farklı model mimarisi denenmiş, Nesne Yönelimli Programlama (OOP) prensipleri kullanılarak temiz ve modüler bir kod yapısı oluşturulmuştur.

## 1. Giriş (Introduction)

Görüntü sınıflandırma, bilgisayarlı görü alanının temel problemlerinden biridir. Bu çalışmanın temel amacı, farklı derin öğrenme ve klasik makine öğrenmesi algoritmalarının karmaşık görüntü veri setleri üzerindeki performanslarını analiz etmek ve karşılaştırmaktır. CIFAR-10 veri seti; 10 farklı sınıfa ait 60.000 adet 32x32 piksel renkli görüntüden oluşmaktadır. Proje gereksinimleri doğrultusunda, tüm yapı Nesne Yönelimli Programlama (OOP) metodolojisine uygun olarak kapsüllenmiştir.

## 2. Metodoloji (Methodology)

### 2.1. Veri Ön İşleme
Veri seti yüklenirken modelin genellenebilirliğini artırmak amacıyla Veri Artırma (Data Augmentation) teknikleri (Rastgele Kırpma, Yatay Döndürme) kullanılmıştır. Görüntüler `[0.4914, 0.4822, 0.4465]` ortalama ve `[0.2023, 0.1994, 0.2010]` standart sapma değerleri ile normalize edilmiştir. VGG-16 modeli için ise görüntüler 224x224 piksel boyutlarına yeniden boyutlandırılmıştır (`resize_224=True`).

### 2.2. Model Mimarileri
Projede birbirini kapsayan ve gelişen 5 farklı model test edilmiştir:
1. **Model 1 (Temel CNN):** 3 evrişim (convolution) ve 2 tam bağlantılı (fully connected) katmandan oluşan standart derin öğrenme modeli.
2. **Model 2 (Geliştirilmiş CNN):** Temel CNN modeline Aşırı Öğrenmeyi (Overfitting) engellemek amacıyla Batch Normalization ve Dropout (%50) katmanları eklenmiş versiyon.
3. **Model 3 (VGG-16 Transfer Learning):** ImageNet veri setinde önceden eğitilmiş VGG-16 modeli. Önceden eğitilmiş ağırlıklar dondurulmuş (freeze), sadece son tam bağlantılı sınıflandırma katmanı (Classifier) CIFAR-10 için 10 sınıfa indirgenerek yeniden eğitilmiştir.
4. **Model 4 (Hibrit - SVM):** Geliştirilmiş CNN (Model 2) mimarisi "Özellik Çıkarıcı" (Feature Extractor) olarak kullanılmış, elde edilen matrisler `.npy` formatında dışa aktarılarak Destek Vektör Makineleri (SVM) ile sınıflandırılmıştır.
5. **Model 5 (Hibrit - Random Forest):** Hibrit modelin sınıflandırıcı kısmı SVM yerine Rastgele Orman (Random Forest) algoritması kullanılarak kurulmuştur.

Tüm bu mimariler `src/classifier.py` ve `src/hybrid.py` içerisindeki `CNNClassifier` ve `HybridClassifier` sınıfları (class) aracılığıyla nesne yönelimli olarak eğitilmiş ve değerlendirilmiştir.

## 3. Bulgular (Results)

Araştırma sonucunda `proje.ipynb` üzerinden elde edilen nihai Test Doğruluk (Accuracy) oranları aşağıda sunulmuştur:

- **Model 1 (Temel CNN):** %71.0
- **Model 2 (Geliştirilmiş CNN):** %61.0
- **Model 3 (VGG-16 Transfer Learning):** %86.0
- **Model 4 (Hibrit - SVM):** %71.0
- **Model 5 (Hibrit - Random Forest):** %54.0

### Performans Analizi
Elde edilen sayılara göre Model 3 (VGG-16) beklendiği gibi %86 doğruluk oranı ile en başarılı model olmuştur. İlginç bir şekilde, Dropout ve Batch Normalization eklenen Model 2 (%61), Temel CNN olan Model 1'in (%71) gerisinde kalmıştır. Bu durum, veri seti için kullanılan hiperparametrelerin (Epoch=10) Model 2'nin kapasitesini tam doldurmasına (convergence) yetmediğine işaret etmektedir.

Klasik Makine Öğrenmesi hibridasyonlarında SVM (%71) oldukça başarılı bir genelleme yaparken, Random Forest (%54) CNN'den gelen yüksek boyutlu sürekli değişken özelliklerinde (continuous features) karar ağaçlarının doğası gereği zorlanmıştır.

## 4. Tartışma ve Sonuç (Discussion)

Çalışma, Transfer Learning (VGG-16) metodunun sıfırdan model eğitmeye (Model 1 & Model 2) kıyasla devasa bir avantaj sağladığını deneysel olarak ispatlamıştır. Sadece son katmanı eğitilen bir VGG-16, 10 epoch gibi kısa bir sürede bile son teknoloji seviyesine yaklaşabilmektedir.

Ayrıca, Hibrit modellerin (Model 4 & Model 5) başarısı doğrudan özellik çıkarıcı (Feature Extractor) olarak kullanılan CNN'in (Model 2) başarısına bağlıdır. Model 2'nin %61 doğrulukta kalması, ondan özellik alan SVM ve RF modellerinin de potansiyel sınırlarını belirlemiştir. Nesne Yönelimli Programlama ile kurulan bu altyapı, ileride yeni modellerin (ResNet, DenseNet vb.) sisteme tek bir satır kod ile entegre edilmesine olanak sağlamaktadır.

---
**Not:** Modellerin eğitilmiş ağırlıkları `.pth` formatında olup boyutları çok büyük olduğu için GitHub'a yüklenmemiştir. Ağırlıklar `models/` klasörüne lokal eğitim sırasında kaydedilir. CIFAR-10 verisi ise `data_preprocessing.py` içindeki `torchvision` tarafından otomatik indirilmektedir.
