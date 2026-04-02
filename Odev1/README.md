# YZM304 Derin Ogrenme - Proje Odevi 1

Wisconsin Breast Cancer veri seti uzerinde meme kanseri (Malign / Benign) teshisi amaciyla ikili siniflandirma yapan sinir agi modelleri gelistirilmistir. Calisma kapsaminda ag yapisi laboratuvar uygulamasina sadik kalinarak NumPy kütüphanesiyle nesne yonelimli programlama (OOP) uzerinden kurgulanmis; ciktilar Scikit-learn ve PyTorch karsiliklariyla kiyaslanmistir.

## 1. Giris (Introduction)

Tibbi tani veroklerinde yanlis negatif (False Negative) oranlarinin minimize edilmesi projenin oncelikli hedefidir. Calismada, sinir aglarinin ogrenme dinamiklerini incelemek uzere laboratuvar altyapisi referans alinan bir Cok Katmanli Algilayici (MLP) kurgulanmistir. Model, mimarideki degisimlerin (katman/noron sayilari) ve farkli optimizasyon yontemlerinin (mini-batch, L2 regularizasyonu) basariya etkisini olcmek uzere optimize edilmistir. Manuel yazilan uygulamanin sonuclari Scikit-learn (MLPClassifier) ve PyTorch ile uretilen aglarla eslestirilerek matematiksel isleyis dogrulanmistir.

## 2. Yontemler (Methods)

2.1. Veri ve On Isleme
Veri seti 569 ornek (212 Malign, 357 Benign) icermektedir. Veri sizintisini (data leakage) engellemek amaciyla veri seti %70 Egitim, %15 Dogrulama (Validation) ve %15 Test olmak uzere tabakali (stratified) oranlarda bolunmustur. StandardScaler fonksiyonu yalnizca egitim seti uzerinde egitilmis ve diger veri kumeleri bu egitime gore donusturulmustur.

2.2. Model Mimarisi
Gelistirilen NeuralNetwork sinifinda gizli katmanlar icin ReLU, tahminsel donusum (cikis) katmani icin Sigmoid aktivasyon fonksiyonlari secilmistir. Agirlik baslatma yonteminde He Initialization uygulanmistir. Kayip fonksiyonu Ikili Capraz Entropi (Binary Cross-Entropy) olarak belirlenmis ve Stochastic Gradient Descent (SGD) algoritmasi ile geri yayilim turevleri alinmistir. Sinif modulleri encapsulation mantigi ile private (gizli) metotlar cercevesinde kurgulanmistir.

2.3. Kontrollu Rastgelelik
Algoritmalarin (NumPy, Scikit-learn, PyTorch) ilk agirlik atamalarinda adil bir kiyaslama olusturmak adina Seed (Random State) degeri 42 olarak ayarlanmistir. 

2.4. Deney Tasarimi ve Hiperparametreler
Model parametreleri 1000 epoch boyunca 0.01 ogrenme orani ile bes farkli varyasyonda test edilmistir:
- Temel: 1 Gizli Katman (64 Noron)
- Genis: 1 Gizli Katman (128 Noron)
- Derin: 2 Gizli Katman (64 ve 32 Noron)
- L2 Reg: Temel modele L2 Regularizasyonu eklentisi (Lambda: 0.01)
- Mini-Batch: Temel modele Mini-batch eklentisi (Batch Size: 32)

## 3. Sonuclar (Results)

Ogrenme egrilerinin yalin accuray ile degerlendirilmesi yaniltici sonuc urettiginden, final test verileri (86 Ornek) uzerinden analizler karmasiklik matrisine donusturulmustur. Tum egitim sureclerine ait Isi haritalari (Heatmap), Loss/Accuracy egrileri ve metriklerin cubuk grafikleri models klasorune kaydedilmistir.

| Model | Test Acc | Precision| Recall | F1 Score | Confusion Matrix (TN, FP, FN, TP) |
|-------|----------|----------|--------|----------|-----------------------------------|
| NumPy Temel (64) | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| NumPy Genis (128) | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| NumPy Derin (64-32) | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| NumPy L2 Reg (0.01) | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| NumPy Mini-batch | %97.67 | 1.000 | 0.9375 | 0.9677 | TN=54, FP=0, FN=2, TP=30 |
| Scikit-learn MLP | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| PyTorch (64) | %97.67 | 1.000 | 0.9375 | 0.9677 | TN=54, FP=0, FN=2, TP=30 |

## 4. Tartisma ve Model Analizleri (Discussion)

4.1. Confusion Matrix Gerekliligi
Tip tespit sistemlerinde hatali teshisin gozden kacilmamasi (False Negative) Accuracy oranlarindan daha elzemdir. Bu nedenden oturu Recall ve F1 Score degerleri referans alinmis ve modellerin False Negative (FN) retme egilimleri esas test degiskeni olarak ayarlanmistir.

4.2. Overfitting (Yuksek Varyans) Analizi
Mini-batch kullanilan varyasyonda validation loss degeri 204. epoch periyodunda minimum (0.0807) seyrinde kalmis, ilerleyen donemde dogruluk artisina karsin validation kaybi ivmelenerek (0.1286) artmistir. Bu durum dogrudan sekilde Overfitting (yuksek varyans) kanitidir. Sisteme entegre edilebilecek bir Early Stopping fonksiyonu ile asiri ogrenmenin onune gecilerek genelleme yetenegi korunabilir.

4.3. Underfitting (Yuksek Bias) Analizi
Ag egitim surecinde hicbir model Underfitting (yetersiz ogrenme / yuksek bias) bulgusu uretmemistir. Ogrenen modeller kisa egitim denemelerinde %97 performans barajini gecmektedir. Veri setinin iyi ayrilabilir uclar sunmasindan dolayi 64 noronlu temel mimari verinin yapisini formilize etmek icin yeterli dogrusal olmayan sinir kapasitesi yakalamaktadir. Hatta katman sayisinin artirildigi (Derin model kurgusu) durumda test performansi sicramasi yasanmamasina ragmen bias seviyesi stabil kalarak; mevcut kapasitenin ogrenme gorevi icin optimum oldugu tespit edilmistir. 

4.4. Dogruluk ve Epoch Kriterine Gore Model Secimi
Farkli mimariler ile test edilen varyasyonlar icinde veriyi ezberlemekten uzaklasarak en hizli sekilde accuracy istikrarini saglayan yapi Genis Model varyasyonudur. Negatif bir validation/train kayıp araligi (GAP degeri: -0.0108) urettigi hesaplanan genis mimarinin optimum dengesine 554. epok sirasinda eristigi sonucuna ulasilmistir.

4.5. Scikit-Learn ve PyTorch Altyapi Karsilastirmasi
Uygulamasi gelistirilen sinif yapisi, ayni hiperparametrelerin atandigi Scikit-learn MLPClassifier kutuphanesi ile capraz test isleminden gecirilmistir. Iki mimarinin Test uzayi baglaminda TP=31, FN=1 ve oransal dogruluklarda (Accuracy: 0.9884, Recall: 0.968) hata payi vermeden eslestigi gozlemlenmistir. OOP yapisina kurulan manuel matris geri yayilim (Backpropagation) mekanizmalarinin optimizasyon sagladigi ve islevselligi pruzsuz olarak dogrulanmistir.


## 5. Ekler ve Referanslar (References)
Veri Seti: 
[Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
From scratch referanslari:
- [NumPy Official Documentation](https://numpy.org/doc/stable/): Matris carpimlari ve gradyan operasyonlarinin (dot product, broadcasting) algoritma ici tespiti referans alinmistir.
- [Scikit-learn Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html): Karmasiklik matrisi parcalanmasi ve siniflandirma metriklerinin denklemleri hedeflenmistir.
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html): Standardizasyon ve olceklendirme adimlari uygulanmistir.
- [PyTorch nn.Module Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html): Sinir agi katmanlarinin ve noronlarin Nesne yonelimli (OOP) yapilanmasina kiyaslama tutulmustur.


## 6. Proje Yapisi ve Calistirma (Structure & Usage)

Platform bagimsiz tekrar edilebilirlik acisindan proje hiyerarsisi asagida listelenmistir:

```text
Odev1/
  src/
    __init__.py
    data_preprocessing.py    
    numpy_model.py           
    metrics.py               
    sklearn_model.py         
    pytorch_model.py         
  data/
    data.csv                 
  models/
    training_curves.png      
    confusion_matrices.png   
    metric_comparison.png    
  proje.ipynb                
  requirements.txt

Sanal Ortam ve Calistirma Yonergesi:

python -m venv venv
venv\Scripts\activate      # Windows ortamlarinda
# source venv/bin/activate # Linux/Mac ortamlarinda

pip install -r requirements.txt
jupyter notebook proje.ipynb
