# YZM304 Derin Ogrenme - Proje Odevi 1

Wisconsin Breast Cancer veri seti uzerinde meme kanseri (Malign / Benign) teshisi amaciyla ikili siniflandirma yapan sinir agi modelleri gelistirilmistir. Calisma kapsaminda ag yapisi laboratuvar kodunun altyapisi baz alinarak NumPy uzerinden nesne yonelimli programlama (OOP) yapisi ile olusturulmus; ardindan elde edilen sonuclar Scikit-learn ve PyTorch kutuphaneleriyle kiyaslanmistir.

## 1. Giris (Introduction)

Tibbi tani problemlerinde yanlis negatif oranlarinin minimize edilmesi oncelikli hedeftir. Bu projede, sinir aglarinin ogrenme dinamiklerini analiz etmek amaciyla 13.03.2026 tarihli laboratuvar uygulamasi altyapisi kullanilarak bir Cok Katmanli Algilayici (MLP) kurgulanmistir. Model, farkli katman sayilari ve guncelleme stratejileri (mini-batch, L2 regularizasyonu) ile egitilmis ve optimize edilmistir. Uretilen modelin ciktilari Scikit-learn (MLPClassifier) ve PyTorch karsiliklari ile kiyaslanarak matematiksel isleyis dogrulanmistir.

## 2. Yontemler (Methods)

2.1. Veri ve On Isleme
Veri seti 569 ornek (212 Malign, 357 Benign) ve 30 sayisal ozellik icermektedir. Veri sizintisini onlemek amaciyla veri seti sirasiyla %70 Egitim, %15 Dogrulama (Validation) ve %15 Test oranlarinda tabakali (stratified) olarak bolunmustur. StandardScaler yalnizca egitim seti uzerinde egitilmis ve test setleri bu olcek uzerinden donusturulmustur.

2.2. Model Mimarisi
Gelistirilen NeuralNetwork sinifi, gizli katmanlarda ReLU, cikis katmaninda Sigmoid aktivasyon fonksiyonlarini kullanmaktadir. Agirlik baslatma (Weight Initialization) isleminde He Initialization yontemi tercih edilmistir. Kayip fonksiyonu olarak Ikili Capraz Entropi (Binary Cross-Entropy) uygulanmis ve Stochastic Gradient Descent (SGD) algoritmasi ile geri yayilim hesaplanmistir. Sınıf icerisindeki moduller encapsulation (private metotlar) ile tanimlanmistir.

2.3. Kontrollu Rastgelelik
Modellerin agirlik atamalarinda ve veri bolme islemlerinde Seed (Random State) degeri 42 olarak sabitlenmistir. 

2.4. Deney Tasarimi ve Hiperparametreler
Model kurgusu uzerinde 5 farkli varyasyon 1000 epoch boyunca egitilmistir (Ogrenme Orani: 0.01):
- Temel: 1 Gizli Katman (64 Noron)
- Genis: 1 Gizli Katman (128 Noron)
- Derin: 2 Gizli Katman (64 ve 32 Noron)
- L2 Reg: Temel model yapisina L2 Regularizasyonu eklentisi (Lambda: 0.01)
- Mini-Batch: Temel model yapisina Mini-batch eklentisi (Batch Size: 32)

## 3. Sonuclar (Results)

Ogrenme oranlarinin tek basina degerlendirilmesi eksik cikarimlara yol acabileceginden, test verileri uzerinden Confusion Matrix (Karmasiklik Matrisi) metrikleri hesaplanmistir. Hesaplanan Isi haritalari (Heatmap), egitim loss/accuracy egrileri ve metriklerin bar kiyaslamalari "models" klasoru altinda paylasilmistir (Test Seti Ornek Sayisi: 86).

| Model | Test Acc | Precision| Recall | F1 Score | Confusion Matrix (TN, FP, FN, TP) |
|-------|----------|----------|--------|----------|-----------------------------------|
| NumPy Temel (64) | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| NumPy Genis (128) | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| NumPy Derin (64-32) | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| NumPy L2 Reg (0.01) | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| NumPy Mini-batch | %97.67 | 1.000 | 0.9375 | 0.9677 | TN=54, FP=0, FN=2, TP=30 |
| Scikit-learn MLP | %98.84 | 1.000 | 0.9687 | 0.9841 | TN=54, FP=0, FN=1, TP=31 |
| PyTorch (64) | %97.67 | 1.000 | 0.9375 | 0.9677 | TN=54, FP=0, FN=2, TP=30 |

## 4. Tartisma ve Overfitting Analizi (Discussion)

1. Confusion Matrix Gerekliligi: 
Model testinde yalnizca dogruluk (accuracy) metrigi kullanilmamistir. Tibbi tahminlerde accuracy oranlarindan ziyade kanser tespit edilememe (False Negative) hatalarindan kacinmak onceliklidir. Bu sebeple calismada Recall ve F1 Score degerleri ana karsilastirma metrikleri olarak atanmis ve False Negative (FN) oranlari uzerinden degerlendirme yapilmistir.

2. Overfitting (Asiri Ogrenme) Analizi: 
Egitim surecinde Mini-batch modelinin validation kaybi 204. epoch itibariyla minimum (0.0807) seviyesine inmis, epochlarin devam etmesiyle (1000. epok) kayip degerinin yeniden artis (0.1286) ivmesine girdigi hesaplanmistir. Egitim loss degerinin azalmasina karsin validation loss degerinin yukselmesi overfitting tablosunu kanitlamaktadir. Early Stopping fonksiyonu devreye alinarak yaklasik 200. epoch civarinda egitimin kesilmesiyle genelleme yetenegi artirilabilecektir. Ote yandan, Genis Model varyasyonunda negatif bir GAP (-0.0108) hesaplanmis olup modelin optimizasyona (good fit) eristigi gorulmustur.

3. Scikit-Learn ve PyTorch Altyapi Kiyaslamasi: 
Manuel NumPy matrisleri ile kurgulanan temel model; Scikit-learn standart kutuphanesi kapsamindaki MLPClassifier ile tamamen ayni hiperparametreler altinda test edilmistir. Test uzayinda modellerin birbirleriyle ayni dogruluk (Accuracy: 0.9884) ve hatirlama (Recall: 0.968) oranlari verdigi, Confusion Matrix elemanlarinin dahi (TP=31, FN=1) eslestigi gorulmustur. Kurgulanan geri yayilim (Backpropagation) matris turev algoritmalarinin amaca ulastigi dogrulanmistir.

## 5. Proje Yapisi ve Calistirma (Structure & Usage)

Proje dosya hiyerarsisi asagida listelenmistir:

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
venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
jupyter notebook proje.ipynb

## 6. Ekler ve Referanslar (References)

Veri Seti:

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

From scratch referanslari:

NumPy Official Documentation: Matris carpimlari ve gradyan operasyonlarinin (dot product, broadcasting) dogrulanmasi.

Scikit-learn Metrics and Scoring: Karmasiklik matrisinin hesaplanmasi ve siniflandirma metriklerinin altyapisindaki denklemlerin karsilastirilmasi.

Scikit-learn Preprocessing: Veri on isleme surecleri.

PyTorch nn.Module Documentation: Sinir agi katmanlarinin ve fonksiyonlarinin (OOP) standartlarina uygun olarak kiyaslanmasi.
