import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class HybridClassifier:
    """
    Egitilmis bir CNN modelini ozellik cikarici (Feature Extractor) olarak kullanarak,
    ozellikleri .npy formatinda kaydeder ve ardindan Klasik Makine Ogrenmesi 
    (SVM veya Random Forest) modellerini egitir.
    """
    def __init__(self, feature_extractor, ml_type='svm', data_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        # Son siniflandirma katmanini (Linear) kaldirarak yalnizca ozellik cikarmasini saglariz.
        # ImprovedCNN icerisindeki classifier asagidaki sekildedir:
        # Flatten(), Linear(), ReLU(), Dropout(), Linear(), ReLU(), Dropout(), Linear()
        # Biz sondan bir onceki (num_classes cikaran) katmani atip oradaki ozellikleri istiyoruz.
        # Basitce agin 'features' kismini + Flatten kismini calistirip, 
        # isterseniz ilk linear katmandan sonrasini dondurebilirsiniz.
        # En saglam yol, modele forward ederken son fc'yi bypass etmektir, 
        # ancak nn.Sequential yapisi nedeniyle 'features' uzerinden direkt cikis alip Flatten yapacagiz.
        
        self.ml_type = ml_type
        if ml_type == 'svm':
            self.ml_model = SVC(kernel='rbf', random_state=42)
        elif ml_type == 'rf':
            self.ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("ml_type 'svm' veya 'rf' olmalidir.")
            
        if data_dir is None:
            self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        else:
            self.data_dir = data_dir
            
        os.makedirs(self.data_dir, exist_ok=True)

    def _extract_features(self, dataloader):
        """Dataloader uzerinden gecer ve CNN'den cikarilan ozellikleri vektorlestirir."""
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                # Modelin sadece 'features' (Conv) kismini kullanip Flatten yapiyoruz
                # Boylece tam anlamiyla CNN'in gorsel ozellik cikarim gucunu kullaniyoruz.
                x = self.feature_extractor.features(inputs)
                x = x.view(x.size(0), -1) # Flatten
                
                features_list.append(x.cpu().numpy())
                labels_list.append(labels.numpy())
                
        features = np.vstack(features_list)
        labels = np.concatenate(labels_list)
        return features, labels

    def fit(self, train_loader, save_prefix='train'):
        """Egitim setinden ozellikleri cikarir, .npy kaydeder ve ML modelini egitir."""
        print(f"[{self.ml_type.upper()}] Egitim seti ozellikleri cikariliyor...")
        features, labels = self._extract_features(train_loader)
        
        features_path = os.path.join(self.data_dir, f"{save_prefix}_features.npy")
        labels_path = os.path.join(self.data_dir, f"{save_prefix}_labels.npy")
        
        np.save(features_path, features)
        np.save(labels_path, labels)
        
        print(f"-> Ozellikler kaydedildi: {features_path} (Boyut: {features.shape})")
        print(f"-> Etiketler kaydedildi: {labels_path} (Boyut: {labels.shape})")
        
        print(f"[{self.ml_type.upper()}] Makine ogrenmesi modeli egitiliyor...")
        self.ml_model.fit(features, labels)
        print("-> Egitim tamamlandi.")

    def predict(self, test_loader, save_prefix='test'):
        """Test setinden ozellikleri cikarir, .npy kaydeder ve tahmin yapar."""
        print(f"[{self.ml_type.upper()}] Test seti ozellikleri cikariliyor...")
        features, labels = self._extract_features(test_loader)
        
        features_path = os.path.join(self.data_dir, f"{save_prefix}_features.npy")
        labels_path = os.path.join(self.data_dir, f"{save_prefix}_labels.npy")
        
        np.save(features_path, features)
        np.save(labels_path, labels)
        
        print(f"-> Test Ozellikleri kaydedildi: {features_path} (Boyut: {features.shape})")
        print(f"-> Test Etiketleri kaydedildi: {labels_path} (Boyut: {labels.shape})")
        
        y_pred = self.ml_model.predict(features)
        return labels.tolist(), y_pred.tolist()
