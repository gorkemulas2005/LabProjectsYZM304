import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class CNNClassifier:
    """
    PyTorch CNN modellerini egitmek, degerlendirmek ve tahmin yapmak icin kullanilan 
    Nesne Yonelimli (OOP) sarmalayici (wrapper) sinif.
    """
    def __init__(self, model, lr=0.001, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, train_loader, val_loader=None, epochs=10):
        """Modeli verilen egitim seti uzerinde egitir."""
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
            epoch_loss = running_loss / total
            epoch_acc = 100. * correct / total
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%", end="")
            
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f" | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            else:
                print()
                
        return history

    def evaluate(self, val_loader):
        """Modeli dogrulama/test seti uzerinde degerlendirir."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss = running_loss / total
        val_acc = 100. * correct / total
        return val_loss, val_acc

    def predict(self, test_loader):
        """Test seti uzerinde tahmin yapar ve gercek/tahmin edilen etiketleri dondurur."""
        self.model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                
        return y_true, y_pred
