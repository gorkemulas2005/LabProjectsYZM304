import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TorchNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_size))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_pytorch(X_train, y_train, X_test, y_test, hidden_sizes=[64], lr=0.01, epochs=1000, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.FloatTensor(y_train)
    X_te = torch.FloatTensor(X_test)

    model = TorchNet(X_train.shape[1], hidden_sizes)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                tr_acc = ((model(X_tr) >= 0.5).float() == y_tr).float().mean().item()
                te_acc = ((model(X_te) >= 0.5).float() == torch.FloatTensor(y_test)).float().mean().item()
            print(f"Epoch {epoch+1}/{epochs} - loss: {loss.item():.4f} - train_acc: {tr_acc:.4f} - test_acc: {te_acc:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = (model(X_te) >= 0.5).int().numpy()
    return model, y_pred
