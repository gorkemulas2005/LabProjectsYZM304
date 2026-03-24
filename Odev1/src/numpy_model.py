import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, lr=0.01, lambda_reg=0.0, seed=42):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.lambda_reg = lambda_reg
        self.n_layers = len(layer_sizes) - 1

        np.random.seed(seed)
        self.weights = []
        self.biases = []
        for i in range(self.n_layers):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(W)
            self.biases.append(b)

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        A = X
        for i in range(self.n_layers - 1):
            Z = A @ self.weights[i] + self.biases[i]
            self.z_values.append(Z)
            A = self._relu(Z)
            self.activations.append(A)
        Z = A @ self.weights[-1] + self.biases[-1]
        self.z_values.append(Z)
        A = self._sigmoid(Z)
        self.activations.append(A)
        return A

    def compute_loss(self, y, y_hat):
        m = y.shape[0]
        eps = 1e-8
        y_hat = np.clip(y_hat, eps, 1 - eps)
        bce = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        if self.lambda_reg > 0:
            reg = sum(np.sum(W ** 2) for W in self.weights)
            bce += (self.lambda_reg / (2 * m)) * reg
        return bce

    def backward(self, y):
        # Geri yayilim: gradyan hesaplama
        m = y.shape[0]
        self.d_weights = []
        self.d_biases = []
        # Cikis katmani gradyani: dL/dZ = a - y (sigmoid + BCE)
        dA = self.activations[-1] - y
        for i in range(self.n_layers - 1, -1, -1):
            dW = (1/m) * (self.activations[i].T @ dA)
            db = (1/m) * np.sum(dA, axis=0, keepdims=True)
            if self.lambda_reg > 0:
                dW += (self.lambda_reg / m) * self.weights[i]
            self.d_weights.insert(0, dW)
            self.d_biases.insert(0, db)
            if i > 0:
                dA = dA @ self.weights[i].T
                dA = dA * self._relu_deriv(self.z_values[i-1])

    def update(self):
        for i in range(self.n_layers):
            self.weights[i] -= self.lr * self.d_weights[i]
            self.biases[i] -= self.lr * self.d_biases[i]

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y)

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=None, verbose=True):
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        m = X_train.shape[0]
        if batch_size is None:
            batch_size = m
        for epoch in range(epochs):
            idx = np.random.permutation(m)
            X_s, y_s = X_train[idx], y_train[idx]
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                self.forward(X_s[start:end])
                self.backward(y_s[start:end])
                self.update()
            y_hat_tr = self.forward(X_train)
            y_hat_val = self.forward(X_val)
            history["train_loss"].append(self.compute_loss(y_train, y_hat_tr))
            history["val_loss"].append(self.compute_loss(y_val, y_hat_val))
            history["train_acc"].append(self.accuracy(X_train, y_train))
            history["val_acc"].append(self.accuracy(X_val, y_val))
            if verbose and (epoch + 1) % 200 == 0:
                tl = history["train_loss"][-1]
                vl = history["val_loss"][-1]
                ta = history["train_acc"][-1]
                va = history["val_acc"][-1]
                print(f"Epoch {epoch+1}/{epochs} - loss: {tl:.4f} - val_loss: {vl:.4f} - acc: {ta:.4f} - val_acc: {va:.4f}")
        return history
