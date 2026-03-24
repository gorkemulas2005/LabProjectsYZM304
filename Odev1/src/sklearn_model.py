import numpy as np
from sklearn.neural_network import MLPClassifier


def train_sklearn(X_train, y_train, X_test, y_test, hidden_layers=(64,), lr=0.01, max_iter=1000, seed=42):
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='sgd',
        learning_rate_init=lr,
        max_iter=max_iter,
        random_state=seed,
        batch_size=len(X_train),
        learning_rate='constant',
        verbose=False
    )
    clf.fit(X_train, y_train.ravel())
    y_pred = clf.predict(X_test).reshape(-1, 1)
    return clf, y_pred
