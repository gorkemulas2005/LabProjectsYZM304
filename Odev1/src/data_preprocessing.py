import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42

def load_data(path="data/data.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["id"])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    X = df.drop(columns=["diagnosis"]).values
    y = (df["diagnosis"] == "M").astype(int).values.reshape(-1, 1)
    return X, y, df

def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def standardize(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test, scaler
