from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

def load_data():
    X_train = joblib.load("files/X_train.pkl")
    X_test = joblib.load("files/X_test.pkl")
    y_train = joblib.load("files/y_train.pkl")
    y_test = joblib.load("files/y_test.pkl")

    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test = load_data()

print(X_train.shape)