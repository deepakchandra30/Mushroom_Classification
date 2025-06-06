import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.model_selection import train_test_split
from pathlib import Path

def read_csv_dropna():
    df = pd.read_csv("files/agaricus-lepiota.csv")
    df = df.dropna()
    shape = df.shape
    print(f"The shape of the dataset after dropping null attributes is {shape}")
    return df

def encode_categoricals(df):
    label_encoders = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    label_encoders_path = "files/label_encoders.pkl"
    joblib.dump(label_encoders, label_encoders_path)
    print(f"Lables have been encoded successfully and pickle file is saved to {label_encoders_path}")
    return df

def split_features(df):
    X = df.drop("class", axis = 1)
    y = df["class"]
    return X,y

def test_train_split(X,y,df):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    
    print("Balance check:")
    print(df['class'].value_counts())

    Path("files").mkdir(exist_ok=True)

    joblib.dump(X_train,"files/X_train.pkl")
    joblib.dump(X_test,"files/X_test.pkl")
    joblib.dump(y_train,"files/y_train.pkl")
    joblib.dump(y_test,"files/y_test.pkl")
    print("✅ Train/test splits saved to 'files/' folder.")

def preprocess():
    df = read_csv_dropna()
    df = encode_categoricals(df)
    X,y = split_features(df)
    test_train_split(X,y,df)

preprocess()