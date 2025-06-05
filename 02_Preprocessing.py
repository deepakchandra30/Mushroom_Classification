import pandas as pd

def preprocess():
    df = pd.read_csv("agaricus-lepiota.csv")
    df = df.dropna()

    print(df.shape)


preprocess()