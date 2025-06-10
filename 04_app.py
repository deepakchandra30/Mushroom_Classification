from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

label_endcoders = joblib.load("files/label_encoders.pkl")

models = {
    "Logistic Regression" : joblib.load("files/Logistic_regression.pkl"),
    "Naive Bayes" : joblib.load("files/Naive_bayes.pkl"),
    "Random Forest": joblib.load("files/Random_Forest.pkl"),
    "Decision Tree" : joblib.load("files/Decision_Tree.pkl")
}

def load_model_metrics(path="files/evaluations/model_metrics_report.csv"):
    metrics= []
    if os.path.exists(path):
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            metrics.append({
                "model": row["Model"],
                "accuracy": float(row["Accuracy"]),
                "precision": float(row["Precision Score"]),
                "recall": float(row["Recall"]),
                "f1": float(row["F1 Score"])
            })

    return metrics  

@app.route("/", methods = ["GET","POST"])
def index():
    prediction = None
    model_outputs = {}
    metrics = load_model_metrics()

    if request.method == 'POST':
        input_data = {}
        for feature in label_endcoders:
            continue
        input_data[feature] = request.form.get(feature)
         