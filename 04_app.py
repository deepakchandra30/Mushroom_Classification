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
