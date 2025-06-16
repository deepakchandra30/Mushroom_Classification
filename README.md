# 🍄 Mushroom Classification

This project aims to classify mushrooms as **edible** or **poisonous** using the UCI Mushroom Dataset. It follows a structured machine learning pipeline, from raw data preprocessing to model training and deployment with a basic Flask app.

## 🚀 Project Overview

The project performs the following steps:
1. **Data Conversion**: Converts `.data` format to `.csv`.
2. **Data Preprocessing**: Encodes categorical variables, handles missing values, and prepares data for modeling.
3. **Model Training**: Trains multiple machine learning models and selects the best based on performance metrics.
4. **Web Deployment**: Provides a simple Flask web app for interactive mushroom classification based on user input.

## 🧰 Libraries Used

- `pandas` – data manipulation
- `numpy` – numerical operations
- `scikit-learn` – machine learning models and preprocessing
- `matplotlib`, `seaborn` – data visualization
- `flask` – web application interface
- `joblib` – model saving and loading

## 📂 Key Files

- `01_Convert_data_to_csv.py` – Converts UCI dataset to CSV.
- `02_Preprocessing.py` – Cleans and encodes the data.
- `03_Train_Models.py` – Trains and evaluates multiple classifiers.
- `04_app.py` – Flask app for user interaction.
- `templates/` – HTML templates for the web app.
- `files/` – Stores serialized models and encoders.

## 💻 How to Run

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run each script in order**:

   ```bash
   python 01_Convert_data_to_csv.py
   python 02_Preprocessing.py
   python 03_Train_Models.py
   ```

3. **Launch the Flask web app**:

   ```bash
   python 04_app.py
   ```

Then, open your browser at `http://localhost:5000` and try predicting mushrooms yourself!

## 📊 Model Performance

Models like Random Forest and Decision Trees achieve high accuracy (≥97%) on the dataset due to its rich feature space and minimal noise.

## 📜 Credits

Dataset Source: [UCI Mushroom Dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom)
Author: Deepak Chandra Nallamothu

