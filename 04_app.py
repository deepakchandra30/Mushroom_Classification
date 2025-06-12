from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

label_encoders = joblib.load("files/label_encoders.pkl")

models = {
    "Logistic Regression": joblib.load("files/Logistic_regression.pkl"),
    "Naive Bayes": joblib.load("files/Naive_bayes.pkl"),
    "Random Forest": joblib.load("files/Random_Forest.pkl"),
    "Decision Tree": joblib.load("files/Decision_Tree.pkl")
}

def load_model_metrics(path="files/evaluations/model_metrics_report.csv"):
    metrics = []
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

# Full descriptive names for each feature's codes
feature_fullnames = {
    "cap-shape": {
        "b": "bell",
        "c": "conical",
        "x": "convex",
        "f": "flat",
        "k": "knobbed",
        "s": "sunken"
    },
    "cap-surface": {
        "f": "fibrous",
        "g": "grooves",
        "y": "scaly",
        "s": "smooth"
    },
    "cap-color": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "r": "green",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "bruises": {
        "t": "bruises",
        "f": "no bruises"
    },
    "odor": {
        "a": "almond",
        "l": "anise",
        "c": "creosote",
        "y": "fishy",
        "f": "foul",
        "m": "musty",
        "n": "none",
        "p": "pungent",
        "s": "spicy"
    },
    "gill-attachment": {
        "a": "attached",
        "d": "descending",
        "f": "free",
        "n": "notched"
    },
    "gill-spacing": {
        "c": "close",
        "w": "crowded",
        "d": "distant"
    },
    "gill-size": {
        "b": "broad",
        "n": "narrow"
    },
    "gill-color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "g": "gray",
        "r": "green",
        "o": "orange",
        "p": "pink",
        "u": "purple",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "stalk-shape": {
        "e": "enlarging",
        "t": "tapering"
    },
    "stalk-root": {
        "b": "bulbous",
        "c": "club",
        "u": "cup",
        "e": "equal",
        "z": "rhizomorphs",
        "r": "rooted",
        "?": "missing"
    },
    "stalk-surface-above-ring": {
        "f": "fibrous",
        "y": "scaly",
        "k": "silky",
        "s": "smooth"
    },
    "stalk-surface-below-ring": {
        "f": "fibrous",
        "y": "scaly",
        "k": "silky",
        "s": "smooth"
    },
    "stalk-color-above-ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "stalk-color-below-ring": {
        "n": "brown",
        "b": "buff",
        "c": "cinnamon",
        "g": "gray",
        "o": "orange",
        "p": "pink",
        "e": "red",
        "w": "white",
        "y": "yellow"
    },
    "veil-type": {
        "p": "partial",
        "u": "universal"
    },
    "veil-color": {
        "n": "brown",
        "o": "orange",
        "w": "white",
        "y": "yellow"
    },
    "ring-number": {
        "n": "none",
        "o": "one",
        "t": "two"
    },
    "ring-type": {
        "c": "cobwebby",
        "e": "evanescent",
        "f": "flaring",
        "l": "large",
        "n": "none",
        "p": "pendant",
        "s": "sheathing",
        "z": "zone"
    },
    "spore-print-color": {
        "k": "black",
        "n": "brown",
        "b": "buff",
        "h": "chocolate",
        "r": "green",
        "o": "orange",
        "u": "purple",
        "w": "white",
        "y": "yellow"
    },
    "population": {
        "a": "abundant",
        "c": "clustered",
        "n": "numerous",
        "s": "scattered",
        "v": "several",
        "y": "solitary"
    },
    "habitat": {
        "g": "grasses",
        "l": "leaves",
        "m": "meadows",
        "p": "paths",
        "u": "urban",
        "w": "waste",
        "d": "woods"
    }
}

label_map = {
    'e': 'Edible',
    'p': 'Poisonous'
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    model_outputs = {}
    metrics = load_model_metrics()
    form_data = {}

    if request.method == "POST":
        # Collect user inputs
        for feature in label_encoders:
            if feature == "class":
                continue
            form_data[feature] = request.form.get(feature)

        # Encode inputs
        encoded_input = []
        for feature, le in label_encoders.items():
            if feature == "class":
                continue
            val = form_data.get(feature)
            encoded_val = le.transform([val])[0]
            encoded_input.append(encoded_val)

        final_input = np.array(encoded_input).reshape(1, -1)

        # Get selected models; default to Decision Tree if none selected
        selected_models = request.form.getlist("model_selection")
        if not selected_models:
            selected_models = ["Decision Tree"]

        for name in selected_models:
            model = models.get(name)
            if model:
                pred = model.predict(final_input)[0]
                pred_label = label_encoders["class"].inverse_transform([pred])[0]
                model_outputs[name] = label_map.get(pred_label.lower(), pred_label)

        prediction = model_outputs

    return render_template(
        "index.html",
        prediction=prediction,
        metrics=metrics,
        label_encoders=label_encoders,
        feature_fullnames=feature_fullnames,
        form_data=form_data,
        models=list(models.keys())  # pass models list for checkbox rendering
    )

if __name__ == "__main__":
    app.run(debug=True)