from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import pandas as pd

def load_data():
    X_train = joblib.load("files/X_train.pkl")
    X_test = joblib.load("files/X_test.pkl")
    y_train = joblib.load("files/y_train.pkl")
    y_test = joblib.load("files/y_test.pkl")

    return X_train,X_test,y_train,y_test

def train_the_models(X_train, X_test,y_train,y_test):
    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes" : GaussianNB(),
        "Decision Tree" : DecisionTreeClassifier(random_state=42)
    }

    results = {}

    for name,model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test,predictions)
        results[name] = acc
        joblib.dump(model,f"files/{name.replace(' ','_')}.pkl")
        print(f"{name} Accuracy: {acc:4f} ✅ Model saved to 'files/{name.replace(' ', '_')}.pkl'")

    return results



if __name__ == "__main__":
    print("Loading train/test datasets...")
    X_train,X_test,y_train,y_test = load_data()

    print("🚀 Training the models and evaluating performance...")
    accuracy_results = train_the_models(X_train,X_test,y_train,y_test)

    print("\n📊 Model Comparison:")
    for name, acc in accuracy_results.items():
        print(f"{name}: {acc:.4f}")