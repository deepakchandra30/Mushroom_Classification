from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

    results = []

    for name,model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test,predictions)
        prec = precision_score(y_test,predictions)
        rec = recall_score(y_test,predictions)
        f1 = f1_score(y_test,predictions)
        cm = confusion_matrix(y_test,predictions)

        joblib.dump(model,f"files/{name.replace(' ','_')}.pkl")

        print(f"\n {name}")
        print(f" Accuracy: {acc}")
        print(f" Precision: {prec}")
        print(f" Recall: {rec}")
        print(f" F1 Score: {f1}")
        print(f"✅ Model saved to 'files/{name.replace(' ', '_')}.pkl'")

        #confusion matrix
        plt.figure(figsize = (4,3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Edible", "Poisonous"],yticklabels=["Edible", "Poisonous"])
        plt.title(f"Confusion matrix of {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"files/evaluations/confusion_matrix_{name.replace(" ", "_")}.png")
        plt.close()

        results.append({
            "Model" : name,
            "Accuracy" : acc,
            "Precision Score" : prec,
            "Recall" : rec,
            "F1 Score" : f1,
        })

    return pd.DataFrame(results)



if __name__ == "__main__":
    print("Loading train/test datasets...")
    X_train,X_test,y_train,y_test = load_data()

    print("🚀 Training the models and evaluating performance...")
    report_df = train_the_models(X_train,X_test,y_train,y_test)

    print("\n📊 Overall Model Comparison:")
    print(report_df.to_string(index=False))

    report_df.to_csv("files/evaluations/model_metrics_report.csv", index=False)
    print("\n📁 Report saved to: files/evaluations/model_metrics_report.csv")