import pandas as pd

column_names = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]
def convert_data_to_csv():
    try:
        df = pd.read_csv('agaricus-lepiota.data', header=None, names=column_names)
        print("Dataset loaded successfully!")
        df.to_csv("agaricus-lepiota.csv", index=False)
        print("Data saved as csv file")
    except FileNotFoundError:
        print("❌ File not found. Make sure 'agaricus-lepiota.data' is in the same folder.")

convert_data_to_csv()