import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import joblib

# load dataset from GitHub
url = "https://raw.githubusercontent.com/AlinaRaducan17/ml-product-tema3/main/data/IMLP4_TASK_03-products.csv"

df = pd.read_csv(url)

# drop all rows with missing values
df = df.dropna()

# Clean column names: strip spaces and replace underscores with spaces (or use other formatting)
df.columns = df.columns.str.replace("_", " ").str.strip().str.lower()

df['category label'] = df['category label'].str.strip().str.lower()

# Optional: merge similar categories
df['category label'] = df['category label'].replace({
    'fridge freezers': 'fridges & freezers',
    'fridges': 'fridges & freezers',
    'fridge': 'fridges & freezers',
    'freezers': 'fridges & freezers',
    'mobile phone': 'mobile phones',
    'cpus': 'cpu'
})

# Drop columns that are not useful for modeling
df = df.drop(columns=['merchant rating', 'product id', 'listing date'])

#Criteria for category predict
def extract_title_features(df):
    df = df.copy()
    df["title_length"] = df["product title"].str.len()
    df["word_count"] = df["product title"].str.split().apply(len)
    df["has_numbers"] = df["product title"].str.contains(r'\d').astype(int)
    df["has_special_chars"] = df["product title"].str.contains(r'[^\w\s]').astype(int)
    df["has_upper_acronyms"] = df["product title"].str.contains(r'\b[A-Z]{2,}\b').astype(int)
    df["longest_word_length"] = df["product title"].str.split().apply(lambda words: max([len(word) for word in words]) if words else 0)
    return df

df = extract_title_features(df)

# Features and label
X = df[[
    "product title", "merchant id", "product code", "number of views",
    "title_length", "word_count", "has_numbers", "has_special_chars",
    "has_upper_acronyms", "longest_word_length"
]]

y = df["category label"]

# Preprocessor pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "product title"),
        ("merchant", OneHotEncoder(handle_unknown='ignore'), ["merchant id"]),
        ("code", OneHotEncoder(handle_unknown='ignore'), ["product code"]),
        ("numeric", MinMaxScaler(), [
            "number of views", "title_length", "word_count",
            "has_numbers", "has_special_chars", "has_upper_acronyms",
            "longest_word_length"
        ])
    ]
)

# Define pipeline with the best model (e.g. RandomForestClassifier)
pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LinearSVC())
])

# Train the model on the entire dataset
pipeline.fit(X, y)

import os

# Create 'model' directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# Save the model to a file
joblib.dump(pipeline, "model/category_label1.pkl")

print("Model trained and saved as 'model/category_label1.pkl'")