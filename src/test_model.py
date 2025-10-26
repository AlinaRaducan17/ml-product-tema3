import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/category_label1.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

# Feature engineering function (same as used in training)
def extract_title_features(df):
    df = df.copy()
    df["title_length"] = df["product title"].str.len()
    df["word_count"] = df["product title"].str.split().apply(len)
    df["has_numbers"] = df["product title"].str.contains(r'\d').astype(int)
    df["has_special_chars"] = df["product title"].str.contains(r'[^\w\s]').astype(int)
    df["has_upper_acronyms"] = df["product title"].str.contains(r'\b[A-Z]{2,}\b').astype(int)
    df["longest_word_length"] = df["product title"].str.split().apply(
        lambda words: max([len(word) for word in words]) if words else 0)
    return df

while True:
    title = input("Enter product: ")
    if title.lower() == "exit":
        print("Exiting...")
        break

    merchant = input("Enter merchant: ")
    if merchant.lower() == "exit":
        print("Exiting...")
        break

    # Prepare input DataFrame
    user_input = pd.DataFrame([{
        "product title": title,
        "merchant id": merchant,
        "product code": "UNKNOWN",        # fallback
        "number of views": 0              # fallback
    }])

    # Add engineered features
    user_input = extract_title_features(user_input)

    # Predict category
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 40)
