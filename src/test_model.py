import joblib
import pandas as pd

# Load the saved model
model = joblib.load("model/category_label.pkl")

print("Model loaded successfully!")
print("Type 'exit' at any point to stop.\n")

while True:
    title = input("Enter product:")
    if title.lower() == "exit":
        print("Exiting...")
        break

    merchant = input("Enter merchant:")
    if merchant.lower() == "exit":
        print("Exiting...")
        break

      # Create a DataFrame from input
    user_input = pd.DataFrame([{
        "product title": title,
        "merchant id": merchant,
        "product code": "UNKNOWN",        # fallback
        "number of views": 0              # fallback
    }])
    # Predict sentiment
    prediction = model.predict(user_input)[0]
    print(f"Predicted category: {prediction}\n" + "-" * 40)