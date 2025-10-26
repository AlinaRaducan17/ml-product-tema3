# ml-product-tema3

📦 Product Category Classification

This project trains a machine learning model to predict product categories based on attributes like title, merchant, and metadata. The dataset is sourced from a product listings CSV and the model uses both natural language processing and numerical feature engineering.

🔧 Files Overview

train_model.py
Trains a LinearSVC classification pipeline.

Text features are extracted using TfidfVectorizer on product titles.

Categorical variables are encoded with OneHotEncoder.

Numerical features (e.g., title length, presence of numbers/special characters) are scaled using MinMaxScaler.

Output model is saved to model/category_label1.pkl.

test_model.py
A CLI tool that loads the trained model and predicts categories from user input.

Accepts product title and merchant input.

Automatically generates required features before prediction.

Runs until the user types "exit".

category_label1.pkl
Serialized machine learning pipeline trained on the full dataset, ready for predictions.

.gitignore
Ensures that model files (*.pkl) and the model/ directory are excluded from version control.

category_label1.ipynb
(Optional) Jupyter notebook for exploratory data analysis or model experimentation.

📈 Model Details

Model used: LinearSVC (Support Vector Machine)

Text preprocessing: TfidfVectorizer

Engineered features include:

title_length, word_count

presence of digits or special characters

longest word length and acronyms

📁 Dataset

Source: IMLP4_TASK_03-products.csv

Preprocessed by removing NaNs, normalizing text, and merging similar categories.