# for data manipulation
import pandas as pd
import os

# for data preprocessing
from sklearn.model_selection import train_test_split

# Hugging Face API
from huggingface_hub import HfApi

# ✅ Use GitHub Actions secret
api = HfApi(token=os.environ["HF_TOKEN"])

# ✅ Load dataset from Hugging Face Hub
DATASET_PATH = "hf://datasets/sankar-guru/tourism-dataset/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")

# Target variable
target = "ProdTaken"

# Drop unnecessary column
tourism_dataset = tourism_dataset.drop(columns=["CustomerID"], errors="ignore")

# Drop missing values
tourism_dataset = tourism_dataset.dropna()

# Split features and target
X = tourism_dataset.drop(columns=[target])
y = tourism_dataset[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload processed files back to Hugging Face dataset
for file_path in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="sankar-guru/tourism-dataset",
        repo_type="dataset",
    )
