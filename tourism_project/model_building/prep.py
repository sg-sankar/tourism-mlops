# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("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"))
DATASET_PATH = "hf://datasets/sankar-guru/tourism-dataset/tourism.csv"

tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'ProdTaken'

# Drop unnecessary ID column
tourism_dataset = tourism_dataset.drop(columns=["CustomerID"], errors="ignore")

# Remove missing values
tourism_dataset = tourism_dataset.dropna()

# Define predictor matrix (X) and target variable (y)
X = tourism_dataset.drop(columns=[target])
y = tourism_dataset[target]

# Split dataset into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Save train and test files
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload train and test files to Hugging Face dataset
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="sankar-guru/tourism-dataset",
        repo_type="dataset",
    )
