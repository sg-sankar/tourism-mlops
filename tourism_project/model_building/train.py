# =========================
# Data & ML libraries
# =========================
import pandas as pd
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

import xgboost as xgb
import joblib
import mlflow

from datasets import load_dataset
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

# =========================
# MLflow setup
# =========================
#mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_experiment("MLOps_Tourism_Training")
# Use local file storage in CI, server locally
if os.environ.get("GITHUB_ACTIONS"):
    mlflow.set_tracking_uri("file:./mlruns")
else:
    mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("MLOps_Tourism_Training")

# =========================
# Hugging Face API (CI-safe)
# =========================
api = HfApi(token=os.environ["HF_TOKEN"])

# =========================
# Load dataset from HF Hub
# =========================
#dataset = load_dataset("sankar-guru/tourism-dataset")
#dataset = load_dataset("sankar-guru/tourism-dataset", download_mode="force_redownload")
#df = dataset["train"].to_pandas()
# Download CSV directly from HF to bypass schema cache issues
csv_path = hf_hub_download(
    repo_id="sankar-guru/tourism-dataset",
    filename="tourism.csv",
    repo_type="dataset"
)
df = pd.read_csv(csv_path)

print("Dataset loaded successfully.")

# =========================
# Target and split
# =========================
target = "ProdTaken"

X = df.drop(columns=[target, "CustomerID", "Unnamed: 0"])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Feature separation
# =========================
numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=["object"]).columns.tolist()

# Handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# =========================
# Preprocessing
# =========================
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# =========================
# Model
# =========================
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# =========================
# Hyperparameter grid
# =========================
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

# =========================
# Training + MLflow logging
# =========================
with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1
    )
    grid_search.fit(Xtrain, ytrain)

    # Log CV results
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    # Best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # Evaluation
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_recall": train_report["1"]["recall"],
        "test_accuracy": test_report["accuracy"],
        "test_recall": test_report["1"]["recall"],
    })

    # =========================
    # Save model locally
    # =========================
    model_path = "best_tourism_model_v1.joblib"

    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved as {model_path}")

    # =========================
    # Upload model to HF Hub
    # =========================
    repo_id = "sankar-guru/tourism-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_tourism_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

    print("Model uploaded to Hugging Face Model Hub.")
