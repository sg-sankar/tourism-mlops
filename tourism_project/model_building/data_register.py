print("RUNNING FINAL data_register.py — FIXED PATH")

import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# =========================
# Hugging Face dataset repo
# =========================
repo_id = "sankar-guru/tourism-dataset"
repo_type = "dataset"

# =========================
# Hugging Face API (token from GitHub Actions)
# =========================
api = HfApi(token=os.environ["HF_TOKEN"])

# =========================
# Ensure dataset repo exists
# =========================
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# =========================
# HARD-LOCKED LOCAL DATA PATH
# =========================
DATASET_PATH = "tourism_project/data"

print("Current working directory:", os.getcwd())
print("Using dataset path:", DATASET_PATH)

if not os.path.isdir(DATASET_PATH):
    raise RuntimeError(
        f"Dataset directory not found: {DATASET_PATH}. "
        f"Files here: {os.listdir('.')}"
    )

print("Files found in dataset folder:", os.listdir(DATASET_PATH))

# =========================
# Upload dataset folder
# =========================
api.upload_folder(
    folder_path=DATASET_PATH,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset upload completed successfully.")
