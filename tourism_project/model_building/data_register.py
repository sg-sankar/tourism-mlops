from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Hugging Face dataset repo
repo_id = "sankar-guru/tourism-dataset"
repo_type = "dataset"

# Initialize API client (uses your HF login)
api = os.getenv("HF_TOKEN")


#api = HfApi(token=os.getenv("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"))

# Check if dataset exists, else create
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Upload data from Drive
api.upload_folder(
    folder_path="/content/drive/MyDrive/tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
