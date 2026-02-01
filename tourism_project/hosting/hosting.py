from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Hugging Face API (CI/CD safe)
api = HfApi(token=os.environ["HF_TOKEN"])

# Hugging Face Space details
repo_id = "sankar-guru/tourism-app"
repo_type = "space"

# Local deployment folder inside repo
DEPLOY_PATH = "tourism_project/deployment"

print("Current dir:", os.getcwd())
print("Deploy path exists:", os.path.isdir(DEPLOY_PATH))
print("Files:", os.listdir(DEPLOY_PATH))

# Ensure Space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists.")
except RepositoryNotFoundError:
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        space_sdk="gradio",   # or "streamlit" if you use that
        private=False,
    )
    print(f"Space '{repo_id}' created.")

# Upload deployment files to Space
api.upload_folder(
    folder_path=DEPLOY_PATH,
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
)

print("Deployment files uploaded to Hugging Face Space")
