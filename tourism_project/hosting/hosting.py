from huggingface_hub import HfApi
import os

# Use hard-coded token OR env var (you prefer hard-coded)
HF_TOKEN = "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"

api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="/content/drive/MyDrive/tourism_project/deployment",
    repo_id="sankar-guru/tourism-app",
    repo_type="space",
    path_in_repo="",
)

print("Deployment files uploaded to Hugging Face Space")
