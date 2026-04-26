"""
Upload the ENTIRE project repository to the rsd-06/PRRegressionAuditEnv Space.
Explicitly uploads each project folder and key root files one by one to ensure 
absolute isolation from the virtual environment (.venv) and other junk.
"""
from huggingface_hub import HfApi
import os

api = HfApi()
SPACE_ID = "rsd-06/PRRegressionAuditEnv"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print(f"Uploading FULL repository to Space: {SPACE_ID}")
print(f"   Local Root: {ROOT}")

# 1. Folders to upload
folders = ["assets", "evaluation", "prevaluation_env", "scripts", "training"]

for folder in folders:
    local_path = os.path.join(ROOT, folder)
    if os.path.exists(local_path):
        print(f"  -> Uploading folder: {folder}/")
        api.upload_folder(
            folder_path=local_path,
            path_in_repo=folder,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"deploy: sync {folder}/ folder",
            ignore_patterns=["**/__pycache__/**", "**/*.pyc"]
        )

# 2. Files in the root to upload
root_files = ["Dockerfile", "README.md", "hfblog.md", "requirements.txt", ".gitignore"]

for file in root_files:
    local_path = os.path.join(ROOT, file)
    if os.path.exists(local_path):
        print(f"  -> Uploading file: {file}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=file,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"deploy: update {file}"
        )

print(f"\nDONE - Full repository is now live on Hugging Face!")
print(f"Space URL: https://huggingface.co/spaces/{SPACE_ID}")
print(f"App URL: https://rsd-06-prregressionauditenv.hf.space/")
print("\nThe Space Docker build will restart automatically (2-4 mins).")
