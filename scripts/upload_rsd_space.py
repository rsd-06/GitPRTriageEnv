"""
Upload the ENTIRE project repository to the rsd-06/PRRegressionAuditEnv Space.
Uses 'git ls-files' to perfectly identify all project files, ensuring NO 
virtual environment or junk files are uploaded.
"""
import subprocess
import os
from huggingface_hub import HfApi

api = HfApi()
SPACE_ID = "rsd-06/PRRegressionAuditEnv"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print(f"Uploading FULL repository (via git-tracked files) to Space: {SPACE_ID}")

# 1. Get the list of tracked files
try:
    files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
except Exception as e:
    print(f"Error running git ls-files: {e}")
    files = []

# 2. Filter out any accidental __pycache__ or other unwanted files that might be tracked
filtered_files = [
    f for f in files 
    if not f.startswith("__pycache__") 
    and not ".venv" in f 
    and not ".git" in f
    and not f.endswith(".pyc")
]

print(f"Found {len(filtered_files)} files to synchronize.")

# 3. Upload files in a batch
# Using upload_folder with allow_patterns based on the git list is the most efficient.
api.upload_folder(
    folder_path=ROOT,
    repo_id=SPACE_ID,
    repo_type="space",
    allow_patterns=filtered_files,
    commit_message="deploy: full repository sync (git-tracked files only)",
)

print(f"\nDONE - Full repository is now live on Hugging Face!")
print(f"Space URL: https://huggingface.co/spaces/{SPACE_ID}")
print("\nThe Space Docker build will restart automatically (2-4 mins).")
