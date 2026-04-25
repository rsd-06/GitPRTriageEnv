"""
Upload only the files needed by the Space Docker build.
Uploads: Dockerfile, prevaluation_env/ (server + data), README.md
Skips: .venv, venv, evaluation/grpo_dataset (binary), training/, __pycache__, egg-info
"""
from huggingface_hub import HfApi
import os

api = HfApi()
SPACE_ID = "rsd-06/PRRegressionAuditEnv"
ROOT = os.path.abspath(".")

print(f"Uploading ENV server files to Space: {SPACE_ID}")

# 1. Upload Dockerfile
print("  -> Uploading Dockerfile...")
api.upload_file(
    path_or_fileobj=os.path.join(ROOT, "Dockerfile"),
    path_in_repo="Dockerfile",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="deploy: add Dockerfile",
)

# 2. Upload prevaluation_env/ recursively (server code + PR data)
# Exclude: .venv, __pycache__, *.pyc, egg-info, evaluation/
print("  -> Uploading prevaluation_env/ (server code + PR data)...")
api.upload_folder(
    folder_path=os.path.join(ROOT, "prevaluation_env"),
    path_in_repo="prevaluation_env",
    repo_id=SPACE_ID,
    repo_type="space",
    ignore_patterns=[
        "**/__pycache__/**",
        "**/*.pyc",
        "**/.venv/**",
        "**/venv/**",
        "**/*.egg-info/**",
        "**/evaluation/**",
        "**/.gitignore",
        "**/pyrightconfig.json",
        "**/uv.lock",
        "**/validate-submission.sh",
    ],
    commit_message="deploy: add prevaluation_env server code and PR dataset",
)

print(f"\n✅ Upload complete!")
print(f"Space URL: https://huggingface.co/spaces/{SPACE_ID}")
print(f"Health URL: https://rsd-06-prregressionauditenv.hf.space/health")
print("\nThe Space Docker build will start automatically (2-4 mins).")
print("Watch the build at: https://huggingface.co/spaces/rsd-06/PRRegressionAuditEnv")
