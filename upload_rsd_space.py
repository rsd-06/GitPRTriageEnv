"""
Upload all files needed by the rsd-06/PRRegressionAuditEnv Space.
Uploads: Dockerfile, prevaluation_env/ (server + data)
Skips: .venv, venv, evaluation/, training/, __pycache__, egg-info
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
    commit_message="deploy: update Dockerfile",
)

# 1.5 Upload README.md
print("  -> Uploading README.md...")
api.upload_file(
    path_or_fileobj=os.path.join(ROOT, "README.md"),
    path_in_repo="README.md",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="deploy: update README",
)

# 1.6 Upload hfblog.md
print("  -> Uploading hfblog.md...")
api.upload_file(
    path_or_fileobj=os.path.join(ROOT, "hfblog.md"),
    path_in_repo="hfblog.md",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="deploy: upload hfblog",
)

# 2. Upload prevaluation_env/ recursively (server code + PR data + templates)
print("  -> Uploading prevaluation_env/ (server code + PR data + dashboard template)...")
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
    commit_message="deploy: update server code and dashboard template",
)

print(f"\nDONE - Upload complete!")
print(f"Space URL: https://huggingface.co/spaces/{SPACE_ID}")
print(f"App URL: https://rsd-06-prregressionauditenv.hf.space/")
print(f"Health URL: https://rsd-06-prregressionauditenv.hf.space/health")
print("\nThe Space Docker build will start automatically (2-4 mins).")
print(f"Watch the build at: https://huggingface.co/spaces/{SPACE_ID}")
