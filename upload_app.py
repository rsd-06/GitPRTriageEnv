import os
from huggingface_hub import upload_file

def upload():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("HF_TOKEN not found in environment, please set it.")
        return
        
    try:
        upload_file(
            path_or_fileobj="app.py",
            path_in_repo="app.py",
            repo_id="SaiSanjayR/GitPRTriage_Environment",
            repo_type="space",
            token=hf_token
        )
        print("Successfully uploaded app.py to Space.")
    except Exception as e:
        print(f"Error uploading app.py: {e}")

if __name__ == "__main__":
    upload()
