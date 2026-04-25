"""Upload training artifacts from a HF Job container to the Space repo."""

import os

from huggingface_hub import HfApi

api = HfApi()
repo_id = "SaiSanjayR/GitPRTriage_Environment"

# Job cwd is often /repo; notebook may write under evaluation/ or repo/evaluation/
possible_bases = ["evaluation", "repo/evaluation", "../evaluation"]

SUBDIRS = ("checkpoints", "plots", "post_training")


def _upload_subdir(base: str, sub: str) -> None:
    folder = os.path.join(base, sub)
    if not os.path.isdir(folder):
        return
    print(f"Uploading {folder} -> evaluation/{sub} ...")
    api.upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        path_in_repo=f"evaluation/{sub}",
        repo_type="space",
    )


def main() -> None:
    print("Uploading training artifacts to Space...")
    try:
        for base in possible_bases:
            for sub in SUBDIRS:
                _upload_subdir(base, sub)
        print("Done.")
    except Exception as e:
        print(f"Error uploading: {e}")
        raise


if __name__ == "__main__":
    main()
