import os
import re

FILES_TO_UPDATE = [
    "train.py",
    "train_v2.py",
    "upload_space.py",
    "upload_app.py",
    "push_dataset_jsonl.py",
    "collect_post_training.py"
]

def main():
    print("="*50)
    print(" 🔄 Hugging Face Workspace Switcher 🔄")
    print("="*50)
    print("\nThis script will update all Hugging Face links, datasets, and space references in your codebase to point to the provided user's workspace.\n")
    
    target_user = input("Enter target HF Username (e.g. rsd-06 or SaiSanjayR): ").strip()
    if not target_user:
        print("Username cannot be empty. Exiting.")
        return
        
    target_space = input("Enter target HF Space Name (e.g. PRRegressionAuditEnv or GitPRTriage_Environment): ").strip()
    if not target_space:
        print("Space Name cannot be empty. Exiting.")
        return

    print(f"\nSwitching workspace to -> User: {target_user} | Space: {target_space}\n")

    for file in FILES_TO_UPDATE:
        if not os.path.exists(file):
            continue
            
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            
        original_content = content
        
        # 1. Update Space URLs (e.g. Ground Truth URL)
        content = re.sub(
            r'https://huggingface\.co/spaces/[^/]+/[^/]+/resolve',
            f'https://huggingface.co/spaces/{target_user}/{target_space}/resolve',
            content
        )
        
        # 2. Update SPACE_ID variables
        content = re.sub(
            r'SPACE_ID\s*=\s*["\'][^/]+/[^/]+["\']',
            f'SPACE_ID = "{target_user}/{target_space}"',
            content
        )
        
        # 3. Update Dataset references
        content = re.sub(
            r'["\'][^/]+/pr-regression-audit-grpo["\']',
            f'"{target_user}/pr-regression-audit-grpo"',
            content
        )
        
        # 4. Update V2 Output Adapter references
        content = re.sub(
            r'["\'][^/]+/pr-regression-audit-grpo-adapter-v2["\']',
            f'"{target_user}/pr-regression-audit-grpo-adapter-v2"',
            content
        )

        # Note: We INTENTIONALLY skip updating V1_ADAPTER references (like SaiSanjayR/pr-triage-grpo-adapter) 
        # so that everyone continues to use the same trained baseline as a starting point.

        # 5. Update direct upload_app / upload_results repo_id args
        content = re.sub(
            r'repo_id=["\'][^/]+/[^/]+["\']',
            f'repo_id="{target_user}/{target_space}"',
            content
        )

        if content != original_content:
            with open(file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Updated: {file}")
        else:
            print(f"➖ No changes needed in: {file}")

    print("\n🎉 Workspace switch complete!")

if __name__ == "__main__":
    main()
