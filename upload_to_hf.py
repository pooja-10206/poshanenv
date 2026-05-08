"""
Upload PoshanEnv to HuggingFace Spaces.
Run: python upload_to_hf.py
"""
from huggingface_hub import HfApi
import os

# ── change this to your HF username ──
HF_USERNAME = "pooja10206"
REPO_ID = f"{HF_USERNAME}/poshanenv"

api = HfApi()

# Step 1: Create the Space
print("Creating HuggingFace Space...")
try:
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False,
    )
    print(f"Space created: https://huggingface.co/spaces/{REPO_ID}")
except Exception as e:
    print(f"Space may already exist: {e}")

# Step 2: Upload server/ folder contents
print("\nUploading server files...")
api.upload_folder(
    folder_path=r"C:\Users\DELL\poshanenv\server",
    repo_id=REPO_ID,
    repo_type="space",
    path_in_repo=".",
)
print("Server files uploaded!")

# Step 3: Upload root files
print("\nUploading root files...")
root = r"C:\Users\DELL\poshanenv"
for filename in ["inference.py", "requirements.txt", "openenv.yaml"]:
    filepath = os.path.join(root, filename)
    if os.path.exists(filepath):
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="space",
        )
        print(f"  Uploaded {filename}")
    else:
        print(f"  MISSING: {filename} — check it exists in poshanenv/")

print(f"\nDone! Your Space: https://huggingface.co/spaces/{REPO_ID}")
print("Wait 2-3 minutes for it to build, then check the URL.")
