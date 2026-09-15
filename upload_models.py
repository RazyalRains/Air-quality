"""
Run this ONCE, on your own machine, in the folder that has your .joblib files.
It pushes them to a Hugging Face model repo so app.py can download them at startup.

    pip install huggingface_hub
    huggingface-cli login          # paste a token with WRITE access when prompted
    python upload_models.py
"""

from huggingface_hub import HfApi, create_repo

REPO_ID = "Razyal/Air-Quality"

FILES = [
    "risk_model.joblib",
    "scaler.joblib",
    "kmeans_model.joblib",
    "similarity_index.joblib",
    "feature_cols.joblib",
    "reference_data.joblib",
]

# Creates the repo if it doesn't exist yet. Set private=True if you don't want it public.
create_repo(REPO_ID, repo_type="model", private=False, exist_ok=True)

api = HfApi()
for filename in FILES:
    print(f"Uploading {filename}...")
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=filename,
        repo_id=REPO_ID,
        repo_type="model",
    )

print("Done. Your models are at:", f"https://huggingface.co/{REPO_ID}")
