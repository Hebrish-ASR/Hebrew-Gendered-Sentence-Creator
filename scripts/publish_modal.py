"""Publish a trained run to the Hebrish org on Hugging Face (public), copying inside Modal so the
16 GB of weights never pass through the laptop.

    MODAL_PROFILE=danielrosehill with-secrets modal run --detach scripts/publish_modal.py --run gemma-4-e4b-v3

Model:   danielrosehill/hebrew-gendered-sentence-creator-<run> (private) -> Hebrish/Hebrew-Gendered-Sentence-Creator,
         with hf/MODEL_CARD.md as README.md.
Dataset: data/train.jsonl, data/eval.jsonl, data/build.json and data/terms/terms.jsonl ->
         Hebrish/Hebrew-Gendered-Sentences, with hf/DATASET_CARD.md as README.md.
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO, DATA_REPO = "Hebrish/Hebrew-Gendered-Sentence-Creator", "Hebrish/Hebrew-Gendered-Sentences"
image = (modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub[hf_xet]")
         .add_local_file(ROOT / "hf/MODEL_CARD.md", "/pub/model/README.md")
         .add_local_file(ROOT / "hf/DATASET_CARD.md", "/pub/data/README.md")
         .add_local_file(ROOT / "data/train.jsonl", "/pub/data/train.jsonl")
         .add_local_file(ROOT / "data/eval.jsonl", "/pub/data/eval.jsonl")
         .add_local_file(ROOT / "data/build.json", "/pub/data/build.json")
         .add_local_file(ROOT / "data/terms/terms.jsonl", "/pub/data/terms.jsonl"))
app = modal.App("hebrew-gendered-publish", image=image)
hf = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]} if modal.is_local() else {})


@app.function(timeout=3 * 3600, secrets=[hf])
def publish(run: str):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    src = f"danielrosehill/hebrew-gendered-sentence-creator-{run}"
    local = snapshot_download(src, local_dir="/tmp/model")
    Path(local, "README.md").write_text(Path("/pub/model/README.md").read_text())
    api.create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=MODEL_REPO, folder_path=local, commit_message=f"{run} from {src}")
    api.create_repo(DATA_REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(repo_id=DATA_REPO, repo_type="dataset", folder_path="/pub/data", commit_message="Training data")
    return [MODEL_REPO, DATA_REPO]


@app.local_entrypoint()
def main(run: str = "gemma-4-e4b-v3"):
    print(publish.remote(run))
