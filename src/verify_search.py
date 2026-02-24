import argparse
import os
import pickle

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import torch
from huggingface_hub import snapshot_download
from transformers import CLIPModel, CLIPProcessor


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_model_local(model_name: str, local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        cache_dir=local_dir,
    )
    return local_dir


def load_features(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    names = list(data.keys())
    vectors = torch.stack([data[name] for name in names], dim=0)
    return names, vectors


def search(query: str, features_path: str, top_k: int):
    device = get_device()
    model_name = "openai/clip-vit-large-patch14"
    local_model_dir = ensure_model_local(model_name, "./data/models/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained(local_model_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(local_model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    inputs = processor(text=[query], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        if not isinstance(text_features, torch.Tensor):
            if hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
                text_features = text_features.text_embeds
            elif hasattr(text_features, "pooler_output"):
                text_features = model.text_projection(text_features.pooler_output)
    text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)
    names, image_vectors = load_features(features_path)
    image_vectors = image_vectors[:, :768]
    image_vectors = torch.nn.functional.normalize(image_vectors, p=2, dim=-1).to(device)

    scores = (text_features @ image_vectors.T).squeeze(0)
    top_k = min(top_k, scores.shape[0])
    values, indices = torch.topk(scores, k=top_k)
    for rank, (score, idx) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {names[idx]} ({score:.4f})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--features_path", default="./data/features.pkl")
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    search(args.query, args.features_path, args.top_k)


if __name__ == "__main__":
    main()
