import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import torch
from PIL import Image
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, extensions=None):
        self.root_dir = root_dir
        if extensions is None:
            extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.paths = []
        root_path = Path(root_dir)
        if root_path.exists():
            for path in root_path.rglob("*"):
                if path.is_file() and path.suffix.lower() in extensions:
                    self.paths.append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        return path.name, image


def collate_fn(batch):
    names, images = zip(*batch)
    return list(names), list(images)


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
        local_dir_use_symlinks=False,
        cache_dir=local_dir,
        resume_download=True,
    )
    return local_dir


def extract_features(
    input_dir: str,
    output_path: str,
    batch_size: int,
):
    device = get_device()
    model_name = "openai/clip-vit-large-patch14"
    local_model_dir = ensure_model_local(model_name, "./data/models/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained(local_model_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(local_model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    dataset = ImageFolderDataset(input_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    features = {}
    for names, images in tqdm(loader, desc="Extracting", unit="batch"):
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
        for name, vector in zip(names, image_features):
            features[name] = vector.detach().cpu()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    return len(features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./data/raw_images")
    parser.add_argument("--output_path", default="./data/features.pkl")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    count = extract_features(
        input_dir=args.input_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
    )
    print(f"Extracted {count} embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
