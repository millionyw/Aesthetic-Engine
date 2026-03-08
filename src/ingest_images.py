import argparse
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import numpy as np
import torch
from PIL import Image

from db import init_db, upsert_image
from feature_extractor import build_models, extract_hybrid_features


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def list_images(src: str):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    root = Path(src)
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]


def load_features(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_missing_features(
    image_paths,
    features,
    models,
    device,
    expected_dim=None,
    batch_size=16,
):
    missing = [
        p
        for p in image_paths
        if str(p.resolve()) not in features
        or (expected_dim is not None and features[str(p.resolve())].shape[-1] != expected_dim)
    ]
    if not missing:
        return
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch]
        hybrid = extract_hybrid_features(images, models, device).cpu()
        for path, vector in zip(batch, hybrid):
            features[str(path.resolve())] = vector


def load_predictor(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def ingest_images(src: str):
    features_path = "./data/features.pkl"
    model_path = "./data/models/aesthetic_predictor.pkl"
    init_db()

    image_paths = list_images(src)
    if len(image_paths) == 0:
        print("未找到可用图片")
        return

    device = get_device()
    models = build_models(device)

    rm_path = "./data/models/rm_predictor.pkl"
    ridge_path = "./data/models/aesthetic_predictor.pkl"
    
    rm_data = load_predictor(rm_path)
    is_rm = False
    
    if rm_data:
        print("使用 Reward Model (Logistic Regression)")
        predictor = rm_data["model"]
        scaler = rm_data["scaler"]
        raw_min = rm_data.get("raw_min", 0.0)
        raw_max = rm_data.get("raw_max", 1.0)
        is_rm = True
    else:
        ridge_data = load_predictor(ridge_path)
        if ridge_data:
            print("使用 Ridge Model")
            predictor = ridge_data["model"]
            scaler = ridge_data["scaler"]
        else:
            print("未找到训练模型")
            return

    expected_dim = getattr(predictor, "n_features_in_", None)

    features = load_features(features_path)
    extract_missing_features(
        image_paths,
        features,
        models,
        device,
        expected_dim=expected_dim,
    )
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    with open(features_path, "wb") as f:
        pickle.dump(features, f)

    valid = [(p, features[str(p.resolve())]) for p in image_paths if str(p.resolve()) in features]
    if len(valid) == 0:
        print("未找到可用特征")
        return

    paths, vectors = zip(*valid)
    vector_array = torch.stack(list(vectors), dim=0).cpu().numpy()
    vector_array = scaler.transform(vector_array)
    
    if is_rm:
        # Logistic Regression decision_function returns raw scores (dot product)
        raw_scores = predictor.decision_function(vector_array)
        # Normalize to 0-100
        if raw_max != raw_min:
            norm_scores = (raw_scores - raw_min) / (raw_max - raw_min) * 100
        else:
            norm_scores = raw_scores
        # Map 0-100 to 0.0-5.0
        scores = norm_scores / 20.0
        scores = np.clip(scores, 0.0, 5.0)
    else:
        scores = predictor.predict(vector_array)

    for path, score in zip(paths, scores):
        upsert_image(str(path.resolve()), float(score))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    ingest_images(args.src)
    print(f"✅ 入库总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
