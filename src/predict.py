import os
import pickle
import time

import numpy as np
import pandas as pd
import torch


def load_features(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    names = list(data.keys())
    vectors = torch.stack([data[name] for name in names], dim=0)
    return names, vectors.numpy()


def load_model(path: str):
    if not os.path.exists(path):
        print("未找到训练模型")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_labels_map(path: str):
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    if "filename" not in df.columns or "score" not in df.columns:
        return {}
    df = df[["filename", "score"]].dropna()
    df = df.drop_duplicates(subset=["filename"], keep="last")
    return dict(zip(df["filename"].astype(str), df["score"]))


def predict():
    start_time = time.time()
    features_path = "./data/features.pkl"
    labels_path = "./data/labels.csv"
    model_path = "./data/models/aesthetic_predictor.pkl"

    names, vectors = load_features(features_path)
    if len(names) == 0:
        print("未找到可用特征")
        return

    data = load_model(model_path)
    if data is None:
        return
    predictor = data["model"]
    scaler = data["scaler"]
    vectors = scaler.transform(vectors)
    scores = predictor.predict(vectors)
    label_map = load_labels_map(labels_path)
    top_k = min(10, len(names))
    indices = np.argsort(scores)[::-1][:top_k]
    for rank, idx in enumerate(indices, start=1):
        true_score = label_map.get(names[idx])
        true_display = true_score if true_score is not None else "-"
        print(f"{rank}. {names[idx]} (Pred: {scores[idx]:.4f} | True: {true_display})")
    print(f"✅ 预测总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    predict()
