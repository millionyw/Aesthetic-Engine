import argparse
import csv
import os
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from db import fetch_images


def load_features(path: str):
    with open(path, "rb") as f:
        data = pickle.load(f)
    names = list(data.keys())
    vectors = torch.stack([data[name] for name in names], dim=0)
    return names, vectors.numpy()


def load_labels(path: str):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["filename", "score"])
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["filename", "score"])
    if "filename" not in df.columns or "score" not in df.columns:
        return pd.DataFrame(columns=["filename", "score"])
    df = df[["filename", "score"]].dropna()
    df = df.drop_duplicates(subset=["filename"], keep="last")
    return df


def spearman_corr(a: np.ndarray, b: np.ndarray):
    if a.size < 2 or b.size < 2:
        return float("nan")
    a_rank = pd.Series(a).rank(method="average").to_numpy()
    b_rank = pd.Series(b).rank(method="average").to_numpy()
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def sync_human_scores():
    """从数据库同步人类打分到 labels.csv"""
    labels_path = "./data/labels.csv"
    
    # 获取 DB 中有人类打分的数据
    images = fetch_images()
    human_scored = {img['path']: img['human_score'] for img in images if img['human_score'] is not None}
    
    if not human_scored:
        return

    # 读取现有的 labels.csv
    current_labels = {}
    if os.path.exists(labels_path):
        try:
            df = pd.read_csv(labels_path)
            if "filename" in df.columns and "score" in df.columns:
                for _, row in df.iterrows():
                    current_labels[str(row['filename'])] = float(row['score'])
        except Exception:
            pass
            
    # 找出差异并追加
    updates = []
    for path, score in human_scored.items():
        # 如果路径不在 csv 中，或者分数不同，则更新（追加）
        if path not in current_labels or abs(current_labels[path] - score) > 0.1:
            updates.append([path, score])
            
    if updates:
        print(f"同步 {len(updates)} 条人类打分到 {labels_path}")
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        file_exists = os.path.exists(labels_path)
        with open(labels_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["filename", "score", "timestamp"])
            
            now = datetime.now(timezone.utc).isoformat()
            for path, score in updates:
                writer.writerow([path, score, now])


def train():
    start_time = time.time()
    sync_human_scores()
    
    features_path = "./data/features.pkl"
    labels_path = "./data/labels.csv"
    model_path = "./data/models/aesthetic_predictor.pkl"

    names, vectors = load_features(features_path)
    labels_df = load_labels(labels_path)
    if labels_df.empty:
        print("未找到可用标注数据")
        return

    label_map = dict(zip(labels_df["filename"].astype(str), labels_df["score"].astype(float)))
    name_to_index = {name: idx for idx, name in enumerate(names)}
    selected = [name for name in names if name in label_map]
    if len(selected) < 2:
        print("可训练样本不足")
        return
    if len(selected) < 100:
        print("Warning: 样本量过少，建议标注至少 100 张图片以防止过拟合。")

    indices = [name_to_index[name] for name in selected]
    x = vectors[indices]
    y = np.array([label_map[name] for name in selected], dtype=np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 50.0, 100.0, 200.0])
    model.fit(x_train_scaled, y_train)
    print(f"Best alpha selected: {model.alpha_}")

    preds = model.predict(x_test_scaled)
    mse = mean_squared_error(y_test, preds)
    spearman = spearman_corr(y_test, preds)
    print(f"MSE: {mse:.4f}")
    print(f"Spearman: {spearman:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print(f"✅ 训练总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    train()
