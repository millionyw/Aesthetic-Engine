import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def train():
    start_time = time.time()
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
