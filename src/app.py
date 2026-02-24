import csv
import io
import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_shortcuts import add_shortcuts

import ingest_images as ingest_module
import train as train_module
from db import fetch_images, init_db, update_human_score
from feature_extractor import build_models, extract_hybrid_features, get_device

def load_features(features_path: str):
    with open(features_path, "rb") as f:
        return pickle.load(f)


def load_labeled(labels_path: str):
    if not os.path.exists(labels_path):
        return set()
    try:
        df = pd.read_csv(labels_path)
    except pd.errors.EmptyDataError:
        return set()
    if "filename" not in df.columns:
        return set()
    return set(df["filename"].astype(str).tolist())


def append_label(labels_path: str, filename: str, score: int):
    os.makedirs(os.path.dirname(labels_path), exist_ok=True)
    file_exists = os.path.exists(labels_path)
    with open(labels_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "score", "timestamp"])
        writer.writerow([filename, score, datetime.utcnow().isoformat()])


def score_and_advance(labels_path: str, filename: str, score: int):
    if score is None:
        return
    append_label(labels_path, filename, score)
    st.rerun()


def get_grouped_labels(labels_path: str):
    if not os.path.exists(labels_path):
        return {}
    try:
        df = pd.read_csv(labels_path)
    except pd.errors.EmptyDataError:
        return {}
    if "filename" not in df.columns or "score" not in df.columns:
        return {}
    df = df[["filename", "score"]].dropna()
    df = df.drop_duplicates(subset=["filename"], keep="last")
    grouped = {}
    for score in [5, 4, 3, 2, 1]:
        items = df[df["score"] == score]["filename"].astype(str).tolist()
        if items:
            grouped[int(score)] = items
    return grouped


def truncate_text(text: str, max_len: int):
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def clear_dialog_state():
    for key in ["dialog_open", "dialog_list", "dialog_idx", "dialog_nav"]:
        if key in st.session_state:
            del st.session_state[key]


@st.cache_data(show_spinner=False, max_entries=500)
def get_thumbnail_bytes(image_path, size=(300, 300)):
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()
    except Exception:
        return None


def load_feature_cache(features_path: str):
    if not os.path.exists(features_path):
        return {}
    with open(features_path, "rb") as f:
        return pickle.load(f)


def load_predictor(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_models():
    device = get_device()
    models = build_models(device)
    return device, models


def get_feature_vector(image_path: str, features, device, models):
    cache = st.session_state.setdefault("feature_cache", {})
    if image_path in cache:
        return cache[image_path]
    name = os.path.basename(image_path)
    if name in features:
        vector = features[name]
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        cache[image_path] = vector
        return vector
    image = Image.open(image_path).convert("RGB")
    hybrid = extract_hybrid_features([image], models, device).cpu().numpy()[0]
    cache[image_path] = hybrid
    return hybrid


def compute_contributions(vector: np.ndarray, predictor, scaler):
    vector = np.asarray(vector, dtype=np.float32)
    scaled = scaler.transform(vector.reshape(1, -1))
    weights = np.asarray(predictor.coef_, dtype=np.float32)
    contrib = np.abs(scaled * weights)[0]
    global_avg = float(np.sum(contrib[0:768]) / 768)
    body_avg = float(np.sum(contrib[768:1536]) / 768)
    face_avg = float(np.sum(contrib[1536:2048]) / 512)
    total = global_avg + body_avg + face_avg
    if total == 0:
        perc = [0.0, 0.0, 0.0]
    else:
        perc = [global_avg / total * 100, body_avg / total * 100, face_avg / total * 100]
    labels = ["Global", "Body", "Face"]
    dominant = labels[int(np.argmax([global_avg, body_avg, face_avg]))]
    dominant_text = {
        "Global": "🔥 气质主导",
        "Body": "💃 体态主导",
        "Face": "✨ 五官主导",
    }[dominant]
    text = (
        f"{dominant_text} "
        f"(Global: {perc[0]:.0f}%, Body: {perc[1]:.0f}%, Face: {perc[2]:.0f}%)"
    )
    return text


@st.dialog("图片详情", width="large")
def preview_dialog(image_list, initial_index, labels_path, model_path, features_path):
    st.session_state.dialog_open = True
    if "dialog_list" not in st.session_state:
        st.session_state.dialog_list = image_list
    if "dialog_idx" not in st.session_state:
        st.session_state.dialog_idx = initial_index
    items = st.session_state.dialog_list
    idx = st.session_state.dialog_idx
    idx = max(0, min(idx, len(items) - 1))
    row = items[idx]
    path = row["path"]
    if os.path.exists(path):
        st.image(path, use_container_width=True)
    name = os.path.basename(path)
    pred_score = row["pred_score"]
    human_score = row["human_score"]
    pred_display = f"{pred_score:.4f}" if pred_score is not None else "-"
    human_display = human_score if human_score is not None else "-"
    st.write(f"{name}")
    st.write(f"Pred: {pred_display} | Human: {human_display}")

    data = load_predictor(model_path)
    if data is not None:
        predictor = data["model"]
        scaler = data["scaler"]
        features = load_feature_cache(features_path)
        try:
            device, models = load_models()
            vector = get_feature_vector(path, features, device, models)
            text = compute_contributions(vector, predictor, scaler)
            st.write(text)
        except Exception:
            st.write("解释信息生成失败")

    cols = st.columns(5)
    for value, col_btn in enumerate(cols, start=1):
        key = f"dialog_feedback_{value}_{path}"
        if col_btn.button(str(value), key=key):
            update_human_score(path, value)
            append_label(labels_path, os.path.basename(path), value)
            with st.spinner("模型增量训练中..."):
                start_time = time.time()
                train_module.train()
                st.success(f"✅ 训练总耗时: {time.time() - start_time:.2f} 秒")
            st.toast("模型已根据反馈自动进化")
            st.rerun()

    nav_cols = st.columns(2)
    if nav_cols[0].button("⬅️ 上一张", key="dialog_prev_btn"):
        st.session_state.dialog_idx = max(0, idx - 1)
        st.rerun()
    if nav_cols[1].button("下一张 ➡️", key="dialog_next_btn"):
        st.session_state.dialog_idx = min(len(items) - 1, idx + 1)
        st.rerun()

    add_shortcuts(
        dialog_prev_btn=["arrowleft", "a"],
        dialog_next_btn=["arrowright", "d"],
        dialog_close_btn="escape",
        **{
            f"dialog_feedback_{score}_{path}": str(score)
            for score in range(1, 6)
        },
    )

    if st.button("关闭预览", key="dialog_close_btn"):
        clear_dialog_state()
        st.rerun()


def main():
    if get_script_run_ctx() is None:
        print("请使用以下命令启动：streamlit run .\\src\\app.py")
        return
    st.set_page_config(page_title="Aesthetic Labeling", layout="centered")
    features_path = "./data/features.pkl"
    labels_path = "./data/labels.csv"
    model_path = "./data/models/aesthetic_predictor.pkl"
    image_root = "./data/raw_images"
    init_db()
    tab_label, tab_gallery, tab_control = st.tabs(["🏷️ 标注大厅", "🗂️ 审美图库", "⚙️ 模型控制台"])

    with tab_label:
        features = load_features(features_path)
        filenames = list(features.keys())
        labeled = load_labeled(labels_path)
        remaining = [
            name
            for name in filenames
            if name not in labeled and os.path.exists(os.path.join(image_root, name))
        ]

        use_active = False
        if "use_active" not in st.session_state:
            st.session_state.use_active = False
        use_active = st.session_state.use_active

        ordered = remaining
        if use_active and os.path.exists(model_path) and len(remaining) > 0:
            data = load_predictor(model_path)
            if data is not None:
                predictor = data["model"]
                scaler = data["scaler"]
                vectors = torch.stack([features[name] for name in remaining], dim=0)
                vector_array = scaler.transform(vectors.cpu().numpy())
                scores = predictor.predict(vector_array)
                order = np.argsort(scores)[::-1]
                ordered = [remaining[i] for i in order]

        col_main, col_right = st.columns([7, 3])

        with col_main:
            current = None
            if len(ordered) == 0:
                st.write("所有图片已标注")
            else:
                current = ordered[0]
                image_path = os.path.join(image_root, current)
                st.image(image_path, width=450)

                cols = st.columns(5)
                for score, col in enumerate(cols, start=1):
                    if col.button(str(score), use_container_width=True, key=f"label_score_{score}"):
                        score_and_advance(labels_path, current, score)
                add_shortcuts(
                    **{f"label_score_{score}": str(score) for score in range(1, 6)}
                )

        with col_right:
            st.checkbox("开启主动学习模式", key="use_active")
            st.subheader("训练数据")
            grouped = get_grouped_labels(labels_path)
            if not grouped:
                st.write("暂无标注数据")
            else:
                for score in [5, 4, 3, 2, 1]:
                    items = grouped.get(score, [])
                    if not items:
                        continue
                    with st.expander(f"⭐ {score} 分 (共 {len(items)} 张)", expanded=False):
                        for name in items:
                            image_path = os.path.join(image_root, name)
                            if os.path.exists(image_path):
                                st.image(image_path, width="stretch")

    with tab_control:
        st.subheader("模型控制台")
        if st.button("触发模型训练"):
            start_time = time.time()
            train_module.train()
            st.success(f"✅ 训练总耗时: {time.time() - start_time:.2f} 秒")

        target_dir = st.text_input("待扫描目录")
        if st.button("扫描并预测"):
            if not target_dir:
                st.write("请输入有效目录")
            else:
                start_time = time.time()
                ingest_module.ingest_images(target_dir)
                st.success(f"✅ 入库总耗时: {time.time() - start_time:.2f} 秒")

    with tab_gallery:
        st.subheader("审美图库")
        st.markdown(
            """
            <style>
            .gallery-meta {
                background: #0f1117;
                border: 1px solid #2a2f3a;
                border-radius: 10px;
                padding: 8px 10px;
                margin-top: 6px;
                margin-bottom: 6px;
            }
            .gallery-name {
                font-size: 12px;
                font-weight: 600;
                color: #e6e6e6;
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .gallery-score {
                font-size: 11px;
                color: #b7bdc9;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        rows = fetch_images()
        if len(rows) == 0:
            st.write("图库暂无数据")
        else:
            if "dialog_open" not in st.session_state:
                st.session_state.dialog_open = False
            st.session_state.dialog_rendered = False
            data = load_predictor(model_path)
            predictor = data["model"] if data is not None else None
            scaler = data["scaler"] if data is not None else None
            features = load_feature_cache(features_path)
            groups = {score: [] for score in [5, 4, 3, 2, 1]}
            for row in rows:
                score_source = (
                    row["human_score"] if row["human_score"] is not None else row["pred_score"]
                )
                if score_source is None:
                    continue
                score_value = int(round(score_source))
                score_value = max(1, min(5, score_value))
                groups[score_value].append(row)

            for score in [5, 4, 3, 2, 1]:
                items = groups[score]
                if not items:
                    continue
                limit_key = f"gallery_limit_{score}"
                if limit_key not in st.session_state:
                    st.session_state[limit_key] = 12
                with st.expander(f"⭐ {score} 分 (共 {len(items)} 张)", expanded=False):
                    shown = items[: st.session_state[limit_key]]
                    grid = st.columns(4)
                    for idx, row in enumerate(shown):
                        path = row["path"]
                        if not os.path.exists(path):
                            continue
                        col = grid[idx % 4]
                        with col:
                            thumb = get_thumbnail_bytes(path)
                            if thumb is None:
                                st.image(path, use_container_width=True)
                            else:
                                st.image(thumb, use_container_width=True)
                            pred_score = row["pred_score"]
                            human_score = row["human_score"]
                            pred_display = f"{pred_score:.4f}" if pred_score is not None else "-"
                            human_display = human_score if human_score is not None else "-"
                            name = truncate_text(os.path.basename(path), 24)
                            st.markdown(
                                f"""
                                <div class="gallery-meta">
                                    <div class="gallery-name">{name}</div>
                                    <div class="gallery-score">Human: {human_display} | Pred: {pred_display}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            detail_key = f"detail_{path}"
                            if st.button("🔍 详情", key=detail_key):
                                st.session_state.dialog_list = shown
                                st.session_state.dialog_idx = idx
                                st.session_state.dialog_open = True
                                st.session_state.dialog_rendered = True
                                preview_dialog(
                                    st.session_state.dialog_list,
                                    st.session_state.dialog_idx,
                                    labels_path,
                                    model_path,
                                    features_path,
                                )

                    if len(items) > st.session_state[limit_key]:
                        if st.button(f"加载更多 {score} 分", key=f"more_{score}"):
                            st.session_state[limit_key] += 12
                            st.rerun()
                    if st.session_state.get("dialog_open"):
                        st.session_state.dialog_rendered = True
                        preview_dialog(
                            st.session_state.dialog_list,
                            st.session_state.dialog_idx,
                            labels_path,
                            model_path,
                            features_path,
                        )
                    if st.session_state.get("dialog_open") and not st.session_state.get(
                        "dialog_rendered"
                    ):
                        clear_dialog_state()
                    if not st.session_state.get("dialog_open") and "dialog_list" in st.session_state:
                        clear_dialog_state()


if __name__ == "__main__":
    main()
