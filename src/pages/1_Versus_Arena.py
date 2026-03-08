import csv
import gc
import hashlib
import io
import os
import pickle
import random
import sys
import tempfile
import warnings
from datetime import datetime, timezone

import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_shortcuts import add_shortcuts

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=FutureWarning)

SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PAIRWISE_LABELS_PATH = "./data/pairwise_labels.csv"
FEATURES_PATH = "./data/features.pkl"
MODEL_PATH = "./data/models/aesthetic_predictor.pkl"


def load_features(features_path: str):
    if not os.path.exists(features_path):
        return [], np.zeros((0, 0), dtype=np.float32)
    with open(features_path, "rb") as f:
        data = pickle.load(f)
    names = list(data.keys())
    if not names:
        return [], np.zeros((0, 0), dtype=np.float32)
    vectors = np.stack([np.asarray(data[name], dtype=np.float32) for name in names], axis=0)
    return names, vectors


def load_predictor(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_pairwise_history(path: str):
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return set()
    history = set()
    for row in rows:
        winner = str(row.get("winner", ""))
        loser = str(row.get("loser", ""))
        if winner and loser:
            history.add(tuple(sorted([winner, loser])))
    return history


def save_pairwise_result(winner_filename: str, loser_filename: str):
    os.makedirs(os.path.dirname(PAIRWISE_LABELS_PATH), exist_ok=True)
    file_exists = os.path.exists(PAIRWISE_LABELS_PATH)
    with open(PAIRWISE_LABELS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["winner", "loser", "timestamp"])
        writer.writerow([winner_filename, loser_filename, datetime.now(timezone.utc).isoformat()])


def pick_pair(names, scores, history):
    if len(names) < 2:
        return None, None
    if scores is not None and scores.size == len(names):
        order = np.argsort(scores)
        sorted_names = [names[i] for i in order]
        sorted_scores = scores[order]
        for i in range(len(sorted_names) - 1):
            left = sorted_names[i]
            right = sorted_names[i + 1]
            if abs(float(sorted_scores[i + 1] - sorted_scores[i])) >= 0.2:
                continue
            key = tuple(sorted([left, right]))
            if key in history:
                continue
            return left, right
    candidates = [name for name in names]
    for _ in range(20):
        left, right = np.random.choice(candidates, size=2, replace=False)
        key = tuple(sorted([left, right]))
        if key not in history:
            return left, right
    if len(candidates) >= 2:
        left, right = np.random.choice(candidates, size=2, replace=False)
        return left, right
    return None, None


def ensure_pair(names, vectors):
    history = load_pairwise_history(PAIRWISE_LABELS_PATH)
    data = load_predictor(MODEL_PATH)
    scores = None
    if data is not None:
        model = data.get("model")
        scaler = data.get("scaler")
        if model is not None and scaler is not None and len(names) == vectors.shape[0]:
            vectors_scaled = scaler.transform(vectors)
            scores = model.predict(vectors_scaled)
    left, right = pick_pair(names, scores, history)
    return left, right


def clear_pair_state():
    for key in ["pair_left", "pair_right"]:
        if key in st.session_state:
            del st.session_state[key]


def load_ranking_engine():
    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)
    from ranking_engine import RankingEngine

    return RankingEngine


def load_reward_engine():
    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)
    from reward_engine import RewardModelEngine

    return RewardModelEngine


def load_export_engine():
    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)
    from export_engine import generate_leaderboard_pdf

    return generate_leaderboard_pdf


def load_analyzer():
    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)
    from analyzer import AestheticAnalyzer

    return AestheticAnalyzer


def load_visual_tracer():
    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)
    from visual_tracer import OcclusionExplainer

    return OcclusionExplainer


@st.cache_resource(show_spinner=False)
def load_global_models():
    """Load all models once and share them across components."""
    from feature_extractor import build_models, get_device
    device = get_device()
    models = build_models(device)
    return models, device


@st.cache_resource(show_spinner=False)
def load_occlusion_explainer(rm_mtime: float):
    models, device = load_global_models()
    OcclusionExplainer = load_visual_tracer()
    return OcclusionExplainer(
        models=models,
        rm_path="./data/models/rm_predictor.pkl",
        device=device,
        mask_fill="mean",
        batch_size=32,
        mode="fixed_crops",
    )


@st.cache_resource(show_spinner=False)
def load_clip_resources():
    models, device = load_global_models()
    clip_model, processor = models[0], models[1]
    return clip_model, processor, device


@st.cache_data(show_spinner=False)
def probe_top_concepts(rm_mtime: float):
    AestheticAnalyzer = load_analyzer()
    analyzer = AestheticAnalyzer()
    rm = analyzer.load_rm_weights()
    clip_model, clip_processor, clip_device = load_clip_resources()
    return analyzer.probe_concepts(
        rm.weights,
        model=clip_model,
        processor=clip_processor,
        device=clip_device,
        top_k=5,
    )


def get_hybrid_next_pair(
    valid_files,
    ignored_pairs,
    ranking_engine,
    reward_engine,
    reward_trained,
    predictor_scores,
):
    if not valid_files or len(valid_files) < 2:
        return None, None
    if reward_trained:
        mode_rand = random.random()
        if mode_rand < 0.3:
            pair = reward_engine.get_uncertain_pair(valid_files, ignored_pairs)
            if pair and pair[0] and pair[1]:
                return pair
            return ranking_engine.get_next_pair(
                valid_files,
                predictor_scores,
                ignored_pairs,
                mode_weights=(0.2, 0.2, 0.15, 0.15),
            )
        return ranking_engine.get_next_pair(
            valid_files,
            predictor_scores,
            ignored_pairs,
            mode_weights=(0.2, 0.2, 0.15, 0.15),
        )
    return ranking_engine.get_next_pair(
        valid_files,
        predictor_scores,
        ignored_pairs,
        mode_weights=(0.3, 0.3, 0.2, 0.2),
    )


@st.cache_data(show_spinner=False, max_entries=2000)
def get_thumbnail_bytes(image_path, size=(300, 300)):
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            return buf.getvalue()
    except Exception:
        return None


def build_trace_cache_key(path: str, rm_mtime: float, patch_size: int = 64, stride: int = 32):
    trace_seed = f"{path}|{rm_mtime}|{patch_size}|{stride}|mean|fixed_crops"
    return f"trace_overlay_{hashlib.md5(trace_seed.encode('utf-8')).hexdigest()}"


@st.dialog("🔍 视觉溯源", width="large")
def show_trace_dialog(image_path: str, overlay_png: bytes, base_score: float):
    cols = st.columns(2)
    with cols[0]:
        st.image(image_path, width="stretch")
        st.caption("原图")
    with cols[1]:
        st.image(overlay_png, width="stretch")
        st.caption(f"遮挡敏感度热力图 | base_score {base_score:.2f}")
    if st.button("关闭", key="close_trace_dialog"):
        st.session_state.trace_dialog_open = False
        st.session_state.trace_dialog_payload = None
        st.rerun()


def handle_trace_callback(path: str, rm_mtime: float):
    """Callback triggered IMMEDIATELY when trace button is clicked."""
    st.session_state.trace_dialog_open = True
    st.session_state.trace_pending_path = path
    st.session_state.trace_pending_mtime = rm_mtime


def load_pairwise_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return 0
    return len([row for row in rows if row.get("winner") and row.get("loser")])


def build_predictor_scores(names, vectors):
    data = load_predictor(MODEL_PATH)
    if data is None:
        return {}
    model = data.get("model")
    scaler = data.get("scaler")
    if model is None or scaler is None or len(names) != vectors.shape[0]:
        return {}
    vectors_scaled = scaler.transform(vectors)
    scores = model.predict(vectors_scaled)
    return {names[i]: float(scores[i]) for i in range(len(names))}


@st.cache_data(show_spinner=False)
def resolve_image_path(name: str):
    if not name:
        return None
    if os.path.exists(name):
        return name
    return None


def main():
    st.set_page_config(page_title="⚔️ 审美竞技场 (Versus Mode)", layout="wide")
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { padding: 2rem 1rem 1.5rem; }
        [data-testid="stHeader"] { height: 2.5rem; }
        </style>

        """,
        unsafe_allow_html=True,
    )

    if "trace_dialog_open" not in st.session_state:
        st.session_state.trace_dialog_open = False
    if "trace_dialog_payload" not in st.session_state:
        st.session_state.trace_dialog_payload = None
    if "trace_pending_path" not in st.session_state:
        st.session_state.trace_pending_path = None

    # Process pending trace calculation (if any)
    if st.session_state.trace_pending_path:
        path = st.session_state.trace_pending_path
        mtime = st.session_state.get("trace_pending_mtime", 0.0)
        trace_key = build_trace_cache_key(path, mtime)
        
        if trace_key in st.session_state:
            cache = st.session_state[trace_key]
            st.session_state.trace_dialog_payload = {
                "image_path": path,
                "overlay_png": cache["overlay_png"],
                "base_score": cache["base_score"],
            }
        else:
            with st.spinner(f"正在分析视觉贡献: {os.path.basename(path)}..."):
                try:
                    explainer = load_occlusion_explainer(float(mtime))
                    result = explainer.explain(path, patch_size=64, stride=32)
                    trace_data = {
                        "overlay_png": result.overlay_png,
                        "base_score": float(result.base_score),
                    }
                    st.session_state[trace_key] = trace_data
                    st.session_state.trace_dialog_payload = {
                        "image_path": path,
                        "overlay_png": result.overlay_png,
                        "base_score": float(result.base_score),
                    }
                    # Explicit cleanup after big task
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as e:
                    st.error(f"溯源失败: {e}")
        
        st.session_state.trace_pending_path = None # Clear pending
        st.session_state.trace_dialog_open = True
        st.rerun()

    # Top-level dialog trigger
    if st.session_state.trace_dialog_open:
        payload = st.session_state.trace_dialog_payload
        if isinstance(payload, dict):
            show_trace_dialog(
                image_path=payload.get("image_path", ""),
                overlay_png=payload.get("overlay_png", b""),
                base_score=float(payload.get("base_score", 0.0)),
            )
        else:
            st.session_state.trace_dialog_open = False

    debug_mode = st.session_state.get("pairwise_debug", False)

    names, vectors = load_features(FEATURES_PATH)
    resolved_names = []
    resolved_vectors = []
    resolved_paths = {}
    for i, name in enumerate(names):
        resolved_path = resolve_image_path(name)
        if resolved_path:
            resolved_names.append(name)
            resolved_vectors.append(vectors[i])
            resolved_paths[name] = resolved_path
    if debug_mode:
        with st.expander("调试信息", expanded=False):
            st.write(
                {
                    "cwd": os.getcwd(),
                    "features_path": os.path.abspath(FEATURES_PATH),
                    "features_exists": os.path.exists(FEATURES_PATH),
                    "model_path": os.path.abspath(MODEL_PATH),
                    "model_exists": os.path.exists(MODEL_PATH),
                }
            )
        with st.expander("特征加载状态", expanded=False):
            st.write(
                {
                    "total_features": len(names),
                    "resolved_features": len(resolved_names),
                    "missing_features": len(names) - len(resolved_names),
                    "vector_shape": list(vectors.shape),
                }
            )
    if len(resolved_names) < 2:
        st.write("未找到足够的图片用于对比")
        return

    vectors_filtered = (
        np.stack(resolved_vectors, axis=0) if resolved_vectors else np.zeros((0, 0), dtype=np.float32)
    )
    predictor_scores = build_predictor_scores(resolved_names, vectors_filtered)
    history_pairs = load_pairwise_history(PAIRWISE_LABELS_PATH)
    if "ignored_pairs" not in st.session_state:
        st.session_state.ignored_pairs = set()
    ignored_pairs = set(st.session_state.ignored_pairs)
    ignored_pairs.update(history_pairs)

    RankingEngine = load_ranking_engine()
    engine = RankingEngine(PAIRWISE_LABELS_PATH, FEATURES_PATH)
    engine.compute_elo()
    n_components, lcc_set = engine.get_connectivity_info(resolved_names)
    RewardModelEngine = load_reward_engine()
    reward_engine = RewardModelEngine(PAIRWISE_LABELS_PATH)
    features_dict = {
        name: vectors_filtered[idx] for idx, name in enumerate(resolved_names) if idx < len(resolved_names)
    }
    reward_trained = reward_engine.train(features_dict)
    if reward_trained:
        reward_engine.save_model()
    name_set = set(resolved_names)
    current_pair = st.session_state.get("current_pair")
    left = None
    right = None
    if current_pair and len(current_pair) == 2:
        left, right = current_pair
        pair_key = tuple(sorted([left, right]))
        if left == right or left not in name_set or right not in name_set or pair_key in ignored_pairs:
            left, right = None, None

    if left is None or right is None:
        left, right = get_hybrid_next_pair(
            resolved_names,
            ignored_pairs,
            engine,
            reward_engine,
            reward_trained,
            predictor_scores,
        )
        if left and right:
            st.session_state.current_pair = (left, right)

    left_path = resolved_paths.get(left)
    right_path = resolved_paths.get(right)
    if debug_mode:
        with st.expander("候选对信息", expanded=False):
            st.write(
                {
                    "left": left,
                    "right": right,
                    "left_path": left_path,
                    "right_path": right_path,
                    "left_exists": os.path.exists(left_path) if left_path else False,
                    "right_exists": os.path.exists(right_path) if right_path else False,
                }
            )
    if not left or not right or not left_path or not right_path:
        st.write("未找到可用的对比图片")
        return

    compare_count = load_pairwise_count(PAIRWISE_LABELS_PATH)
    total_count = len(resolved_names)
    connectivity = 0.0 if total_count == 0 else min(1.0, compare_count / (total_count * 5))
    summary_text = (
        f"已完成比对: {compare_count} 次 | 当前库容: {total_count} 张 | "
        f"图库连通性进度: {connectivity * 100:.1f}% | 独立群体: {n_components} 个"
    )

    main_col, side_col = st.columns([0.7, 0.3])

    with main_col:
        st.info(summary_text)
        st.title("⚔️ 审美竞技场 (Versus Mode)")
        col_left, col_right = st.columns(2)
        with col_left:
            st.image(Image.open(left_path).convert("RGB"), width="stretch")
            left_clicked = st.button("⬅️ 左边更好 (A)", width="stretch", key="left_better")

        with col_right:
            st.image(Image.open(right_path).convert("RGB"), width="stretch")
            right_clicked = st.button("右边更好 (D) ➡️", width="stretch", key="right_better")

        skip_cols = st.columns([3, 1, 3])
        with skip_cols[1]:
            skip_clicked = st.button("⏭️ 跳过 (S)", width="stretch", key="skip_pair")

        add_shortcuts(
            left_better=["arrowleft", "a"],
            right_better=["arrowright", "d"],
            skip_pair=["s"],
        )

        if left_clicked:
            save_pairwise_result(left, right)
            st.session_state.current_pair = None
            st.session_state.request_rerun = True
        if right_clicked:
            save_pairwise_result(right, left)
            st.session_state.current_pair = None
            st.session_state.request_rerun = True
        if skip_clicked:
            st.session_state.ignored_pairs.add(tuple(sorted([left, right])))
            st.session_state.current_pair = None
            st.session_state.request_rerun = True

    with side_col:
        st.subheader("审美巅峰榜")
        tab_leaderboard, tab_profile = st.tabs(["🏆 排行榜", "🧬 审美画像"])

        with tab_leaderboard:
            engine_options = ["图论胜绩直排 (Elo)", "特征泛化预测 (Reward Model)"]
            engine_choice = st.radio(
                "⚙️ 排名算法引擎",
                engine_options,
                horizontal=True,
                key="active_ranking_engine",
            )

            if engine_choice == "特征泛化预测 (Reward Model)":
                if not reward_trained:
                    st.write("数据不足，请继续打分")
                    leaderboard_all = []
                    leaderboard = []
                else:
                    leaderboard_all = reward_engine.get_leaderboard(top_n=len(resolved_names))
                    leaderboard_all = [item for item in leaderboard_all if item[0] in resolved_paths]
                    leaderboard = list(leaderboard_all)
                score_label = "RM"
            else:
                scores = engine.elo_scores
                match_counts = engine.match_counts
                show_only_matched = st.toggle(
                    "仅显示已比对次数 > 3 的图片",
                    value=st.session_state.get("filter_min_matches", True),
                    key="filter_min_matches",
                )
                leaderboard_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                leaderboard_all = [item for item in leaderboard_all if item[0] in resolved_paths]
                leaderboard = list(leaderboard_all)
                if show_only_matched:
                    leaderboard = [item for item in leaderboard if match_counts.get(item[0], 0) > 3]
                score_label = "Elo"

            if not leaderboard:
                st.write("暂无排行榜数据")
            else:
                if "leaderboard_page" not in st.session_state:
                    st.session_state.leaderboard_page = 0
                page_size = 10
                total_pages = max(1, (len(leaderboard) + page_size - 1) // page_size)
                page = max(0, min(st.session_state.leaderboard_page, total_pages - 1))
                st.session_state.leaderboard_page = page
                start = page * page_size
                end = (page + 1) * page_size
                page_items = leaderboard[start:end]

                grid = st.columns(2)
                for idx, (name, score) in enumerate(page_items):
                    path = resolved_paths.get(name)
                    if not path:
                        continue
                    sigma = engine.get_sigma(name)
                    if engine_choice == "图论胜绩直排 (Elo)":
                        badge = " 🌐" if name in lcc_set else ""
                    else:
                        badge = " 🧠"
                    thumb = get_thumbnail_bytes(path, size=(200, 200))
                    col = grid[idx % 2]
                    with col:
                        if thumb is None:
                            st.image(path, width="stretch")
                        else:
                            st.image(thumb, width="stretch")
                        st.caption(f"{score_label} {score:.1f} | σ {sigma:.3f}{badge}")

                nav_cols = st.columns(2)
                if nav_cols[0].button("上一页", disabled=page == 0):
                    st.session_state.leaderboard_page = page - 1
                    st.rerun()
                if nav_cols[1].button("查看更多", disabled=page >= total_pages - 1):
                    st.session_state.leaderboard_page = page + 1
                    st.rerun()

                export_key = f"export_pdf_bytes_{engine_choice}"
                top_items = leaderboard_all
                if export_key not in st.session_state:
                    st.session_state[export_key] = None
                if st.button("📄 导出审美排名 (PDF)"):
                    generate_leaderboard_pdf = load_export_engine()
                    with st.spinner("正在生成 PDF..."):
                        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                            output_path = tmp.name
                        generate_leaderboard_pdf(top_items, output_path, resolved_paths)
                        with open(output_path, "rb") as f:
                            st.session_state[export_key] = f.read()
                        os.remove(output_path)
                if st.session_state[export_key]:
                    suffix = "elo" if engine_choice == "图论胜绩直排 (Elo)" else "rm"
                    st.download_button(
                        "下载 PDF",
                        data=st.session_state[export_key],
                        file_name=f"aesthetic_leaderboard_{suffix}.pdf",
                        mime="application/pdf",
                    )

        with tab_profile:
            AestheticAnalyzer = load_analyzer()
            analyzer = AestheticAnalyzer()
            try:
                rm_mtime = os.path.getmtime(analyzer.rm_path) if os.path.exists(analyzer.rm_path) else 0.0
                rm = analyzer.load_rm_weights()
            except Exception as e:
                st.warning(f"审美画像暂不可用：{e}")
                rm = None

            if rm is not None:
                block = analyzer.block_weight_stats(rm.weights)
                chart_data = {
                    "占比": {
                        "环境": block.get("global_ratio", 0.0),
                        "身形": block.get("body_ratio", 0.0),
                        "面部": block.get("face_ratio", 0.0),
                    }
                }
                st.bar_chart(chart_data)
                st.caption(
                    f"关注度占比：环境 {block.get('global_ratio', 0.0) * 100:.1f}% | "
                    f"身形 {block.get('body_ratio', 0.0) * 100:.1f}% | "
                    f"面部 {block.get('face_ratio', 0.0) * 100:.1f}%"
                )

                try:
                    top_concepts = probe_top_concepts(float(rm_mtime))
                except Exception as e:
                    top_concepts = []
                    st.warning(f"语义标签探测失败：{e}")

                if top_concepts:
                    tags = " ".join([f"#{name.replace(' ', '')}" for name, _ in top_concepts])
                    st.write(f"你的审美偏好：{tags}")
                else:
                    st.write("你的审美偏好：暂无（RM 或 CLIP 尚未就绪）")

                best, worst = analyzer.fetch_extreme_samples(top_n=3)
                dims = analyzer.top_weight_dims(rm.weights, k=8)

                st.divider()
                st.subheader("极端样本对比")
                st.caption(f"这些图片之所以得分高/低，通常与 {analyzer.top_block(rm.weights)} 相关的高权重维度更一致/更背离。")
                if not best and not worst:
                    st.info("未在 gallery.db 中找到可用的 pred_score 样本，请先完成入库打分。")

                cols = st.columns(2)
                with cols[0]:
                    st.write("模型最爱")
                    for idx, (path, score) in enumerate(best):
                        thumb = get_thumbnail_bytes(path, size=(260, 260))
                        if thumb is None:
                            st.image(path, width="stretch")
                        else:
                            st.image(thumb, width="stretch")
                        vec = features_dict.get(path)
                        expl = analyzer.explain_image_by_dims(vec, rm.weights, dims) if vec is not None else "缺少特征缓存，无法解释"
                        st.caption(f"pred_score {score:.2f} | {expl}")
                        trace_key = build_trace_cache_key(path, float(rm_mtime), patch_size=64, stride=32)
                        st.button(
                            "🔍 溯源",
                            key=f"trace_btn_best_{idx}_{trace_key}",
                            on_click=handle_trace_callback,
                            args=(path, float(rm_mtime)),
                        )
                with cols[1]:
                    st.write("模型最恨")
                    for idx, (path, score) in enumerate(worst):
                        thumb = get_thumbnail_bytes(path, size=(260, 260))
                        if thumb is None:
                            st.image(path, width="stretch")
                        else:
                            st.image(thumb, width="stretch")
                        vec = features_dict.get(path)
                        expl = analyzer.explain_image_by_dims(vec, rm.weights, dims) if vec is not None else "缺少特征缓存，无法解释"
                        st.caption(f"pred_score {score:.2f} | {expl}")
                        trace_key = build_trace_cache_key(path, float(rm_mtime), patch_size=64, stride=32)
                        st.button(
                            "🔍 溯源",
                            key=f"trace_btn_worst_{idx}_{trace_key}",
                            on_click=handle_trace_callback,
                            args=(path, float(rm_mtime)),
                        )

    if st.session_state.get("request_rerun"):
        st.session_state.request_rerun = False
        st.rerun()


if __name__ == "__main__":
    main()
