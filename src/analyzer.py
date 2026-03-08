import os
import pickle
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
import transformers
try:
    from transformers import AutoModel
except ImportError:
    try:
        from transformers.models.auto import AutoModel
    except ImportError:
        from transformers import CLIPModel as AutoModel

try:
    from transformers import AutoProcessor
except ImportError:
    try:
        from transformers.models.auto import AutoProcessor
    except ImportError:
        from transformers import CLIPProcessor as AutoProcessor


@dataclass(frozen=True)
class RMWeights:
    weights: np.ndarray
    timestamp: Optional[str]
    raw_min: Optional[float]
    raw_max: Optional[float]


class AestheticAnalyzer:
    GLOBAL_DIM = 768
    BODY_DIM = 768
    FACE_DIM = 512
    TOTAL_DIM = GLOBAL_DIM + BODY_DIM + FACE_DIM

    def __init__(
        self,
        rm_path: str = "./data/models/rm_predictor.pkl",
        db_path: str = "./data/gallery.db",
        clip_repo_id: str = "openai/clip-vit-large-patch14",
        clip_local_dir: str = "./data/models/clip-vit-large-patch14",
        concepts: Optional[Sequence[str]] = None,
    ):
        self.rm_path = rm_path
        self.db_path = db_path
        self.clip_repo_id = clip_repo_id
        self.clip_local_dir = clip_local_dir
        self.concepts = list(concepts) if concepts is not None else self.default_concepts()

    @staticmethod
    def default_concepts() -> List[str]:
        return [
            "Minimalist",
            "Cinematic",
            "Vivid Colors",
            "Soft Lighting",
            "Symmetry",
            "Cyberpunk",
            "High Contrast",
            "Low Key Lighting",
            "Pastel",
            "Golden Hour",
            "Street Photography",
            "Film Grain",
            "Clean Background",
            "Moody",
            "Vintage",
            "Neon",
            "Soft Dick",
            "Hard Dick",
            "bear",
            "络腮胡",
            "Mutton chops",
            "Sideburns",
            "Full Beard",
            "Whiskers",
            "Cyberpunk",
            "Muscle covered by fat",
            "High muscle mass with higher body fat",
            "Bulky body",
            "Soft gains",
            "fat-over-muscle"
        ]

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _safe_normalize(tensor: torch.Tensor) -> torch.Tensor:
        norms = tensor.norm(p=2, dim=-1, keepdim=True)
        return tensor / torch.where(norms == 0, torch.ones_like(norms), norms)

    def load_rm_weights(self) -> RMWeights:
        if not os.path.exists(self.rm_path):
            raise FileNotFoundError(f"未找到 RM 模型文件: {os.path.abspath(self.rm_path)}")
        with open(self.rm_path, "rb") as f:
            data = pickle.load(f)
        model = data.get("model")
        if model is None or not hasattr(model, "coef_"):
            raise ValueError("RM 模型文件缺少可解析的线性权重 (model.coef_)")
        coef = np.asarray(model.coef_[0], dtype=np.float32)
        if coef.ndim != 1 or coef.shape[0] != self.TOTAL_DIM:
            raise ValueError(f"RM 权重维度异常: {list(coef.shape)}")
        return RMWeights(
            weights=coef,
            timestamp=data.get("timestamp"),
            raw_min=data.get("raw_min"),
            raw_max=data.get("raw_max"),
        )

    def block_weight_stats(self, weights: np.ndarray) -> Dict[str, float]:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        g = np.mean(np.abs(weights[: self.GLOBAL_DIM]))
        b = np.mean(np.abs(weights[self.GLOBAL_DIM : self.GLOBAL_DIM + self.BODY_DIM]))
        f = np.mean(np.abs(weights[self.GLOBAL_DIM + self.BODY_DIM :]))
        total = float(g + b + f)
        if total <= 0:
            return {
                "global_mean_abs": float(g),
                "body_mean_abs": float(b),
                "face_mean_abs": float(f),
                "global_ratio": 0.0,
                "body_ratio": 0.0,
                "face_ratio": 0.0,
            }
        return {
            "global_mean_abs": float(g),
            "body_mean_abs": float(b),
            "face_mean_abs": float(f),
            "global_ratio": float(g / total),
            "body_ratio": float(b / total),
            "face_ratio": float(f / total),
        }

    def _ensure_clip_local(self) -> str:
        os.makedirs(self.clip_local_dir, exist_ok=True)
        snapshot_download(
            repo_id=self.clip_repo_id,
            local_dir=self.clip_local_dir,
            cache_dir=self.clip_local_dir,
        )
        return self.clip_local_dir

    def build_clip(self, device: Optional[torch.device] = None) -> Tuple[Any, Any, torch.device]:
        if device is None:
            device = self.get_device()
        local_dir = self._ensure_clip_local()
        processor = AutoProcessor.from_pretrained(local_dir, local_files_only=True)
        model = AutoModel.from_pretrained(local_dir, local_files_only=True)
        model.to(device)
        model.eval()
        return model, processor, device

    def clip_text_embeddings(
        self,
        texts: Sequence[str],
        model: Any,
        processor: Any,
        device: torch.device,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.GLOBAL_DIM), dtype=np.float32)
        inputs = processor(text=list(texts), return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats: Optional[torch.Tensor] = None
            if hasattr(model, "get_text_features"):
                out = model.get_text_features(**inputs)
                if isinstance(out, torch.Tensor):
                    feats = out
                else:
                    if hasattr(out, "text_embeds") and isinstance(getattr(out, "text_embeds"), torch.Tensor):
                        feats = getattr(out, "text_embeds")
                    elif hasattr(out, "pooler_output") and isinstance(getattr(out, "pooler_output"), torch.Tensor):
                        pooled = getattr(out, "pooler_output")
                        if hasattr(model, "text_projection") and pooled.shape[-1] == model.text_projection.in_features:
                            feats = model.text_projection(pooled)
                        else:
                            feats = pooled
            if feats is None:
                text_outputs = model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                pooled = text_outputs.pooler_output
                if hasattr(model, "text_projection") and pooled.shape[-1] == model.text_projection.in_features:
                    feats = model.text_projection(pooled)
                else:
                    feats = pooled
        feats = self._safe_normalize(feats)
        feats = feats.detach().cpu().numpy().astype(np.float32)
        if feats.ndim != 2 or feats.shape[1] != self.GLOBAL_DIM:
            raise ValueError(f"CLIP 文本特征维度异常: {list(feats.shape)}")
        return feats

    def _expand_text_to_hybrid(self, text_768: np.ndarray) -> np.ndarray:
        text_768 = np.asarray(text_768, dtype=np.float32).reshape(-1)
        if text_768.shape[0] != self.GLOBAL_DIM:
            raise ValueError(f"text embedding 维度异常: {list(text_768.shape)}")
        face_zeros = np.zeros((self.FACE_DIM,), dtype=np.float32)
        hybrid = np.concatenate([text_768, text_768, face_zeros], axis=0)
        return hybrid

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def probe_concepts(
        self,
        weights: np.ndarray,
        model: Any,
        processor: Any,
        device: torch.device,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if weights.shape[0] != self.TOTAL_DIM:
            raise ValueError(f"RM 权重维度异常: {list(weights.shape)}")
        texts = list(self.concepts)
        text_feats = self.clip_text_embeddings(texts, model=model, processor=processor, device=device)
        pairs = []
        for text, vec in zip(texts, text_feats):
            hybrid = self._expand_text_to_hybrid(vec)
            sim = self._cosine_similarity(hybrid, weights)
            pairs.append((text, sim))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[: max(1, int(top_k))]

    def fetch_extreme_samples(self, top_n: int = 3) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        if not os.path.exists(self.db_path):
            return [], []
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                "SELECT path, pred_score FROM images WHERE pred_score IS NOT NULL ORDER BY pred_score DESC LIMIT ?",
                (int(top_n),),
            )
            best = [(str(p), float(s)) for p, s in cur.fetchall() if p]
            cur.execute(
                "SELECT path, pred_score FROM images WHERE pred_score IS NOT NULL ORDER BY pred_score ASC LIMIT ?",
                (int(top_n),),
            )
            worst = [(str(p), float(s)) for p, s in cur.fetchall() if p]
            conn.close()
        except Exception:
            return [], []
        best = [(p, s) for p, s in best if os.path.exists(p)]
        worst = [(p, s) for p, s in worst if os.path.exists(p)]
        return best, worst

    @staticmethod
    def dim_block_name(dim: int) -> str:
        if dim < 768:
            return "环境"
        if dim < 1536:
            return "身形"
        return "面部"

    def top_weight_dims(self, weights: np.ndarray, k: int = 8) -> List[int]:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        k = max(1, int(k))
        idx = np.argsort(np.abs(weights))[-k:][::-1]
        return [int(i) for i in idx.tolist()]

    def top_block(self, weights: np.ndarray) -> str:
        stats = self.block_weight_stats(weights)
        items = [
            ("环境", stats.get("global_ratio", 0.0)),
            ("身形", stats.get("body_ratio", 0.0)),
            ("面部", stats.get("face_ratio", 0.0)),
        ]
        items.sort(key=lambda x: x[1], reverse=True)
        return items[0][0] if items else "环境"

    def explain_image_by_dims(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        dims: Sequence[int],
    ) -> str:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        parts = []
        for d in dims:
            if d < 0 or d >= weights.shape[0] or d >= x.shape[0]:
                continue
            contrib = float(weights[d] * x[d])
            parts.append((d, contrib))
        if not parts:
            return "缺少可解释的特征贡献"
        parts.sort(key=lambda x: abs(x[1]), reverse=True)
        best = parts[: min(3, len(parts))]
        dims_text = "、".join([f"{self.dim_block_name(d)}维度#{d}" for d, _ in best])
        return f"主要线性贡献来自 {dims_text}"

    def build_report(
        self,
        features_lookup: Optional[Dict[str, np.ndarray]] = None,
        top_k_concepts: int = 5,
        top_k_dims: int = 8,
        clip_model: Optional[Any] = None,
        clip_processor: Optional[Any] = None,
        clip_device: Optional[torch.device] = None,
    ) -> Dict:
        rm = self.load_rm_weights()
        weights = rm.weights
        stats = self.block_weight_stats(weights)
        report: Dict = {
            "rm_timestamp": rm.timestamp,
            "rm_path": os.path.abspath(self.rm_path),
            "block_stats": stats,
            "top_block": self.top_block(weights),
        }
        if clip_model is not None and clip_processor is not None and clip_device is not None:
            report["concepts_top"] = self.probe_concepts(
                weights,
                model=clip_model,
                processor=clip_processor,
                device=clip_device,
                top_k=top_k_concepts,
            )
        else:
            report["concepts_top"] = []
        dims = self.top_weight_dims(weights, k=top_k_dims)
        report["top_dims"] = [(d, self.dim_block_name(d), float(abs(weights[d]))) for d in dims]
        best, worst = self.fetch_extreme_samples(top_n=3)
        report["extremes"] = {"best": best, "worst": worst}
        if features_lookup:
            best_expl = []
            for p, s in best:
                x = features_lookup.get(p)
                if x is None:
                    continue
                best_expl.append((p, s, self.explain_image_by_dims(x, weights, dims)))
            worst_expl = []
            for p, s in worst:
                x = features_lookup.get(p)
                if x is None:
                    continue
                worst_expl.append((p, s, self.explain_image_by_dims(x, weights, dims)))
            report["extremes_explain"] = {"best": best_expl, "worst": worst_expl}
        else:
            report["extremes_explain"] = {"best": [], "worst": []}
        return report
