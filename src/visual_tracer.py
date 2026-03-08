import gc
import io
import os
import pickle
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image

from feature_extractor import clip_feature, extract_hybrid_feature, get_device


@dataclass(frozen=True)
class OcclusionResult:
    image_path: str
    base_score: float
    heatmap: np.ndarray
    overlay_png: bytes
    patch_size: int
    stride: int
    width: int
    height: int


MaskFill = Literal["mean", "gray", "black"]
Mode = Literal["fixed_crops", "full_hybrid"]


class OcclusionExplainer:
    def __init__(
        self,
        models,
        rm_path: str = "./data/models/rm_predictor.pkl",
        device: torch.device | None = None,
        mask_fill: MaskFill = "mean",
        batch_size: int = 32,
        mode: Mode = "fixed_crops",
    ):
        self.rm_path = rm_path
        self.device = device if device is not None else get_device()
        self.mask_fill = mask_fill
        self.batch_size = int(batch_size)
        self.mode = mode
        self._models = models
        self._rm_data = None

    def _load_rm(self):
        if self._rm_data is not None:
            return self._rm_data
        if not os.path.exists(self.rm_path):
            self._rm_data = None
            return None
        with open(self.rm_path, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or "model" not in data or "scaler" not in data:
            raise ValueError("rm_predictor.pkl 格式不正确")
        self._rm_data = data
        return data

    def _score_rm(self, vectors_2048: np.ndarray) -> np.ndarray:
        data = self._load_rm()
        if data is None:
            raise FileNotFoundError(self.rm_path)
        predictor = data.get("model")
        scaler = data.get("scaler")
        raw_min = float(data.get("raw_min", 0.0))
        raw_max = float(data.get("raw_max", 1.0))
        if predictor is None or scaler is None:
            raise ValueError("rm_predictor.pkl 缺少 model 或 scaler")

        vectors_scaled = scaler.transform(vectors_2048)
        raw_scores = predictor.decision_function(vectors_scaled)
        if raw_max != raw_min:
            norm_scores = (raw_scores - raw_min) / (raw_max - raw_min) * 100.0
        else:
            norm_scores = raw_scores
        scores = np.clip(norm_scores / 20.0, 0.0, 5.0)
        return scores.astype(np.float32)

    @staticmethod
    def _jet_colormap01(x01: np.ndarray) -> np.ndarray:
        x01 = np.clip(x01, 0.0, 1.0)
        r = np.clip(1.5 - np.abs(4.0 * x01 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * x01 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * x01 - 1.0), 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        return (rgb * 255.0).round().astype(np.uint8)

    def _mask_color(self, base_rgb: np.ndarray) -> np.ndarray:
        if self.mask_fill == "gray":
            return np.array([127, 127, 127], dtype=np.uint8)
        if self.mask_fill == "black":
            return np.array([0, 0, 0], dtype=np.uint8)
        mean = base_rgb.reshape(-1, 3).mean(axis=0)
        return np.clip(mean.round(), 0, 255).astype(np.uint8)

    def explain(
        self,
        image_path: str,
        patch_size: int = 64,
        stride: int = 32,
        overlay_alpha: float = 0.45,
    ) -> OcclusionResult:
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        if patch_size <= 0 or stride <= 0:
            raise ValueError("patch_size/stride 必须为正数")

        with Image.open(image_path) as img:
            base_img = img.convert("RGB")
        base_rgb = np.asarray(base_img, dtype=np.uint8)
        height, width = int(base_rgb.shape[0]), int(base_rgb.shape[1])
        fill = self._mask_color(base_rgb)

        clip_model, processor, yolo_model, mtcnn, resnet = self._models
        
        with torch.inference_mode():
            base_hybrid = extract_hybrid_feature(
                base_img, clip_model, processor, yolo_model, mtcnn, resnet, self.device
            )
            base_vec = base_hybrid.detach().cpu().numpy()[None, :]
            base_score = float(self._score_rm(base_vec)[0])

            x_positions = list(range(0, max(1, width - patch_size + 1), stride))
            y_positions = list(range(0, max(1, height - patch_size + 1), stride))
            heat = np.zeros((len(y_positions), len(x_positions)), dtype=np.float32)

            if self.mode == "fixed_crops":
                base_v2 = base_hybrid[768:1536]
                base_v3 = base_hybrid[1536:]

            pending_imgs = []
            pending_idx = []

            def flush():
                if not pending_imgs:
                    return
                if self.mode == "full_hybrid":
                    masked = [
                        extract_hybrid_feature(
                            m, clip_model, processor, yolo_model, mtcnn, resnet, self.device
                        )
                        for m in pending_imgs
                    ]
                    masked_hybrid = torch.stack(masked, dim=0)
                else:
                    v1_list = [clip_feature(m, clip_model, processor, self.device) for m in pending_imgs]
                    v1 = torch.stack(v1_list, dim=0)
                    v2 = base_v2.unsqueeze(0).expand(v1.shape[0], -1)
                    v3 = base_v3.unsqueeze(0).expand(v1.shape[0], -1)
                    masked_hybrid = torch.cat([v1, v2, v3], dim=-1)

                masked_vecs = masked_hybrid.detach().cpu().numpy()
                masked_scores = self._score_rm(masked_vecs)
                deltas = base_score - masked_scores
                for (iy, ix), delta in zip(pending_idx, deltas):
                    heat[iy, ix] = float(delta)
                pending_imgs.clear()
                pending_idx.clear()

            for iy, y in enumerate(y_positions):
                for ix, x in enumerate(x_positions):
                    masked_rgb = base_rgb.copy()
                    masked_rgb[y : y + patch_size, x : x + patch_size] = fill
                    pending_imgs.append(Image.fromarray(masked_rgb))
                    pending_idx.append((iy, ix))
                    if len(pending_imgs) >= self.batch_size:
                        flush()
            flush()

        # Free some memory immediately
        torch.cuda.empty_cache()
        gc.collect()

        heat_pos = np.maximum(heat, 0.0)
        denom = float(heat_pos.max())
        heat01 = heat_pos / denom if denom > 0 else heat_pos
        heat_img = Image.fromarray((heat01 * 255.0).round().astype(np.uint8), mode="L").resize(
            (width, height), resample=Image.BILINEAR
        )
        heat01_full = np.asarray(heat_img, dtype=np.float32) / 255.0
        heat_rgb = self._jet_colormap01(heat01_full)

        orig = base_rgb.astype(np.float32)
        overlay = (1.0 - float(overlay_alpha)) * orig + float(overlay_alpha) * heat_rgb.astype(np.float32)
        overlay = np.clip(overlay.round(), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        overlay_png = buf.getvalue()

        return OcclusionResult(
            image_path=image_path,
            base_score=base_score,
            heatmap=heat,
            overlay_png=overlay_png,
            patch_size=int(patch_size),
            stride=int(stride),
            width=width,
            height=height,
        )
