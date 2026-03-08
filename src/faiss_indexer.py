import os
import pickle
from typing import Dict, List

import faiss
import numpy as np
import torch


class FaissIndexer:
    def __init__(self, features_path: str = "./data/features.pkl"):
        self.features_path = features_path
        self.names: List[str] = []
        self.pools: Dict[str, np.ndarray] = {}
        self.indexes: Dict[str, faiss.IndexFlatIP] = {}
        self._build()

    def _load_features(self):
        if not os.path.exists(self.features_path):
            return {}
        with open(self.features_path, "rb") as f:
            return pickle.load(f)

    def _to_numpy(self, vector):
        if isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy()
        return np.asarray(vector, dtype=np.float32)

    def _build(self):
        data = self._load_features()
        if not data:
            return
        names = list(data.keys())
        vectors = np.stack([self._to_numpy(data[name]) for name in names], axis=0)
        vectors = np.asarray(vectors, dtype=np.float32)
        pools = {
            "full": vectors[:, 0:2048],
            "global": vectors[:, 0:768],
            "body": vectors[:, 768:1536],
            "face": vectors[:, 1536:2048],
        }
        indexes = {}
        for key, arr in pools.items():
            if arr.size == 0:
                continue
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            faiss.normalize_L2(arr)
            index = faiss.IndexFlatIP(arr.shape[1])
            index.add(arr)
            pools[key] = arr
            indexes[key] = index
        self.names = names
        self.pools = pools
        self.indexes = indexes

    def search_similar(self, target_filename: str, search_type: str, top_k: int = 20):
        if not self.names:
            return []
        if search_type not in self.indexes:
            return []
        if target_filename not in self.names:
            return []
        idx = self.names.index(target_filename)
        vectors = self.pools.get(search_type)
        index = self.indexes.get(search_type)
        if vectors is None or index is None or vectors.size == 0:
            return []
        k = min(top_k + 1, len(self.names))
        _, indices = index.search(vectors[idx : idx + 1], k)
        results = []
        for i in indices[0].tolist():
            name = self.names[i]
            if name == target_filename:
                continue
            results.append(name)
            if len(results) >= top_k:
                break
        return results


_INDEXER = None


def get_indexer(features_path: str = "./data/features.pkl"):
    global _INDEXER
    if _INDEXER is None or _INDEXER.features_path != features_path:
        _INDEXER = FaissIndexer(features_path)
    return _INDEXER


def search_similar(target_filename: str, search_type: str, top_k: int = 20):
    indexer = get_indexer()
    return indexer.search_similar(target_filename, search_type, top_k=top_k)
