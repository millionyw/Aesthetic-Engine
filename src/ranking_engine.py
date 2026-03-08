import os
import random
from typing import Dict, List, Tuple

import pandas as pd


class RankingEngine:
    def __init__(self, labels_path: str, features_path: str):
        self.labels_path = labels_path
        self.features_path = features_path
        self.elo_scores = {}
        self.match_counts = {}
        self.lcc_set = set()
        self.n_components = 0
        self.k_factor = 32
        self.initial_score = 1500

    def _k_for(self, match_count: int) -> int:
        return 64 if match_count <= 3 else 16

    def compute_elo(self) -> Dict[str, float]:
        if not os.path.exists(self.labels_path):
            self.elo_scores = {}
            self.match_counts = {}
            return {}

        scores = {}
        match_counts = {}
        try:
            df = pd.read_csv(self.labels_path)
            for _, row in df.iterrows():
                winner = row.get("winner")
                loser = row.get("loser")
                if not winner or not loser:
                    continue
                if winner not in scores:
                    scores[winner] = self.initial_score
                if loser not in scores:
                    scores[loser] = self.initial_score
                if winner not in match_counts:
                    match_counts[winner] = 0
                if loser not in match_counts:
                    match_counts[loser] = 0
                ra = scores[winner]
                rb = scores[loser]
                ea = 1 / (1 + 10 ** ((rb - ra) / 400))
                eb = 1 / (1 + 10 ** ((ra - rb) / 400))
                k_winner = self._k_for(match_counts[winner])
                k_loser = self._k_for(match_counts[loser])
                scores[winner] = ra + k_winner * (1 - ea)
                scores[loser] = rb + k_loser * (0 - eb)
                match_counts[winner] += 1
                match_counts[loser] += 1
        except Exception:
            self.elo_scores = {}
            self.match_counts = {}
            return {}

        self.elo_scores = scores
        self.match_counts = match_counts
        return scores

    def get_leaderboard(self, top_n: int = 50) -> List[Tuple[str, float]]:
        sorted_scores = sorted(self.elo_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]

    def get_sigma(self, name: str) -> float:
        count = self.match_counts.get(name, 0)
        return 1 / ((count + 1) ** 0.5)

    def get_next_pair(
        self,
        all_filenames: List[str],
        predictor_scores: Dict[str, float],
        ignored_pairs: set,
        mode_weights: Tuple[float, float, float, float] | None = None,
        sample_size: int = 60,
        uncertain_top_n: int = 30,
    ) -> Tuple[str, str]:
        if len(all_filenames) < 2:
            return None, None

        if not self.match_counts:
            self.match_counts = {name: 0 for name in all_filenames}
        if not self.elo_scores:
            self.elo_scores = {name: self.initial_score for name in all_filenames}

        self._get_connectivity_info(all_filenames)

        if mode_weights is None:
            mode_weights = (0.3, 0.3, 0.2, 0.2)
        total_weight = sum(mode_weights)
        if total_weight <= 0:
            mode_weights = (0.3, 0.3, 0.2, 0.2)
            total_weight = 1.0
        bridge_w, exploration_w, calibration_w = (
            mode_weights[0] / total_weight,
            mode_weights[1] / total_weight,
            mode_weights[2] / total_weight,
        )

        mode_rand = random.random()
        if mode_rand < bridge_w:
            left, right = self._sample_bridge(all_filenames, ignored_pairs)
            if left and right:
                return left, right
        elif mode_rand < bridge_w + exploration_w:
            left, right = self._sample_exploration(all_filenames, ignored_pairs)
            if left and right:
                return left, right
        elif mode_rand < bridge_w + exploration_w + calibration_w:
            left, right = self._sample_calibration(all_filenames, ignored_pairs)
            if left and right:
                return left, right

        left, right = self._sample_refinement(all_filenames, predictor_scores, ignored_pairs, sample_size, uncertain_top_n)
        if left and right:
            return left, right

        return self._sample_random(all_filenames, ignored_pairs)

    def _sample_exploration(self, all_filenames: List[str], ignored_pairs: set) -> Tuple[str, str]:
        counts = sorted(((self.match_counts.get(name, 0), name) for name in all_filenames), key=lambda x: x[0])
        lowest = counts[:max(5, min(20, len(counts)))]
        candidates = [name for _, name in lowest]
        for _ in range(20):
            left, right = random.sample(candidates, 2)
            if (left, right) not in ignored_pairs and (right, left) not in ignored_pairs:
                return left, right
        return None, None

    def _sample_calibration(self, all_filenames: List[str], ignored_pairs: set) -> Tuple[str, str]:
        scored = [(self.elo_scores.get(name, self.initial_score), name) for name in all_filenames]
        scored.sort(key=lambda x: x[0], reverse=True)
        if len(scored) < 2:
            return None, None
        top_count = max(1, int(len(scored) * 0.1))
        bottom_count = max(1, int(len(scored) * 0.3))
        top_pool = [name for _, name in scored[:top_count]]
        bottom_pool = [name for _, name in scored[-bottom_count:]]
        for _ in range(20):
            left = random.choice(top_pool)
            right = random.choice(bottom_pool)
            if left != right and (left, right) not in ignored_pairs and (right, left) not in ignored_pairs:
                return left, right
        return None, None

    def _sample_bridge(self, all_filenames: List[str], ignored_pairs: set) -> Tuple[str, str]:
        if not self.lcc_set:
            return None, None
        scored = [(self.elo_scores.get(name, self.initial_score), name) for name in all_filenames]
        scored.sort(key=lambda x: x[0], reverse=True)
        top_count = max(1, int(len(scored) * 0.1))
        top_pool = [name for _, name in scored[:top_count]]
        outliers = [name for name in top_pool if name not in self.lcc_set]
        if not outliers:
            return None, None
        lcc_candidates = sorted(
            ((self.match_counts.get(name, 0), name) for name in self.lcc_set),
            key=lambda x: x[0],
            reverse=True,
        )
        anchor_pool = [name for _, name in lcc_candidates[:max(5, min(20, len(lcc_candidates)))]]
        for _ in range(30):
            left = random.choice(outliers)
            right = random.choice(anchor_pool)
            if left != right and (left, right) not in ignored_pairs and (right, left) not in ignored_pairs:
                return left, right
        return None, None

    def _sample_refinement(
        self,
        all_filenames: List[str],
        predictor_scores: Dict[str, float],
        ignored_pairs: set,
        sample_size: int,
        uncertain_top_n: int,
    ) -> Tuple[str, str]:
        if not predictor_scores:
            return None, None
        sample_files = random.sample(all_filenames, min(sample_size, len(all_filenames)))
        pairs = []
        for i in range(len(sample_files)):
            for j in range(i + 1, len(sample_files)):
                f1 = sample_files[i]
                f2 = sample_files[j]
                if (f1, f2) in ignored_pairs or (f2, f1) in ignored_pairs:
                    continue
                diff = abs(predictor_scores.get(f1, 3) - predictor_scores.get(f2, 3))
                pairs.append((diff, f1, f2))
        pairs.sort(key=lambda x: x[0])
        top_uncertain = pairs[:uncertain_top_n]
        if top_uncertain:
            _, left, right = random.choice(top_uncertain)
            return left, right
        return None, None

    def _sample_random(self, all_filenames: List[str], ignored_pairs: set) -> Tuple[str, str]:
        for _ in range(30):
            left, right = random.sample(all_filenames, 2)
            if (left, right) not in ignored_pairs and (right, left) not in ignored_pairs:
                return left, right
        return random.sample(all_filenames, 2)

    def _get_connectivity_info(self, all_filenames: List[str]) -> Tuple[int, set]:
        if not all_filenames:
            self.n_components = 0
            self.lcc_set = set()
            return 0, set()

        adjacency = {name: set() for name in all_filenames}
        if os.path.exists(self.labels_path):
            try:
                df = pd.read_csv(self.labels_path)
                for _, row in df.iterrows():
                    winner = row.get("winner")
                    loser = row.get("loser")
                    if not winner or not loser:
                        continue
                    if winner not in adjacency:
                        adjacency[winner] = set()
                    if loser not in adjacency:
                        adjacency[loser] = set()
                    adjacency[winner].add(loser)
                    adjacency[loser].add(winner)
            except Exception:
                pass

        visited = set()
        components = []
        for node in adjacency.keys():
            if node in visited:
                continue
            queue = [node]
            visited.add(node)
            component = set()
            while queue:
                current = queue.pop()
                component.add(current)
                for neighbor in adjacency.get(current, []):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    queue.append(neighbor)
            components.append(component)

        if not components:
            self.n_components = 0
            self.lcc_set = set()
            return 0, set()

        components.sort(key=lambda x: len(x), reverse=True)
        self.n_components = len(components)
        self.lcc_set = components[0]
        return self.n_components, self.lcc_set

    def get_connectivity_info(self, all_filenames: List[str]) -> Tuple[int, set]:
        return self._get_connectivity_info(all_filenames)

    def is_in_lcc(self, name: str) -> bool:
        return name in self.lcc_set
