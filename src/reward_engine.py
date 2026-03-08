import os
import pickle
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class RewardModelEngine:
    def __init__(self, labels_path: str):
        self.labels_path = labels_path
        # fit_intercept=False 是 Bradley-Terry 模型在特征空间的数学要求
        self.model = LogisticRegression(max_iter=1000, C=1.0, fit_intercept=False)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.global_scores = {}

    def train(self, features_dict: dict) -> bool:
        """从 pairwise_labels.csv 提取特征差值，训练全局偏好模型"""
        if not os.path.exists(self.labels_path):
            return False

        try:
            df = pd.read_csv(self.labels_path)
            if len(df) < 5:
                return False

            X, y = [], []
            for _, row in df.iterrows():
                winner, loser = row['winner'], row['loser']
                if winner in features_dict and loser in features_dict:
                    f_w = features_dict[winner]
                    f_l = features_dict[loser]
                    
                    if hasattr(f_w, "cpu"):
                        f_w = f_w.detach().cpu().numpy()
                    if hasattr(f_l, "cpu"):
                        f_l = f_l.detach().cpu().numpy()

                    # 正样本：winner - loser -> 1 (赢)
                    X.append(f_w - f_l)
                    y.append(1)
                    # 数据增强 (Symmetry Augmentation)：loser - winner -> 0 (输)
                    X.append(f_l - f_w)
                    y.append(0)

            if not X:
                return False

            X = np.array(X)
            y = np.array(y)
            
            # 标准化特征差异
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # 训练完成后，立刻推断全库绝对分数
            self._compute_all_scores(features_dict)
            return True
            
        except Exception as e:
            print(f"Reward Model 训练失败: {e}")
            return False

    def save_model(self, model_dir="./data/models/"):
        """保存训练好的 Reward Model 及相关元数据"""
        if not self.is_trained:
            return False
        
        try:
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, "rm_predictor.pkl")
            
            data = {
                "model": self.model,
                "scaler": self.scaler,
                "raw_min": getattr(self, "raw_min", 0.0),
                "raw_max": getattr(self, "raw_max", 1.0),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            with open(model_path, "wb") as f:
                pickle.dump(data, f)
            
            # print(f"Reward Model saved to {model_path}")
            return True
            
        except Exception as e:
            print(f"保存模型失败: {e}")
            return False

    def _compute_all_scores(self, features_dict: dict):
        """将学到的审美权重 (w) 映射到全库所有图片，得出绝对分数 S(x) = w^T * x"""
        if not self.is_trained:
            return

        filenames = list(features_dict.keys())
        X_all = []
        for name in filenames:
            vec = features_dict[name]
            if hasattr(vec, "cpu"):
                vec = vec.detach().cpu().numpy()
            X_all.append(vec)

        X_all = np.array(X_all)
        weights = self.model.coef_[0]  # 提取模型学到的 2048 维偏好权重
        
        # 点乘：计算每张图片特征在“偏好方向”上的投影长度
        raw_scores = np.dot(X_all, weights)
        
        self.raw_min = raw_scores.min()
        self.raw_max = raw_scores.max()
        
        # 归一化到 0~100 方便人类阅读
        if self.raw_max != self.raw_min:
            norm_scores = (raw_scores - self.raw_min) / (self.raw_max - self.raw_min) * 100
        else:
            norm_scores = raw_scores

        self.global_scores = {name: float(score) for name, score in zip(filenames, norm_scores)}


    def get_leaderboard(self, top_n=50) -> List[Tuple[str, float]]:
        """获取泛化后的排行榜"""
        if not self.is_trained:
            return []
        sorted_scores = sorted(self.global_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_n]

    def get_uncertain_pair(self, all_filenames: List[str], ignored_pairs: set) -> Tuple[str, str]:
        """Active Learning: 挑选模型最拿不准的两个图 (即全库泛化分数最接近的对)"""
        if not self.is_trained or len(all_filenames) < 2:
            return None, None

        sample_size = min(60, len(all_filenames))
        samples = random.sample(all_filenames, sample_size)
        
        pairs = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                a, b = samples[i], samples[j]
                if (a, b) in ignored_pairs or (b, a) in ignored_pairs:
                    continue
                # 分差越小，说明在特征空间上模型越难抉择，点击的价值越高
                diff = abs(self.global_scores.get(a, 50) - self.global_scores.get(b, 50))
                pairs.append((a, b, diff))
                
        if not pairs:
            return None, None
        
        # 按分差升序排列，从最纠结的前 5 对中随机选 1 对
        pairs.sort(key=lambda x: x[2])
        best_pair = random.choice(pairs[:5])
        return best_pair[0], best_pair[1]
