# 个人审美偏好引擎项目指引 (AGENTS.md)

## 项目摘要
构建一个本地运行的基于 CLIP 的个人审美偏好引擎，实现小样本标注驱动的大规模图片筛选。

## 技术栈约定 (Strict Constraints)
- **环境**: Python 3.10+ (小于 3.13)。
- **深度学习**: PyTorch (必须适配 MPS/Mac 或 CUDA/Nvidia 加速)。
- **模型库**: HuggingFace Transformers (CLIP), facenet-pytorch, ultralytics (YOLO)。
- **存储与检索**: sqlite3 (内置), faiss-cpu。
- **存储规范**: 所有模型必须强制缓存在 `./data/models` 目录下。
- **规范检查**: 使用 `ruff check .` 进行 Linter 检查；禁止使用 mypy，保持灵活性。ruff检查需要激活 conda 的 aesthetic312 环境

## 核心架构决策
- **训练模式**: 采用 **Linear Probing** (冻结骨干，训练末端小模型)，禁止 Fine-tuning [2]。
- **特征工程**: 使用 Global CLIP + YOLO Person Crop + FaceNet 构建 **2048 维**混合特征。
- **排序模型**: Pairwise 排名以 **Elo** 等级分为基础。

## 核心目录结构
- `/data/` : 存放原图、模型缓存、数据库 (gallery.db) 及特征缓存 (features.pkl)。
- `/src/` : 存放核心业务模块 (ingest, extraction, db, engines)。
- `/src/pages/` : Streamlit 页面文件。

## UI 状态管理原则
- **Session State**: 必须使用 `dialog_idx` 管理预览索引；竞技场 UI 组件必须使用指定 Key（如 st.radio 必须绑定 key="active_ranking_engine"）以防状态丢失。
- **性能**: 必须使用 `st.cache_data` 处理缩略图。