# 🎨 Aesthetic Preference Engine

> **Teach AI your unique taste with just a few clicks.**

`Aesthetic Preference Engine` 是一个轻量级、私有化的个人审美偏好学习引擎。它通过多模态特征提取（全局语义、人体比例、面部细节）和主动学习算法，仅需少量（<200张）标注，即可在海量图片库中精准筛选出符合你审美偏好的内容。

---

## 🌟 核心特性

- **🧬 多模态 2048 维混合特征**：整合 CLIP（全局气质）、YOLO Person Crop（身形体态）和 FaceNet（五官细节）的高维特征，捕捉细粒度审美偏好。
- **🔍 审美视觉溯源 (Visual Tracing)**：基于 **Occlusion Sensitivity** 技术生成热力图，直观展示 AI 眼中图片的“美学贡献区”。
- **🧬 审美画像 (DNA Profiling)**：深度解析 Reward Model 权重，量化你在环境、身形、面部三个维度的偏好权重，并支持语义标签探测。
- **⚡ 相似度检索 (FAISS)**：集成 FAISS 向量索引，支持按“全局气质”、“身形构图”或“面部特征”在万级库中秒级检索相似图片。
- **⚔️ 双向排序竞技场 (Versus Arena)**：支持 Elo 与 Reward Model 排行切换，内置 **Bridge/Exploration/Calibration/Refinement** 四种混合采样策略。
- **⌨️ 键盘流操作**：集成 `streamlit-shortcuts`，支持 `A/D` 快速翻页或选图，`1-5` 数字键实时打分纠偏。
- **🧠 主动学习闭环**：标注后模型自动重训，评分实时更新，PDF 报告随引擎切换自适应导出。

---

## 🏗️ 系统架构

核心逻辑遵循 **Linear Probing** 方案：冻结预训练骨干网络，仅训练轻量回归模型。
- **单图评分**：采用 Ridge Regression (L2 正则) 进行快速拟合与增量学习。
- **排序模型**：采用基于 Logistic Regression 的 **Reward Model**，学习样本间的偏好差值。
- **采样算法**：集成混合采样策略，通过 Bridge 采样连接评分孤岛，通过 Exploration 挖掘潜在偏好。

---

## 🚀 快速开始——端到端使用指南 (End-to-End Workflow)

本指南将带你从零开始，构建并训练一个属于你的审美模型。

### 第一步：准备与特征提取

1.  **准备环境**：
    ```powershell
    # Windows 推荐使用脚本初始化
    ./scripts/setup_env.ps1
    # 或者手动安装
    pip install -r requirements.txt
    ```

2.  **准备数据**：
    将你的图片放入 `./data/raw_images/` 目录中。

3.  **提取特征 (Feature Extraction)**：
    运行特征提取脚本，这会为每张图片生成 2048 维的混合特征指纹（CLIP + YOLO + FaceNet）。
    ```bash
    python src/feature_extractor.py --input_dir ./data/raw_images
    ```
    *首次运行会自动下载模型权重到 `./data/models/`，请耐心等待。*

### 第二步：冷启动与模型训练

1.  **启动工作台**：
    ```bash
    streamlit run src/app.py
    ```
    浏览器会自动打开 `http://localhost:8501`。

2.  **定义你的品味 (Cold Start)**：
    - 进入 **“🏷️ 标注大厅”** 选项卡。
    - 系统会展示未标注的图片。
    - 凭直觉为图片打分（1-5分）。
    - *建议至少标注 10-20 张不同风格的图片，以便模型捕捉你的偏好。*

3.  **训练模型 (Model Training)**：
    - 进入 **“⚙️ 模型控制台”** 选项卡。
    - 点击 **“触发模型训练”** 按钮。
    - 系统会同时训练 **Ridge 预测器** 与 **Reward Model**。
    - 等待系统提示 `✅ 训练总耗时: X.XX 秒`，此时你的专属审美模型已生成。

4.  **全量预测 (Inference)**：
    - 点击 **“扫描并预测”**。
    - 系统将使用新模型为所有图片打分，并自动构建 **FAISS** 索引。

### 第三步：浏览与进化 (Active Learning)

1.  **探索图库与溯源**：
    - 进入 **“🗂️ 审美图库”** 选项卡。
    - 点击图片进入 **沉浸式预览 (Dialog)**。
    - 点击 **“🔍 视觉溯源”**：AI 会揭示其审美热力图（遮挡敏感度分析）。
    - 使用 `D` / `→` 下一张，`A` / `←` 上一张。

2.  **沉浸式纠偏**：
    - `1` - `5`：**直接数字键打分**。
    - *按下数字键时，系统会自动记录并触发增量训练，模型会越用越懂你！*

---

## 📂 项目结构

```text
├── data/
│   ├── raw_images/        # 图片原片
│   ├── models/            # 离线模型权重缓存
│   │   ├── aesthetic_predictor.pkl  # 岭回归模型
│   │   └── rm_predictor.pkl         # 奖励模型 (Reward Model)
│   ├── gallery.db         # SQLite 数据库（分数、元数据）
│   ├── features.pkl       # 2048 维混合特征缓存
│   ├── labels.csv         # 单图评分记录
│   └── pairwise_labels.csv# 两两对比记录
├── src/
│   ├── app.py             # Streamlit 交互大厅
│   ├── feature_extractor.py # 多模态特征提取 (CLIP + YOLO + FaceNet)
│   ├── train.py           # 模型训练与增量更新
│   ├── visual_tracer.py   # 视觉溯源核心 (Occlusion Sensitivity)
│   ├── analyzer.py        # 审美画像与 DNA 分析
│   ├── faiss_indexer.py   # FAISS 向量检索索引
│   ├── ranking_engine.py  # Elo 排序与混合采样策略
│   ├── reward_engine.py   # Reward Model 偏好学习
│   ├── db.py              # 数据库底层操作
│   └── export_engine.py   # PDF 导出 (自适应布局)
├── src/pages/
│   └── 1_Versus_Arena.py  # 双向排序竞技场
└── requirements.txt
```

---

## ⚔️ 双向排序竞技场 (Versus Arena)

- **核心功能**：
  - **混合采样策略**：支持 `Bridge` (桥接断层)、`Exploration` (偏好探索)、`Calibration` (基准校准) 和 `Refinement` (势均力敌)。
  - **引擎切换**：在 Elo 等级分与 Reward Model 全局排序间一键切换。
  - **溯源与分析**：在对战过程中可点击“🔍溯源”直接查看图片的热力图分析。
  - **导出报告**：生成高 DPI 的审美巅峰榜单 PDF。

---

## 🛠️ 技术栈

- **Core**: PyTorch, Transformers, Ultralytics (YOLOv8), Facenet-PyTorch
- **Algorithm**: Scikit-Learn (Ridge/Logistic Regression), **FAISS** (向量检索)
- **Frontend**: Streamlit, Streamlit-Shortcuts
- **Ranking**: Elo + Reward Model (Bradley-Terry based), 混合主动采样策略
- **Optimization**: VRAM Batch 推理优化（适配 4GB 显存），Occlusion Sensitivity 遮挡敏感度。

---

## 💡 开发者说

这个项目的初衷是解决“收藏夹图片太多却难以筛选”的问题。通过 2048 维混合特征，AI 不仅能读懂图片语义（CLIP），还能感知你对人脸比例或身材曲线的偏好。

**欢迎 Fork 并开启你的私有化审美进化之路！**

---

## 🔧 开发者指南

- Lint：

  ```powershell
  conda run -n aesthetic312 ruff check .
  conda run -n aesthetic312 ruff check . --fix
  ```

- 常用命令：

  ```powershell
  # 工作台
  streamlit run src/app.py

  # 竞技场
  streamlit run src/pages/1_Versus_Arena.py

  # 训练
  python src/train.py
  ```
