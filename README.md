# 🎨 Aesthetic Preference Engine

> **Teach AI your unique taste with just a few clicks.**

`Aesthetic Preference Engine` 是一个轻量级、私有化的个人审美偏好学习引擎。它通过多模态特征提取（全局语义、人体比例、面部细节）和主动学习算法，仅需少量（<200张）标注，即可在海量图片库中精准筛选出符合你审美偏好的内容。

---

## 🌟 核心特性

- **🧬 多模态 2048 维混合特征**：整合 CLIP（全局气质）、YOLO Person Crop（身形体态）和 FaceNet（五官细节）的高维特征，捕捉细粒度审美偏好。
- **⚡ 沉浸式标注体验**：基于 Streamlit 构建图库界面，缩略图缓存加载，万级图片库依旧流畅。
- **⌨️ 键盘流操作**：集成 `streamlit-shortcuts`，支持 `A/D` 或 `←/→` 快速翻页，`Esc` 关闭预览。
- **🧠 主动学习闭环**：标注后模型后台静默重训，评分实时更新并支持解释性分析。
- **🗄️ 工业级存储**：SQLite 存储评分与元数据，Pickle 缓存特征张量，保证数据安全与读取性能。
- **⚔️ 双向排序竞技场 (Versus Arena)**：支持 Elo 与 Reward Model 排行切换、σ 与 LCC 徽标展示、分页与 PDF 导出（随引擎切换）。

---

## 🏗️ 系统架构

核心逻辑遵循 **Linear Probing** 方案：冻结预训练骨干网络，仅训练轻量回归模型（Ridge Regression），以低算力成本实现高个性化拟合。

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
    - 等待系统提示 `✅ 训练总耗时: X.XX 秒`，此时你的专属审美模型 `aesthetic_predictor.pkl` 已生成。

4.  **全量预测 (Inference)**：
    - 在控制台中输入图片目录（默认为空则使用 `ingest_images.py` 逻辑，通常无需输入）。
    - 点击 **“扫描并预测”**。
    - 系统将使用刚训练好的模型，为所有图片打分并存入数据库。

### 第三步：浏览与进化 (Active Learning)

1.  **探索图库**：
    - 进入 **“🗂️ 审美图库”** 选项卡。
    - 图片已按预测分数降序排列（5分 -> 1分）。（支持数字键1~5直接打分）

2.  **沉浸式纠偏**：
    - 点击任意感兴趣的图片，进入 **沉浸式预览模式 (Dialog)**。
    - 查看 **“贡献度分析”**：AI 会告诉你它是被“气质”、“体态”还是“五官”吸引。
    - **键盘流操作**：
        - `D` / `→`：下一张
        - `A` / `←`：上一张
        - `1` - `5`：**直接打分纠正**。
    - *当你按下数字键时，系统会自动将新分数写入数据库，并**即时触发增量训练**。你的模型会越用越懂你！*

---

## 📂 项目结构

```text
├── data/
│   ├── raw_images/        # 图片原片
│   ├── models/            # 离线模型权重缓存
│   │   └── aesthetic_predictor.pkl  # 审美回归模型与标准化器
│   ├── gallery.db         # SQLite 数据库（分数、路径、时间）
│   ├── features.pkl       # 2048 维多模态特征缓存
│   ├── labels.csv         # 单图评分记录
│   └── pairwise_labels.csv# 两两对比记录（winner, loser, timestamp）
├── src/
│   ├── app.py             # Streamlit 交互大厅
│   ├── ingest_images.py   # 数据入库流水线
│   ├── feature_extractor.py # 多模态特征提取核心
│   ├── train.py           # 模型训练与增量更新
│   ├── db.py              # 数据库底层操作
│   ├── ranking_engine.py  # Elo 排序、连通性/LCC、混合采样
│   ├── reward_engine.py   # Reward Model 泛化评分与不确定性采样
│   └── export_engine.py   # PDF 导出（自适应布局与高 DPI）
├── src/pages/
│   └── 1_Versus_Arena.py  # 双向排序竞技场页面
├── scripts/
│   └── setup_env.ps1
└── requirements.txt
```

---

## ⚔️ 双向排序竞技场 (Versus Arena)

- 启动：

  ```bash
  streamlit run src/pages/1_Versus_Arena.py
  ```

- 操作与功能：
  - `A` / `←` 选左图，`D` / `→` 选右图，`S` 跳过
  - 右侧“审美巅峰榜”：
    - Elo / Reward Model 排行切换（Radio）
    - “仅显示已比对次数 > 3”（Toggle）
    - 显示 σ、LCC 徽标（🌐）或 RM 徽标（🧠）
    - 分页浏览，PDF 导出随当前引擎生成

---

## 🛠️ 技术栈

- **Core**: PyTorch, Transformers, Ultralytics (YOLOv8), Facenet-PyTorch
- **Algorithm**: Scikit-Learn (Ridge Regression)
- **Frontend**: Streamlit
- **Database**: SQLite
- **Ranking**: Elo + Reward Model（可切换），混合采样（桥接/探索/校准/精修）
- **Export**: PDF 导出（保持长宽比、自适应布局、更高清晰度）

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
