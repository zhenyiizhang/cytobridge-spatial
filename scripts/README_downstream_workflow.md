# CytoBridge 下游分析一键示例（对应 ST-1104 downstream notebooks）

这个目录提供一个**固定流程**的示例脚本：`scripts/run_downstream_workflow_example.py`，用于在**不改训练/预处理逻辑**的前提下，复现 ST-1104 的下游分析链路（插值/分类/velocity 分解/growth+gene/communication+sankey+3D）。

## 1) 环境准备（推荐 DeepRUOTv2）

你说可以用 `DeepRUOTv2` conda 环境测试，下面按它写：

```bash
source /lustre/home/2501111653/miniconda3/etc/profile.d/conda.sh
conda activate DeepRUOTv2
```

建议用 editable 安装（方便脚本在任意路径运行并正确 import 包）：

```bash
cd /lustre/home/2501111653/CytoBridge-ST-package/CytoBridge
pip install -e .
```

依赖（下游会用到）：`torch`、`numpy`、`pandas`、`sklearn`、`anndata`、`matplotlib`、`scanpy`、`scvelo`、`umap-learn`、`plotly`、`torch_geometric`。

## 2) 输入数据要求（AnnData-only）

### `--aligned-h5ad`（必需）

你在预处理/对齐后通常会得到一个 aligned 的 `.h5ad`（例如 `preprocess_align_to_files(..., output_h5ad=...)` 生成的）。脚本会从这个 h5ad 里直接取：

- 时间：`adata.obs[time_key]`
  - `time_key` 可以用 `--time-key` 指定；如果不填，会自动尝试 `time_point_processed` → `samples` → `time`
- 特征矩阵（兼容当前训练输入）：
  - 默认自动使用：`[adata.obsm['spatial_aligned'], adata.obsm[obsm_key]]`（如果 `spatial_aligned` 存在）
  - 其中 `obsm_key` 默认 `X_latent`，不存在时回退 `adata.X`
  - 可用 `--no-concat-spatial` 关闭拼接；可用 `--spatial-key` 改空间坐标键名
- 注释（可选）：`adata.obs[Annotation]`（或 `--annotation-key` 指定的列）

这条路线的好处是：**不会丢 `obs_names`（cell id）**，最不容易出现你说的 “index 乱 / 对不齐”。
脚本内部已经统一复用 `CytoBridge.tl.downstream_data` 的适配函数（time parsing / feature inference / annotation merge），避免 notebook 与脚本出现分叉行为。

## 3) 运行：一键固定流程

最小可跑（h5ad 输入 + 不画 3D）：

```bash
cd /lustre/home/2501111653/CytoBridge-ST-package/CytoBridge
python scripts/run_downstream_workflow_example.py \
  --model-dir /path/to/results_dir \
  --aligned-h5ad /path/to/aligned.h5ad \
  --obsm-key X_latent \
  --concat-spatial \
  --spatial-key spatial_aligned \
  --out-dir /path/to/out \
  --device cpu \
  --skip-3d
```

带 annotation（推荐，用于 classifier / sankey / 3D ribbons；若已在 h5ad 的 `obs` 里就不需要额外提供）：

```bash
python scripts/run_downstream_workflow_example.py \
  --model-dir /path/to/results_dir \
  --aligned-h5ad /path/to/aligned.h5ad \
  --annotation-key Annotation \
  --time-subdivisions 5 \
  --out-dir /path/to/out \
  --device cuda
```

启用 3D + focus-anchor（可选）：

```bash
python scripts/run_downstream_workflow_example.py \
  --model-dir /path/to/results_dir \
  --aligned-h5ad /path/to/aligned.h5ad \
  --out-dir /path/to/out \
  --comm-focus-label "YourCellType" \
  --edge-top-k 6 \
  --fate-focus-label "YourCellType" \
  --fate-min-flow 10 \
  --focus-anchor-label "YourCellType"
```

### `--model-dir` 需要包含什么？

脚本会从 `--model-dir` 读取：
- `config.yaml`
- 一个 stage 目录下的 `last_model.pth`（优先 `Finetune/last_model.pth`，否则自动回退到存在的最后一个 stage）
- 若存在 score checkpoint，则优先加载 `Train_Score_Final/score_model.pth`，否则 `Train_Score/score_model.pth`

### 连续轨迹怎么生成

- 用 `--time-subdivisions N` 控制相邻观测时间点之间的插值密度：
  - `N=1`：只在观测时间点输出
  - `N=2`：每段加一个中点（原默认行为）
  - `N=5/10`：更连续、更密集的轨迹
- 脚本内部会把这个时间网格传给 `simulate_sde_points_split(...)`，因此生成的是连续时间上的 split-SDE 轨迹。

### 现在已经支持 adata-first 的下游 API

- `CytoBridge.tl.simulate_sde_points_split(...)`：只接受 `adata`，并按 `spatial_aligned + X_latent` 组装
- `CytoBridge.tl.train_mlp_classifier(...)`：只接受 `adata` 训练下游分类器
- `CytoBridge.tl.compute_drift(...)`：只接受 `adata` 计算分时 drift

## 4) 输出结构

脚本会在 `--out-dir`（默认 `<model-dir>/downstream_workflow`）下生成：

- `01_interpolation/`：split-SDE 轨迹 + 对比图
- `02_classifier/`：分类器指标 + 轨迹预测 label（若有 annotation）
- `03_velocity/`：intrinsic / interaction / full 的 scVelo stream 图 + 方向相关性图
- `04_growth_gene/`：growth map + gene-velocity embedding（UMAP/PCA）
- `05_communication/`：attention 保存、communication 统计、sankey html、3D html（若启用）
- `manifest.json`：记录输入路径、timepoints、stage 选择、输出目录等

## 5) 常见问题

1) **报 `scanpy/scvelo/torch_geometric` 缺失**
   - 说明当前环境没装全依赖；建议直接在 `DeepRUOTv2` 环境里补齐或切到你平时跑 notebook 的环境。

2) **时间点顺序不对**
   - 脚本内部对 `samples` 做了 `sorted(unique)`；但要求 `samples` 可转成数值（`float(samples)`）。

3) **没有 annotation 也想跑 communication**
   - 可以跑，但会用 `Unknown` 作为 label；Sankey/3D fate ribbon 需要 `predicted_labels_list`（因此需要 annotation 来训练 classifier 或你自己提供预测标签）。
