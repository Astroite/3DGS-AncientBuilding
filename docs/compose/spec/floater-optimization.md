---
feature: floater-optimization
status: delivered
updated: 2026-09-21
branch: main
commits: 1f2aecd..df96c9f
---

# 杂云消除：Bilateral Grid + Sparse Depth + 古建采集 SOP

## Report

**What was built**

按知天下 AI《室内场景 3DGS 消除杂云》（BV1siXrBsE2t）及 BilARF/gsplat 结论，在原生 gsplat 路径落地两项杂云抑制，并补齐古建采集 SOP：

- **Bilateral Grid 匀光**（`src/gsdb/photometric.py`）：按全景时间组的局部仿射双边网格，默认替换 `PanoramaExposure`；`--no-use-bilateral-grid` 回退旧补偿，`--no-photo-comp` 恒等。`select_photometric` 固化三路路由。
- **Sparse Depth 深度锚**（`src/gsdb/sparse_depth.py`）：`prepare_segment` 从 COLMAP 稀疏点导出 `sparse_depth.npz`（row_index/u/v/depth/point_id），纳入 package 哈希；训练在 `RGB+ED` 期望深度上做 L1。旧包无锚文件自动 `unavailable` 并跳过。默认开，`--no-use-sparse-depth` 关。
- **入口**：CLI `train` / `train_package` / Studio 均支持新开关（默认 True）；`dispatch.json` 身份含旗标；缺键或显式 null 按历史 False 对称比较，兼容旧 gsplat resume。Postshot 算法与默认后端不变。
- **采集 SOP**：`docs/CAPTURE-SOP.md` 表盘走位法（向心环绕→井字→离心扫墙→匾额对联细节）、快门 ≥1/500s、匀速约 1 m/s；`CURRENT-WORKFLOW.md` §2/§4 挂链并注明新开关。

**Verification**

| 命令 | 结果 |
| --- | --- |
| `PYTHONPATH=src .venv\Scripts\python.exe -m pytest tests -ra` | **10 passed**（最终） |
| `py_compile` 全部改动模块 | PASS |
| 历史 308 测试套件 | **PRE-EXISTING unavailable**（sparse checkout 无 `tests/`） |
| 真实 GPU 训练 / Capture 重建 | 未运行（AGENTS 禁止本切片执行） |

**Journey log**

1. 视频结论可映射为：光度残差→Bilateral Grid；距离错位→Sparse Depth；走位/快门→采集 SOP。
2. 独立审查曾误判 gsplat 输出为 CHW——现有 `evaluate()` 的 `permute(2,0,1)` 证明 `raw[0]` 为 HWC；`stack(..., dim=-1)` 已显式最后一维。
3. 新 dispatch 旗标不能用 `previous.get(k) != record.get(k)`：缺键/显式 null 必须两侧对称记为历史 False，否则旧 gsplat resume 被误拒。
4. Postshot 本就不允许 `train_package --resume`（走已存项目）；null 身份只影响 gsplat 侧对称比较。
5. 检查点同时写 `photometric*` 与旧名 `exposure*`，加载优先新名。

## [S1] Problem

室内/古建 3DGS 常见「杂云」（floater）：

1. 多视角 ISP 光度差被当成残差留在空间中（视频 BV1siXrBsE2t / BilARF）。
2. 点云扩增在纯色墙面等处把远处平面错判成凸起团状物；需用初始化稀疏点深度把结果压回正确平面（视频称 sparse depth，gsplat 一系已采用）。

现状：`native_train.PanoramaExposure` 仅做全景组增益/偏置；`prepare_segment` 只导出 `points.npz` 初始化点，未导出逐视图稀疏深度锚；采集路径无表盘走位等经验文档。

约束：兼容 schema 1–5；不改默认后端（仍为 Postshot Splat ADC）；不改 QA/失败语义；不自动全量训练；不外发图像；历史 Run 恢复语义不变。

## [S2] Design

### [S2.1] Bilateral Grid 光度补偿（原生 gsplat）

- 新增 `src/gsdb/photometric.py`：`BilateralGrid` 模块。按全景时间组（与现 `PanoramaExposure` 同组）维护粗网格局部仿射颜色变换（强度维 × 空间维），近恒等 + 空间平滑正则。
- `native_train.train(..., use_bilateral_grid=True, use_sparse_depth=True)`：
  - `photo_comp=False`：恒等，不用网格也不用 Exposure。
  - `photo_comp=True, use_bilateral_grid=True`（默认）：BilateralGrid。
  - `photo_comp=True, use_bilateral_grid=False`：保留 `PanoramaExposure`（旧补偿）。
- `config.json` / `training.json` / `dispatch.json` 记录 `use_bilateral_grid`、`use_sparse_depth`、`photo_comp`；同目录 resume 要求配置一致。
- Postshot 路径不改算法（其 `photo_comp` 语义保持）。

### [S2.2] Sparse Depth 导出与正则

- `prepare_segment` 在写出 `points.npz` 的同时写出 `sparse_depth.npz`：
  - 仅训练 split、未遮罩观测；来自同一 reservoir 保留点。
  - 字段：`row_index`（`meta.images` 下标）、`u`、`v`、`depth`（COLMAP 相机系 z）、`point_id`。
  - 写入 `dataset.json` 的 `files` 哈希；`meta` 增加 `sparse_depth_observations` 计数。
- 旧包无 `sparse_depth.npz` 仍可通过 `validate_package`；`use_sparse_depth=True` 且文件缺失时记录 `sparse_depth='unavailable'` 并跳过该损失（不失败）。
- 训练：在稀疏像素上对期望深度（gsplat `RGB+ED`）与锚深度做稳健 L1；权重可配置常量，默认温和，不替代光度损失。

### [S2.3] 入口与契约

- `training.train_package` / CLI `train` / `native_train` CLI / `studio_runtime`：
  - 新选项 `use_bilateral_grid`、`use_sparse_depth`，**默认 True**；`--no-use-bilateral-grid` / `--no-use-sparse-depth` 关闭。
  - `dispatch.json` 身份字段包含上述开关；resume 不得变更。
- 不改 `compare-backends-v5` 默认矩阵（仍仅 `photo_comp` 开关）；不把 gsplat 提为默认后端。

### [S2.4] 古建室内采集 SOP

- 新增 `docs/CAPTURE-SOP.md`：Insta360 360 适配「表盘走位法」（向心环绕定框架 → 井字补深 → 离心环绕扫墙 → 匾额/对联/窗棂细节多角度慢速补拍）、匀速约 1 m/s、转弯减速、尽量高快门减运动模糊、与 schema 6 Capture/分段衔接。
- `docs/CURRENT-WORKFLOW.md` §2 增加指向该 SOP 的链接；不复制粘贴全文、不改命令。

### [S2.5] 测试边界

- CPU 单测（无 CUDA/真实 Capture）：BilateralGrid 前向形状/恒等初值/正则非负；sparse_depth 导出形状与深度符号；train_package 命令行含默认/关闭旗标；配置哈希含新开关。
- 不在本切片启动真实重建或 GPU 训练。

## [S3] Out of Scope

杂云自动检测/删除、SuperSplat 后处理、PPISP、LiDAR 深度源、Postshot 内部算法、默认后端变更、全场景自动训练、外部 vision QA、Studio UI 路线、worktree。

## Tasks

- [x] T1: 实现 `photometric.BilateralGrid` 并接入 `native_train` 补偿路径 — acceptance: 默认走 BilateralGrid；`--no-use-bilateral-grid` 走 PanoramaExposure；`--no-photo-comp` 恒等 (covers: S2.1)
- [x] T2: `prepare_segment` 导出 `sparse_depth.npz` 并纳入 package 哈希 — acceptance: 新包含逐观测 u/v/depth；旧包仍 validate (covers: S2.2)
- [x] T3: `native_train` 稀疏深度损失与配置/恢复字段 — acceptance: 有锚则计损失；无锚记录 unavailable；resume 配置一致 (covers: S2.2)
- [x] T4: CLI / `train_package` / studio 传递新开关（默认开）— acceptance: dispatch 含旗标；身份校验拒绝变更 (covers: S2.3)
- [x] T5: 撰写 `docs/CAPTURE-SOP.md` 并在 CURRENT-WORKFLOW §2 挂链 — acceptance: SOP 含表盘走位/快门/匀速/细节补拍；手册可点链 (covers: S2.4)
- [x] T6: CPU 单测覆盖 photometric / sparse_depth / 命令构造 — acceptance: 新测试通过；不依赖 GPU (covers: S2.5)
