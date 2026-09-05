# Yunxiu 3DGS：环境与配置交接

用途：供后续 Codex/Claude 会话接手 `yunxiu-20260822` 的完整视频任务。  
更新日期：2026-08-26。此文件不包含 API Key、令牌、Cookie 或其他凭据。

关联规划：[TASK_PLAN_20260826.md](TASK_PLAN_20260826.md)。该规划固定 **100,000** 次训练迭代，对完整 427 秒视频保持原有抽帧密度，并**按视图数自动放大致密化窗口**（`stop_split_at` 15,000 → 71,222），以免 5,128 张图在默认窗口下每张只被采样 2.9 次、几何建不起来。

## 当前任务状态

- 状态：`planned`；尚未创建 location/scene/capture/run 清单，尚未执行导入或训练。
- 当前导出视频：`scenes/VID_20260823_104518_00_014.mp4`
- 视频大小：3.10 GiB；时长：427 秒（00:07:07）。
- 原始素材：`E:\Photo\PhotosRaw\2026\2026-08-23\Insta360\VID_20260823_104518_00_014.insv`
- 当前 MP4 直接置于 `scenes/` 根目录；流水线运行前，必须将其放入新 scene 的 `inputs/stitched/` 下（保留原始文件，未经用户许可不要删除）。

## 路径与启动方式

| 项目 | Windows | WSL |
| --- | --- | --- |
| 仓库根目录 | `D:\Project\3DGS` | `/mnt/d/Project/3DGS` |
| 位置目录 | `D:\Project\3DGS\locations\yunxiu-20260822` | `/mnt/d/Project/3DGS/locations/yunxiu-20260822` |
| Conda 可执行文件 | — | `/home/astroite/miniforge3/bin/conda` |
| Conda 环境 | `3dgs` | `3dgs` |

从 PowerShell 执行任何 gsdb 命令时，使用这一形式，确保是在正确的 WSL 工作目录和 Conda 环境中运行：

```powershell
wsl.exe -d Ubuntu-22.04 --cd /mnt/d/Project/3DGS -- `
  /home/astroite/miniforge3/bin/conda run -n 3dgs --no-capture-output gsdb <command>
```

不要直接在普通 WSL shell 中调用 `ffmpeg` 或 `ffprobe`；它们位于 `3dgs` Conda 环境内。

## 已验证环境

在 2026-08-26 通过 `gsdb doctor --minimum-free-gib 50`。注意磁盘余量检查现在针对**工作目录所在的文件系统**，而工作目录已改到 WSL 的 ext4 磁盘（`GSDB_SCRATCH_ROOT`，默认 `$HOME/gsdb-scratch`），不再是 D 盘：

| 组件 | 已验证值 |
| --- | --- |
| WSL 发行版 | Ubuntu-22.04 |
| Python | 3.10.12 |
| FFmpeg / FFprobe | 6.1.1 |
| COLMAP | 3.8，CUDA SIFT 通过（GPU 0） |
| PyTorch | 2.1.2，CUDA 11.8 |
| gsplat | 1.4.0，rasterization/backward 通过 |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER，16,376 MiB 显存 |
| WSL 可用内存 | 62.8 GiB |
| D 盘可用空间 | 278.7 GiB（只需容纳发布产物，约 1.5–3 GiB） |
| WSL ext4 可用空间 | 903 GiB（工作目录与 checkpoint 都在这里，需 ≥ 80 GiB） |
| 人像分割 | torchvision Mask R-CNN / COCO_V1；首次 mask 时可能下载或使用已缓存的权重 |

已验证的可执行文件均在：`/home/astroite/miniforge3/envs/3dgs/bin/`，包括 `ffmpeg`、`ffprobe`、`colmap`、`ns-train` 和 `gsdb`。

## 场景清单与媒体摆放

流水线要求的结构如下。`<scene-id>` 和 `<capture-id>` 必须是稳定 ASCII slug，且后续命令始终使用同一对值。

```text
locations/yunxiu-20260822/
  location.yaml
  scenes/
    <scene-id>/
      scene.yaml
      captures/<capture-id>.yaml
      inputs/stitched/VID_20260823_104518_00_014.mp4
      work/
      runs/
      exports/
      qa/
```

当前地点还没有 `location.yaml`。先初始化地点和 scene；显示名称由执行会话按用户确认的名称填写：

```powershell
# 以下命令中的名称和 <scene-id> 是待确认占位符，不要原样照抄到正式清单。
gsdb location init yunxiu-20260822 --name "<location display name>" --capture-date 2026-08-23
gsdb scene init yunxiu-20260822 <scene-id> --name "<scene display name>"
```

`gsdb` 没有 capture init 子命令；需以 YAML 创建 `captures/<capture-id>.yaml`。可参照已成功运行的：

```text
locations/yanguan-ancient-town-20260822/scenes/night-walk-4k/captures/capture-009-4k.yaml
```

capture 清单至少要正确记录：外部 INSV 的 Windows 路径、`immutable: true`、相对于 scene 的 stitched MP4 路径、相机/Studio 导出参数，以及完整选择范围 `start_seconds: 0.0`、`end_seconds: 427.0`。原始与拼接视频的 SHA-256、尺寸、FPS、时长及编解码器由 `gsdb ingest` 校验并写回；不要手工伪造它们。

## 本次完整视频运行参数

基准是已验证的 90 秒优化流程：270 候选全景帧、135 个模糊感知入重建帧、每个全景帧投影 8 张透视图、100k `splatfacto-big` 训练。

| 参数 | 本次固定值 |
| --- | ---: |
| 完整选择时长 | 427 秒 |
| `target_frames` | 1,281（3 帧/秒） |
| `primary_frames` | 641（约 1.5 帧/秒） |
| 主投影 | 8 张/全景帧，120°，2048 px，底部裁切 0.20 |
| 主路径透视图总数 | 5,128 |
| 遮罩 | CUDA，score=0.25，probability=0.50，gamma=0.75，dilation=24 px，QA 样本=16 |
| 重建 | GPU SIFT、固定内参、rig、注册阈值 0.70 |
| 训练 | `splatfacto-big`，**100,000 iterations**，`downscale_factor=1`，CPU uint8 图像缓存 |
| 致密化调度 | **自动按视图数推导**，无需传参：`warmup=2374`、`stop_screen_size_at=18993`、`stop_split_at=71222` |
| 训练仪表 | 每 10,000 步保留一个 checkpoint（共 10 个），`--vis tensorboard` 记录评估曲线 |
| 导出 | Y-up PLY（方向由素材重力推导）+ 外围高斯剔除（距离因子 3.0、尺度因子 1.0） |

为保持 fallback 的抽帧密度，若需要在新 run 中显式配置 fallback，应使用 `fallback_frames: 854`（2 帧/秒）与原有的 14 投影/110°/1746 px 参数。fallback 不是正常预算的一部分；它会额外耗费大量时间。

## 命令顺序模板

在 WSL 的 `/mnt/d/Project/3DGS` 中，依次执行。先把上文的占位符替换为已创建清单的真实 slug。

```bash
# 0. 先用缩减档确认整条链路在这个新地点结构下能跑通（约 20 分钟，落在自己的 RunId 下）
gsdb preprocess yunxiu-20260822 <scene-id> <capture-id> --smoke

# 1. 运行环境检查（建议在开始前和恢复前各执行一次）
gsdb doctor --minimum-free-gib 50

# 2. 验证并指纹化已放入 inputs/stitched 的 MP4
gsdb ingest yunxiu-20260822 <scene-id> <capture-id>

# 3. 创建 run，并执行完整视频的预处理
gsdb preprocess yunxiu-20260822 <scene-id> <capture-id> \
  --target-frames 1281 \
  --primary-frames 641 \
  --primary-fov 120 \
  --primary-projection-size 2048 \
  --fallback-frames 854 \
  --fallback-fov 110 \
  --fallback-projection-size 1746 \
  --mask-device cuda \
  --mask-score-threshold 0.25 \
  --mask-probability-threshold 0.50 \
  --mask-gamma 0.75 \
  --mask-dilation-pixels 24 \
  --mask-qa-sample-count 16 \
  --train-iterations 100000

# 4. 记录 preprocess 输出的 RUN_ID，再继续
gsdb mask yunxiu-20260822 <scene-id> <RUN_ID>
gsdb reconstruct yunxiu-20260822 <scene-id> <RUN_ID>
gsdb train yunxiu-20260822 <scene-id> <RUN_ID>
gsdb export yunxiu-20260822 <scene-id> <RUN_ID> --version <unused-vNNN>
gsdb qa report yunxiu-20260822 <scene-id> <RUN_ID>

# 导出默认会剔除外围高斯（距离因子 3.0、尺度因子 1.0）并按素材重力校正朝向。
# 若人工检查发现远景建筑被误删，改用更大的因子重发一个版本即可，不需要重训：
#   gsdb export ... --version <next-vNNN> --cull-distance-factor 5.0
# 完全关闭剔除用 --no-cull。
```

中断后沿用同一个 `RUN_ID` 并为相应阶段追加 `--resume`；不要创建新 run 来“接着跑”，否则已完成的工作不会复用。导出版本目录已存在时使用新的 `vNNN`，或按日志提示在同一导出阶段使用 `--resume`。

## 运行预算与监控

预期总时长为 **9–14 小时**：预处理 3–5 分钟、遮罩 45–55 分钟、主重建 3.5–7 小时（其中 mapping 2.7–6 小时，是唯一的大不确定项）、100k 训练 4.2–6 小时、导出/测试/LFS 推送 10–20 分钟。预留 **14 小时**连续 GPU 窗口。详细依据见规划文档的时间预算一节。

- 基准运行工作目录峰值为 4.2 GiB；按帧数估算本次约 22 GiB。**另需为 10 个训练 checkpoint 预留 20–45 GiB**（基准单个 1.08 GiB / 878,467 高斯，本次因致密化窗口放大预计更多）。ext4 上 ≥ 80 GiB 空闲是最低操作余量。
- 基准训练 GPU 峰值为 8,847 MiB / 16,376 MiB。**本次致密化窗口从 15,000 放大到 71,222 步，高斯数量会显著多于基准的 878,467，显存是本次最大风险。**建议在第一个 checkpoint（10,000 步）落盘后就查看显存曲线与高斯数量，不要等到几小时后才发现。不要并行运行其他 GPU 任务。
- `gsdb train` 会在 OOM 时按既定逻辑以 `downscale_factor: 2` 重试。保留失败日志与 run manifest；不要删除失败工作目录。
- 100k iterations 固定不变，每张图的**精修**覆盖约为 90 秒基准的 21%；但**致密化**覆盖已由自动推导的调度补回到与基准相同的 13.9 次/张。若后段区域仍未收敛，应新建 run 提高迭代数或改为拆段，不要修改已完成 run 的配置。

## 验收、交付与敏感信息

1. `ingest` 必须确认 MP4 是单一 2:1 等距柱状全景流，并将哈希写入 capture 清单。
2. 遮罩 QA、主重建注册率/相机中心分布检查均须通过；主重建失败再评估 fallback。
3. 导出应包含 `artifact.yaml`、`splat-yup.ply`、预览视频、缩略图，以及三个能互相对齐的坐标文件：`transforms.json`（与 PLY 同坐标系）、`transforms-colmap.json`（重建原始）、`dataparser_transforms.json`。
   QA 报告中还需确认：rig 重力偏差 p95 < 5°、发布后相机高度跨度占比 < 10%、位姿归一化峰值 ≈ 1.0、外围高斯剔除比例合理（基准 2.05%）。
4. `qa report` 只生成技术审查材料。**人工必须在查看器中载入 `splat-yup.ply` 与同目录的 `transforms.json`**，确认：场景正立、相机落在场景内成连续轨迹、远处浮点球消失、街道本体与远景建筑没有被误删。指标不能替代这一步。确认后用 `gsdb qa approve ... --notes "..."` 记录，才提交与推送。
5. PLY 使用 Git LFS。提交前运行测试，检查 `git status`、`git lfs status`，再执行 `git add`、`git commit`、`git push`。

DeepSeek 视觉 QA 默认为关闭；只有用户明确要求且安全提供了 `DEEPSEEK_API_KEY` 时才使用 `--vision-qa`。绝不把密钥写入 YAML、Markdown、Git 或命令日志。
