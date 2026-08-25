# GSDB：古建筑 360 视频到 3DGS 参考库

这个仓库按“地点 → 场景 → 采集 → 运行 → 产物”管理全景视频重建。地点是资料目录，场景才是一次 COLMAP/Splatfacto 可以独立处理的空间单元；长距离古镇素材应拆成有连续视觉重叠的院落、街段或走廊，不能把几十分钟视频直接塞进一个模型。

当前试点：

- 地点：`yanguan-ancient-town-20260822`（盐官古镇）
- 场景：`night-walk-4k`
- 原始素材：`E:\Photo\PhotosRaw\2026\2026-08-23\Insta360\VID_20260822_211338_00_009.insv`
- 选段：完整素材约 293.86 秒，首跑使用前 90 秒
- 目标：证明处理闭环可行；当前夜景素材不是正式质量标杆

原先的 `night-pilot-8k` / `capture-004-8k` 清单继续保留，作为另一条独立采集血缘，不被 009 覆盖。

## 数据边界

- E 盘原始 `.insv` 永远按只读来源处理，CLI 不解码、不复制、不修改它。
- Insta360 Studio 手工拼接得到的标准 2:1 MP4 和全部派生数据放在地点子项目内，由 Git 忽略。
- YAML 是权威数据源；`catalog/catalog.sqlite` 只能通过 `gsdb catalog build` 重建。
- 帧、检查点、COLMAP 数据库、训练输出和 Z-up 中间 PLY 不进入 Git；人工预览用的版本化 Y-up PLY 与 MP4 才通过 Git LFS 发布。

## 1. 在 Insta360 Studio 导出试点输入

不要使用目前项目时间线右上角的 16:9“项目导出”。从原始 360 媒体入口选择 `VID_20260822_211338_00_009.insv`，导出：

- 导出为“全景视频”；Studio 不单独显示宽高比，但 `3840×1920` 本身就是 2:1 等距柱状全景
- 分辨率选择“匹配原素材 - 3840×1920”，不要上采样到 5.7K 或 8K
- 匹配原素材帧率 29.97 fps
- H.265、原件码率约 62 Mbps、Rec.709 10bit
- 保留正常拼接/防抖
- 不做视角重构、关键帧、AI 降噪、锐化或 16:9 裁切

输出文件必须保存为：

```text
D:\Project\3DGS\locations\yanguan-ancient-town-20260822\scenes\night-walk-4k\inputs\stitched\capture-009-4k-equirect.mp4
```

没有这个标准 MP4 时，`ingest` 会明确失败；它不会尝试把双鱼眼 `.insv` 当成拼接全景。

## 2. 安装 WSL2 环境

已有的 Ubuntu 22.04 和 NVIDIA 驱动可直接使用。首次安装：

```powershell
wsl.exe -d Ubuntu-22.04 -- bash /mnt/d/Project/3DGS/scripts/bootstrap-wsl.sh
```

脚本会在 WSL 用户目录安装 Miniforge（如尚未安装）、创建 `3dgs` 环境、安装固定依赖、应用受保护的 Nerfstudio 夜景投影钳制补丁，并运行 `gsdb doctor`。项目使用环境内的 CUDA 11.8；Windows 已安装的 CUDA 12.9 不参与依赖解析。

固定的关键版本：

- Python 3.10.12
- PyTorch 2.1.2 / Torchvision 0.16.2 / CUDA 11.8
- gsplat 1.4.0
- FFmpeg 6.1
- CUDA COLMAP `3.8=gpuhe53869c_110`（同时锁定 `cudatoolkit=11.8`）
- Nerfstudio `758ea1918e082aa44776009d8e755c2f3a88d2ee`

`conda-lock.yml` 以 Ubuntu 22.04 的 glibc 2.35、linux-64 和 CUDA 11.8
虚拟平台生成；`virtual-packages.yml` 是这组平台边界的可审计输入。首次
`doctor` 会实际运行 CUDA SIFT 和 gsplat 反向传播，并要求 WSL 至少 28 GiB 内存；首次编译可能需要数分钟，后续运行会命中缓存。

重新生成锁时使用 conda-lock 4.0.2，并先运行
`python scripts/patch-conda-lock-manylinux.py`。这个受文本校验保护的小补丁只补充
该版本遗漏的 manylinux 2.29–2.35 wheel 标签；否则 Open3D/PyMeshLab 的 Ubuntu
22.04 wheel 会被锁生成器误判为不可安装。

```bash
conda install -n base -c conda-forge conda-lock=4.0.2
python scripts/patch-conda-lock-manylinux.py
conda-lock lock --conda "$(command -v conda)" \
  --virtual-package-spec virtual-packages.yml \
  --file environment.yml --platform linux-64 --lockfile conda-lock.yml
```

全新环境由 bootstrap 直接安装锁；检测到已有 `3dgs` 环境时则根据
`environment.yml` 做幂等更新，随后重新校验并应用 Nerfstudio 补丁。

## 3. 执行试点

所有 Windows 命令通过 `gsdb.ps1` 进入固定 WSL 环境。

明早从 PowerShell 启动整条 009 遮罩试跑：

```powershell
Set-Location D:\Project\3DGS
.\scripts\run-009-demo.ps1
```

这个脚本依次运行环境检查、输入校验、90 秒抽帧、透视投影、人物分割、带遮罩 COLMAP、Splatfacto、导出、QA 报告和目录重建。它不会删除中间文件，也不会自动把最终资产标为 `accepted`。更完整的启动和恢复说明见 [009 遮罩试跑手册](docs/RUNBOOK-009-MASKED.md)。

需要逐阶段观察时，先执行：

```powershell
.\gsdb.ps1 doctor

.\gsdb.ps1 ingest `
  yanguan-ancient-town-20260822 night-walk-4k capture-009-4k

.\gsdb.ps1 preprocess `
  yanguan-ancient-town-20260822 night-walk-4k capture-009-4k
```

`preprocess` 会打印新运行 ID，例如 `20260823T120000Z-1a2b3c4d`。后续命令均使用该 ID：

```powershell
$RunId = '<上一步输出的运行 ID>'

.\gsdb.ps1 mask        yanguan-ancient-town-20260822 night-walk-4k $RunId
.\gsdb.ps1 reconstruct yanguan-ancient-town-20260822 night-walk-4k $RunId
.\gsdb.ps1 train       yanguan-ancient-town-20260822 night-walk-4k $RunId
.\gsdb.ps1 export      yanguan-ancient-town-20260822 night-walk-4k $RunId --version v002
.\gsdb.ps1 qa report   yanguan-ancient-town-20260822 night-walk-4k $RunId --baseline-run-id 20260824T022046Z-5679786b
.\gsdb.ps1 catalog build
```

人工查看 PLY、预览和 QA 清单后：

```powershell
.\gsdb.ps1 qa approve yanguan-ancient-town-20260822 night-walk-4k $RunId --notes '人工检查通过'
# 或
.\gsdb.ps1 qa reject  yanguan-ancient-town-20260822 night-walk-4k $RunId --notes '说明拒绝原因'
```

## 阶段行为

### ingest

用 FFprobe 检查单视频流、2:1 比例、时长与解码元数据，计算 SHA-256 并写回采集清单。输入必须位于本项目内；CLI 不会偷偷复制桌面或 E 盘上的大文件。

### preprocess

创建 schema v2、不可静默覆盖的运行清单；均匀抽取并分析 270 个候选全景 JPEG，再按 135 个时间桶各取最清晰的一帧作为主重建输入。候选数、选中数和清晰度提升都会进入指标。历史 schema v1 仍按原形加载和验哈希。

当前 009 试点只选择 0–90 秒，因此 270 帧仍约等于每秒 3 帧。完整 293.86 秒素材后续应按空间连续的街段或走廊拆成多个场景，而不是降低抽帧密度后硬塞进一个模型。

### mask

将 135 张清晰度优选全景显式投影为 135×8 张 `2048×2048`、120° 透视图，然后使用 Torchvision Mask R-CNN 在本地识别人像。图像按 `view_XX/frame_NNNNNN.jpg` 分相机目录，遮罩与金字塔完整镜像相对路径；白色可用、黑色忽略。

确定性 QA 会核对图片/遮罩一一对应、尺寸、二值范围和最大遮挡比例，并从 16 个透视视图生成两张联系表。首次调用会下载并缓存官方 Mask R-CNN 权重；代码导入和 `doctor` 不会提前下载。

可选的 DeepSeek `deepseek-v4-flash-vision-exp` 多模态门禁默认关闭。决定把抽样画面发送到 DeepSeek 后，可在新运行中传入 `--vision-qa`；API key 只通过 `DEEPSEEK_API_KEY` 环境变量提供，不写入清单或日志，Base URL 固定为官方 `https://api.deepseek.com`。联系表以 Base64 JPEG、`detail: original` 发送，并在本地预检官方的 32 MiB 单图与 48 MiB 请求体限制。DeepSeek 只审核人物遮罩，不接管本地处理任务，也不替代最终 3DGS 资产审核。接口格式见 [DeepSeek 图像理解文档](https://api-docs.deepseek.com/zh-cn/guides/vision/)。

### reconstruct

COLMAP 3.8 使用 GPU SIFT，`doctor` 不允许静默退回 CPU。每个 `view_XX/` 固定为独立 PINHOLE 相机，焦距/主点在 mapper 与 rig BA 中全部锁定。同一全景的 8 个视图再通过已知纯旋转、零平移的 rig 约束合并光心，后验 p95 光心散布不得超过中位帧间基线的 0.1%。

匹配分两遍。第一遍 `sequential_matcher` 负责时间方向：COLMAP 3.8 按**图片名字**排序，所以它实际做的是每个 `view_XX/` 内部的时间链，视图之间没有任何 pair。只有这一遍时，mapper 会按视图目录把场景切成 8 个互不相连的组件。第二遍 `matches_importer` 补上缺的连接：显式列出同一帧内全部视图两两组合（135 帧 × 28 对 = 3780 个 pair），把 8 条链缝成一个模型。这些 pair 共用光心、基线为零，两视图几何是单应而非本质矩阵，无法自行三角化，但它们把特征 track 连起来，由前后有真实基线的帧完成三角化；不重叠的对面视图由 COLMAP 自己的几何验证剔除。两遍各有独立完成标记，可分别续作。

主流程使用 135×8 个 2048²/120° 透视视图、20% 底部裁剪和两个下采样层级。如果注册率或最大连通模型覆盖低于 70%，只进行一次确定性的降级尝试：从原始 270 候选帧按桶选 180 帧，转换为 180×14 个 1746²/110° 视图、裁底 15%，并生成自己的完整遮罩与 rig。第二次仍失败就停止。

### 入镜人物与拍摄者

- 底部裁剪先去掉自拍杆、手臂和大部分位于天底的拍摄者身体；如果身体仍伸出裁剪区，则给全部透视视图增加固定天底遮罩。
- 游客属于随时间移动的瞬态物体。正式流程应在透视视图生成后做人像分割，适度扩张遮罩边缘，并为每张图保存同尺寸黑白遮罩。
- 同一组遮罩必须同时用于 COLMAP 特征提取和 Nerfstudio 训练：黑色人物区域不提特征、也不参与像素监督，避免错误相机匹配和 3DGS 漂浮人影。
- 当前 009 试跑默认就是动态人像遮罩版本；关闭遮罩或改变阈值会形成不同配置哈希和运行 ID，禁止覆盖既有运行。

### train

运行 `splatfacto-big` 100,000 步，CPU/uint8 图像缓存、scale regularization、classic rasterizer、bilateral grid、`SO3xR3` 相机优化和 `downscale-factor=1` 均显式入配置。首次日志明确出现 CUDA OOM 时只用 `downscale-factor=2` 重试一次；不启用当前版本未暴露的 MCMC 参数。

### export / qa / catalog

Nerfstudio 的 Z-up PLY 只留在忽略的工作区；发布目录只包含旋转位置、法线、wxyz 四元数与 1–3 阶实 SH 后的 `splat-yup.ply`、预览、缩略图、`transforms.json` 和清单。QA 对比 v001/v002 指标；低于 2M 高斯只告警，不自动追加训练，状态始终先进入 `needs_review`。

## 90 秒试跑耗时预估

按本机 RTX 4070 Ti SUPER 16 GiB、135×8 @2048² 与 CUDA COLMAP 做无人值守规划，主流程先按 **2.5–8 小时**，触发 180×14 降级时按 **5–14 小时**；夜景匹配与 100k 训练仍可能波动。这不是进度承诺。建议接通电源并关闭休眠；程序保留 20 GiB 硬余量，脚本不会主动删除中间数据。

只想确认链路能跑通、不看质量时用 `gsdb preprocess ... --smoke`：候选帧 80、主流程 40×8、投影边长 1024、训练 5000 步。所有值都是上限，显式传更小的仍生效；配置哈希不同，因此它落在自己的 RunId 下，不会覆盖或冒充正式运行。**smoke 产物不能用于质量判断。**

## 中间数据存放位置

`gsdb.ps1` 会把运行工作目录指向 WSL 自己的 ext4 磁盘，默认 `$HOME/gsdb-scratch`，由环境变量 `GSDB_SCRATCH_ROOT` 控制。`work/<run-id>` 在项目内仍然存在，只是变成指向 ext4 的符号链接，因此清单里记录的路径、`--resume` 和产物发布都不受影响。

这样做的原因是 `/mnt/d` 是 9p 挂载（`msize=65536`）：实测读一张 2048² JPEG 在 9p 上 50.6 ms，在 ext4 上 12.7 ms，而后者里约 12 ms 是纯解码 CPU 时间。一次重建要读写上万张图，差距完全体现在墙钟上。

- 想留在项目内，把 `GSDB_SCRATCH_ROOT` 设为空即可。
- 重定向启用前就已经存在的 `work/<run-id>` 实体目录会继续原地使用，不会被搬走或改写。
- 磁盘余量检查针对工作目录所在的文件系统，不再是场景目录所在的盘。

## 断点和清理

- 已成功阶段再次执行会拒绝覆盖；`--resume` 只复用同一配置和已完成文件。
- 人像遮罩可逐文件续作；COLMAP feature/matching/cross-view/mapping/rig 各有完成标记，安全前置结果会复用；不完整 sparse/rig 输出不会被覆盖。
- 缓存图像集的续跑校验只读文件头尺寸，不再整张解码：所有中间图都经由"写临时文件 → fsync → 原子改名"落盘，能出现在最终文件名下的内容一定是完整的，逐张重新解码只是重复付出解码成本。金字塔每一层写完并整体校验后会留下 `.pyramid-complete-*` 标记；没有标记的旧数据仍走完整解码校验。
- 配置变化必须创建新运行，失败运行不会覆盖历史成功产物。
- `gsdb clean <location> <scene>` 只显示可清理容量，v1 永远不删除文件。

## 开发测试

轻量开发环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
```

完整 Nerfstudio 投影测试只会在 WSL `3dgs` 环境中运行；缺少 Torch/Nerfstudio 时相应用例会跳过。

## 已知边界

- 当前输出是引擎中立参考资产，不包含 Unreal/Unity 适配。
- Splatfacto 输出默认不是米制尺度；精确尺寸需后续加入控制点或测量数据。
- 首版不做网格、全镇模型拼接、网页浏览应用或自动删除。
- 夜景拖影、曝光变化、漂浮噪点和几何断裂必须留在 QA 报告中，不能靠状态字段掩盖。
