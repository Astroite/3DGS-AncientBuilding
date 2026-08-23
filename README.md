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
- PLY、检查点、视频、COLMAP 数据库、帧和 SQLite 都不进入 Git，也不使用 Git LFS。

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
- CPU COLMAP 3.8
- Nerfstudio `758ea1918e082aa44776009d8e755c2f3a88d2ee`

`conda-lock.yml` 以 Ubuntu 22.04 的 glibc 2.35、linux-64 和 CUDA 11.8
虚拟平台生成；`virtual-packages.yml` 是这组平台边界的可审计输入。首次
`doctor` 会编译 gsplat CUDA 扩展，可能需要数分钟，后续运行会命中缓存。

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

所有 Windows 命令通过 `gsdb.ps1` 进入固定 WSL 环境。当前只完成准备，**不要在导出完成并确认电脑可长时间运行之前执行下列处理命令**。

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
.\gsdb.ps1 export      yanguan-ancient-town-20260822 night-walk-4k $RunId --version v001
.\gsdb.ps1 qa report   yanguan-ancient-town-20260822 night-walk-4k $RunId
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

创建不可静默覆盖的运行清单；均匀抽取 270 个全景 JPEG，记录时间戳、拉普拉斯模糊度、平均亮度、黑位占比和高光裁切占比。预处理前要求保留 20 GiB 余量，并额外估算中间文件空间。

当前 009 试点只选择 0–90 秒，因此 270 帧仍约等于每秒 3 帧。完整 293.86 秒素材后续应按空间连续的街段或走廊拆成多个场景，而不是降低抽帧密度后硬塞进一个模型。

### mask

将 270 张全景投影为 270×8 张透视图，然后使用 Torchvision Mask R-CNN 在本地识别人像。人物概率遮罩经过闭运算和 24 px 边缘扩张，保存为与图像同尺寸的二值 PNG：白色可用、黑色忽略。`images_2/images_4` 与 `masks_2/masks_4` 使用相同文件名和最近邻遮罩缩放。

确定性 QA 会核对图片/遮罩一一对应、尺寸、二值范围和最大遮挡比例，并生成抽样联系表。首次调用会下载并缓存官方 Mask R-CNN 权重；代码导入和 `doctor` 不会提前下载。

可选的 MiMo v2.5 多模态门禁默认关闭。只有在服务方确认所用套餐/端点允许该自动化用途，并确认抽样画面中游客人像的外发与留存边界后，才应在新运行中传入 `--vision-qa`；API key 与 base URL 只通过 `MIMO_API_KEY`、`MIMO_BASE_URL` 环境变量提供，不写入清单或日志。MiMo 只审核人物遮罩，不接管本地处理任务，也不替代最终 3DGS 资产审核。

### reconstruct

COLMAP 3.8 以 CPU 模式运行，逐图读取 `mask_path`；人物黑区既不产生 SIFT 特征，也不会进入后续训练像素监督。透视图使用与投影 FOV 对应的 PINHOLE 初始内参，而不是依赖缺失 EXIF 的默认焦距。转换后，每个注册帧的 `mask_path` 被写入 Nerfstudio `transforms.json`。

首跑使用 270×8 个透视视图、20% 底部裁剪、sequential matching、两个下采样层级。如果注册率或最大连通模型覆盖低于 70%，只进行一次确定性的降级尝试：每个时间桶选相对清晰帧，共 180 帧，转换为 180×14 个视图、裁底 15%，并生成自己的完整遮罩集。第二次仍失败就停止，不继续无限调参。

### 入镜人物与拍摄者

- 底部裁剪先去掉自拍杆、手臂和大部分位于天底的拍摄者身体；如果身体仍伸出裁剪区，则给全部透视视图增加固定天底遮罩。
- 游客属于随时间移动的瞬态物体。正式流程应在透视视图生成后做人像分割，适度扩张遮罩边缘，并为每张图保存同尺寸黑白遮罩。
- 同一组遮罩必须同时用于 COLMAP 特征提取和 Nerfstudio 训练：黑色人物区域不提特征、也不参与像素监督，避免错误相机匹配和 3DGS 漂浮人影。
- 当前 009 试跑默认就是动态人像遮罩版本；关闭遮罩或改变阈值会形成不同配置哈希和运行 ID，禁止覆盖既有运行。

### train

运行 Splatfacto 30,000 步。首次日志明确出现 CUDA OOM 时只用 `downscale-factor=2` 重试一次；其他错误不自动换算法或吞掉。

### export / qa / catalog

导出 Gaussian PLY、3 秒 3DGS 预览、缩略图和坐标变换。QA 先进入 `needs_review`，只有人工命令可以改为 `accepted` 或 `rejected`。SQLite 使用临时数据库构建并原子替换，随时可从 YAML 重建。

## 90 秒试跑耗时预估

按本机 Ryzen 9 5950X（16 核）、RTX 4070 Ti SUPER 和 CPU COLMAP 做无人值守规划，主流程保守按 **5–15 小时**；如果触发唯一一次 180×14 降级重建，整体按 **10–24 小时**。夜景特征不足时 COLMAP 波动最大，这不是进度承诺。建议出门前接通电源、关闭自动睡眠；当前 D 盘约 294 GiB 可用，足够按 30–100 GiB 峰值规划并保留 20 GiB 硬余量。脚本不会主动删除中间数据。

## 断点和清理

- 已成功阶段再次执行会拒绝覆盖；`--resume` 只复用同一配置和已完成文件。
- 人像遮罩可逐文件续作；COLMAP 外部命令若中途失败，`--resume` 会创建新的 `attempt-NNN`，保留失败数据库和日志，不猜测或覆盖半成品。
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
