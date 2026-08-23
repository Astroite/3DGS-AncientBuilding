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

所有 Windows 命令通过 `gsdb.ps1` 进入固定 WSL 环境：

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

### reconstruct

首跑使用 270×8 个透视视图、20% 底部裁剪、sequential matching、两个下采样层级。如果注册率或最大连通模型覆盖低于 70%，只进行一次确定性的降级尝试：每个时间桶选相对清晰帧，共 180 帧，转换为 180×14 个视图并裁底 15%。第二次仍失败就停止，不继续无限调参。

### 入镜人物与拍摄者

- 底部裁剪先去掉自拍杆、手臂和大部分位于天底的拍摄者身体；如果身体仍伸出裁剪区，则给全部透视视图增加固定天底遮罩。
- 游客属于随时间移动的瞬态物体。正式流程应在透视视图生成后做人像分割，适度扩张遮罩边缘，并为每张图保存同尺寸黑白遮罩。
- 同一组遮罩必须同时用于 COLMAP 特征提取和 Nerfstudio 训练：黑色人物区域不提特征、也不参与像素监督，避免错误相机匹配和 3DGS 漂浮人影。
- 首次 009 流程验证可以先运行无动态遮罩基线；若注册失败、人物重影或漂浮噪点明显，再启用人物遮罩形成新的配置哈希和运行 ID，禁止覆盖基线结果。

### train

运行 Splatfacto 30,000 步。首次日志明确出现 CUDA OOM 时只用 `downscale-factor=2` 重试一次；其他错误不自动换算法或吞掉。

### export / qa / catalog

导出 Gaussian PLY、3 秒 3DGS 预览、缩略图和坐标变换。QA 先进入 `needs_review`，只有人工命令可以改为 `accepted` 或 `rejected`。SQLite 使用临时数据库构建并原子替换，随时可从 YAML 重建。

## 断点和清理

- 已成功阶段再次执行会拒绝覆盖；`--resume` 只复用同一配置和已完成文件。
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
