# GSDB：古建筑 360 视频到 3DGS 参考库

> 当前工具版本为 **0.2.0**。新建采集可使用 2:1 视频、2:1 图片序列，
> 或获批 Desktop MediaSDK 明确支持型号的 INSV；旧 schema 1 清单和历史 run
> 保持原样读取。完整命令见 [v0.2 多输入与质量门禁手册](docs/V0.2-MULTI-INPUT.md)。

这个仓库按“地点 → 场景 → 采集 → 运行 → 产物”管理全景视频重建。地点是资料目录，场景才是一次 COLMAP/Splatfacto 可以独立处理的空间单元；长距离古镇素材应拆成有连续视觉重叠的院落、街段或走廊，不能把几十分钟视频直接塞进一个模型。

当前试点：

- 地点：`yanguan-ancient-town-20260822`（盐官古镇）
- 场景：`night-walk-4k`
- 原始素材：`E:\Photo\PhotosRaw\2026\2026-08-23\Insta360\VID_20260822_211338_00_009.insv`
- 选段：完整素材约 293.86 秒，首跑使用前 90 秒
- 目标：证明处理闭环可行；当前夜景素材不是正式质量标杆

原先的 `night-pilot-8k` / `capture-004-8k` 清单继续保留，作为另一条独立采集血缘，不被 009 覆盖。

## 数据边界

- E 盘原始 `.insv` 永远按只读来源处理，不复制、不修改。Python 不逆向格式；
  仅在获批 Desktop MediaSDK 和 Windows helper 可用、且型号已验证时，helper 才会
  只读解码本次选中的帧。X6 已随 MediaSDK 一并验证，可直接走 `insta360_insv`
  源类型，不再要求 Insta360 Studio 手动导出。
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

在 009 前 90 秒的真实数据上，用同一批图像与同一份特征对照：

| | 只有 sequential | 加上同帧跨视图 |
| --- | ---: | ---: |
| 连通组件数 | 8（按视图目录切开） | 1 |
| 注册率 | 867/1080 = 0.803 | 1067/1080 = 0.988 |
| 最大组件覆盖 | 346/1080 = 0.320 | 1067/1080 = 0.988 |
| 70% 门槛 | 未过，触发降级 | 通过 |

跨视图匹配本身耗时 20 秒（3780 个 pair 中 2793 个通过几何验证）。代价在 mapper：它不再半途放弃、而是真的去建一个完整模型，从 6.2 分钟增加到 36.9 分钟——换来的是整条 180×14 降级分支不必再跑。

主流程使用 135×8 个 2048²/120° 透视视图、20% 底部裁剪和两个下采样层级。如果注册率或最大连通模型覆盖低于 70%，只进行一次确定性的降级尝试：从原始 270 候选帧按桶选 180 帧，转换为 180×14 个 1746²/110° 视图、裁底 15%，并生成自己的完整遮罩与 rig。第二次仍失败就停止。

### postshot-prepare

成功重建后，可把最终的 rig 约束 COLMAP 模型整理为 Postshot 可直接导入的数据集：

```powershell
.\gsdb.ps1 postshot-prepare LOCATION_ID SCENE_ID RUN_ID
```

新 v3 run 默认输出到 `locations/<location>/scenes/<scene>/inputs/postshot/<run>/`；
旧 run 仍保留 `work/<run>/postshot/`。数据集包含平铺且唯一命名的
`images/`、Postshot 白色忽略语义的 `masks/`、同步改名后的 `colmap/`
二进制模型、`image-map.csv`、`dataset.json` 和 `IMPORT.md`。只导出最终模型中
已注册的图像；原始图像、遮罩和 COLMAP 模型保持只读。中断后使用 `--resume`
续作，也可用 `--output PATH` 指定其他 Windows 可访问位置。

`postshot-train` 的输出必须使用 `.psht` 后缀。默认训练日志与血缘清单为项目旁的
`postshot-train.log`、`training.json`；自定义项目名则使用
`<name>.postshot-train.log`、`<name>.training.json`。成功与失败都会保留命令、版本、
GPU 指标及已生成产物哈希。

在 Postshot v1.1.69 或更新版本中同时导入 `images/` 与 `colmap/`，再把 `masks/` 添加到 Image Masks，选择 `Remove Occluders`，并确认 Camera Poses 为 `Import`、Image Selection 为 `Use All`。该数据集已有外部相机位姿，不要重新运行 camera tracking。

### postshot-review

可启动本地遮罩审核器，对已经整理的 Postshot 数据集做额外人工抽查：

```powershell
.\gsdb.ps1 postshot-review LOCATION_ID SCENE_ID RUN_ID
```

打开命令输出的 `http://127.0.0.1:8765`。页面按遮罩占比排序，支持原图、
红色遮罩叠加、并排和遮罩单独查看，也可记录通过、漏遮人物、误遮背景或
整图剔除建议。审核结论原子写入 Postshot 数据集下的 `mask-review.json`，不会
修改原图或遮罩；页面右上角可导出 CSV。它是额外 QA 记录，不是训练门禁，也
不会改写已经冻结的训练清单；强制门禁是在 COLMAP 前执行的 `mask-review` 与
`mask-finalize`。使用 `--dataset PATH` 审核自定义输出，使用 `--port` 更改端口，
终端按 `Ctrl+C` 停止服务。

### 入镜人物与拍摄者

- 底部裁剪先去掉自拍杆、手臂和大部分位于天底的拍摄者身体；如果身体仍伸出裁剪区，则给全部透视视图增加固定天底遮罩。
- 游客属于随时间移动的瞬态物体。正式流程应在透视视图生成后做人像分割，适度扩张遮罩边缘，并为每张图保存同尺寸黑白遮罩。
- 同一组遮罩必须同时用于 COLMAP 特征提取和 Nerfstudio 训练：黑色人物区域不提特征、也不参与像素监督，避免错误相机匹配和 3DGS 漂浮人影。
- 当前 009 试跑默认就是动态人像遮罩版本；关闭遮罩或改变阈值会形成不同配置哈希和运行 ID，禁止覆盖既有运行。

### train

运行 `splatfacto-big` 100,000 步，CPU/uint8 图像缓存、scale regularization、classic rasterizer、bilateral grid、`SO3xR3` 相机优化和 `downscale-factor=1` 均显式入配置。首次日志明确出现 CUDA OOM 时只用 `downscale-factor=2` 重试一次；不启用当前版本未暴露的 MCMC 参数。

训练现在是整条流水线的大头：009 那次跑了 4 小时 09 分，占总时长 81%（重建只有 46 分钟）。每步 144.3 ms，其中前向 45 ms、反向加优化器 99 ms。

关键背景是 splatfacto 的 `stop_split_at`：**致密化到这一步就停止，之后只是在精修一组固定的高斯**（009 最终 878,467 个，还低于 QA 的 2M 告警线，说明 splatfacto-big 的额外容量没被用上）。所以"100k 步是否必要"很可能有很大的下调空间，但这应当由数据决定。

#### 致密化调度按视图数推导

`stop_split_at` 的默认值 15,000 是**步数**，但真正决定几何能否建起来的是**覆盖度**——致密化结束前每张训练图被采样了多少次。默认值是按 1,000 张量级的数据集调的，而 009 基准恰好是 1,080 张（135×8），所以正好合适：15000/1080 = 13.9 次/张。

数据集变大时照搬这个步数会出事。427 秒素材是 5,128 张，同样 15,000 步只有 **2.9 次/张**，大部分区域来不及累积足够的位置梯度去分裂——结果不是细节软，而是**几何压根建不起来**。

因此 `warmup_length`、`stop_screen_size_at`、`stop_split_at` 三个步数边界现在从配置里的 `frame_count × images_per_equirect` 自动推导，保持 13.9 次/张不变：

| | 透视图 | warmup | screen_size | split_at | 次/张 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 009 基准 | 1,080 | 500 | 4,000 | 15,000 | 13.9 |
| 427 秒全长 | 5,128 | 2,374 | 18,993 | 71,222 | 13.9 |

1,080 张时推导结果与 Nerfstudio 默认值逐字相同，因此已验证的基准行为没有改变。推导只依赖已经进入配置哈希的两个字段，所以"同一个哈希 = 同一个模型"仍然成立，**不需要新增配置项，已有的运行清单也不会失效**。`stop_split_at` 始终不超过 `max_iterations` 的 75%，保证留有精修尾巴。实际值写入 `metrics.train.densification`。

因此每次运行默认保留全部 checkpoint（`--steps-per-save 10000`，共 10 个）并用 `--vis tensorboard` 记录评估曲线：**一次运行就能回答"质量在第几步走平"**，可以分别从 15k / 30k / 50k / 100k 导出对比，不需要为了找拐点重复训练。曲线会进 `metrics.train.eval_curve` 并输出到 QA 报告。

这些是仪表参数，**不进配置哈希** —— checkpoint 频率和日志后端都不改变训练结果，把它们计入哈希会让已有的运行清单全部失效。代价是多占约 9 个 checkpoint 的磁盘。

### export / qa / catalog

Nerfstudio 的原始 PLY 只留在忽略的工作区；发布目录包含旋转位置、法线、wxyz 四元数与 1–3 阶实 SH 后的 `splat-yup.ply`、预览、缩略图、`transforms.json` 和清单。QA 对比 v001/v002 指标；低于 2M 高斯只告警，不自动追加训练，状态始终先进入 `needs_review`。

#### 发布坐标系

发布方向由**素材自身的重力**决定，不再写死一个轴交换。每个透视视图都是从重力稳定的全景帧按已知 yaw/pitch 切出来的，所以 rig 坐标系（Nerfstudio 等距柱状约定 `[forward, right, up]`）的第三列就是世界上方向；对全部已注册图求平均即可恢复重力。009 实测逐图偏差中位 0.45°、p95 0.81°，非常一致；p95 超过 5° 会直接报错，说明素材没有做重力稳定。

之所以必须这样做：Nerfstudio 的 dataparser 用相机 up 向量的均值来定向，而全景 rig 的 8 个视图指向四面八方，这个均值恢复不出重力。009 那次它把场景摆成了 Y-up，而导出端硬编码假设输入是 Z-up 又转了一次，结果发布出来的"y-up"文件实际是 −Z 朝上，与 +Y 差 90.1°。

同时发布三个文件，互相能对齐：

- `transforms.json` —— 换算到 PLY 同一坐标系的相机，可直接与 splat 叠加
- `transforms-colmap.json` —— 重建阶段的原始文件，留作审计
- `dataparser_transforms.json` —— Nerfstudio 的归一化参数（009 的 scale 是 0.19451，即 1/5.14）

两道断言保护这条链路：换算后相机坐标绝对值的峰值必须约等于 1.0（`auto_scale_poses` 的归一化结果），发布后相机沿 +Y 的高度跨度占轨迹范围必须低于 10%（009 实测 1.58%）。

#### 外围高斯剔除

判据是**到最近相机的距离**加一个绝对尺度上限，**不是不透明度**。009 实测：距离超过轨迹半径 2 倍的 29,321 个高斯，中位不透明度是 1.000 —— 它们完全不透明，任何 opacity 阈值都删不掉；反过来低透明度的高斯（opacity < 0.02）中位距离只有 0.29，是近处细节，删了会掉质量。

默认距离因子 3.0、尺度因子 1.0，都相对相机轨迹半径定义，换场景不失效。两道安全闸：剔除比例超过 5% 直接报错；剔除前先确认相机确实落在点云内部（相机到最近高斯的中位距离必须小于轨迹半径），这道检查专门拦"相机和 PLY 不在同一坐标系"这种会一次删掉大半场景的情况。

剔除参数是**导出期参数，不进配置哈希**：它改变的是"发布什么"，不是"学到了什么"。因此同一个 run 可以用不同参数发布多个版本，`run.artifacts` 按版本累加而不是覆盖，已发布的版本不会被抹掉。

```powershell
.\gsdb.ps1 export yanguan-ancient-town-20260822 night-walk-4k $RunId --version v003 `
  --cull-distance-factor 3.0 --cull-scale-factor 1.0
```

加 `--no-cull` 可以完全关闭。

## 90 秒试跑耗时预估

`20260825T062740Z-46f434b6` 是第一次完整跑通的运行，各阶段实测（RTX 4070 Ti SUPER 16 GiB，未触发降级）：

| 阶段 | 耗时 | 占比 |
| --- | ---: | ---: |
| preprocess | 40 s | 0.2% |
| mask | 574 s | 3.1% |
| reconstruct | 2,766 s | 15.0% |
| **train** | **14,934 s** | **81.0%** |
| export | 113 s | 0.6% |
| 合计 | **5.1 小时** | |

主流程按 **4.5–8 小时**规划，触发 180×14 降级时按 **7–14 小时**；夜景匹配与 100k 训练仍会波动，这不是进度承诺。**训练现在是唯一的大头**，见上面 `train` 一节关于步数的说明。建议接通电源并关闭休眠；程序保留 20 GiB 硬余量，脚本不会主动删除中间数据。

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
