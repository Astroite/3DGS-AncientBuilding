# 009 前 90 秒 schema v4：时间密度与自动遮罩过滤运行手册

> **历史方案。当前操作请先读 [CURRENT-WORKFLOW.md](CURRENT-WORKFLOW.md)。**
> 本页的前 90 秒、WSL、旧目录及一键训练/导出命令不适用于 2026-09-09 接手的完整 293.86 秒 Windows/Postshot Run。

> 本页对应 schema v4 流水线。命令会实际解码视频、生成高分辨率视图、运行动态物体分割、RealityScan 与 Postshot；启动后保持供电并禁用休眠。

本次运行固定使用：

- 地点：`yanguan-ancient-town-20260822`
- 场景：`night-walk-4k`
- 采集：`capture-009-4k`
- 片段：009 视频的 `0–90` 秒
- 候选：全时段 5 fps；每秒保留综合质量最好的 2 张全景
- Primary：每秒 rank 1 × 8 个视图；Fallback：每秒 rank 1–2 × 14 个视图
- 动态物体处理：本地 Mask R-CNN 分割、边缘闭合与扩张；最终遮罩占比严格大于 5% 的单张透视图与遮罩成对删除
- 最终状态：`needs_review`，脚本不会自动批准结果

本次新 RunId 会把以下关键配置完整写入运行 YAML 并参与配置哈希：

```yaml
preprocess:
  candidate_fps: 5.0
  selected_per_second: 2
masking:
  enabled: true
  model: maskrcnn_resnet50_fpn_v2
  weights: DEFAULT          # 由固定 torchvision 0.16.2 解析为 COCO_V1
  device: cuda
  person_class_id: 1
  score_threshold: 0.25
  probability_threshold: 0.50
  inference_gamma: 0.75
  dilation_pixels: 24
  closing_pixels: 7
  mask_discard_threshold: 0.05
  mask_review_required: false
  qa_sample_count: 16
vision_qa:
  enabled: false            # 设置 DEEPSEEK_API_KEY 后用 --vision-qa 新建运行
  provider: deepseek
  model: deepseek-v4-flash-vision-exp
  image_detail: original
  minimum_confidence: 0.80
reconstruction:
  use_gpu_sift: true
  gpu_index: 0
  fix_intrinsics: true
  registration_threshold: 0.70
  rig_center_spread_ratio_limit: 0.001
  primary:
    {temporal_rank_limit: 1, images_per_equirect: 8, projection_fov_degrees: 120,
     projection_size: 2048, crop_bottom: 0.20, use_rig: true}
  fallback:
    {temporal_rank_limit: 2, images_per_equirect: 14, projection_fov_degrees: 110,
     projection_size: 1746, crop_bottom: 0.15, use_rig: true}
train:
  method: splatfacto-big
  max_iterations: 100000
  downscale_factor: 1
  cache_images: cpu
  cache_images_type: uint8
  use_scale_regularization: true
  rasterize_mode: classic
  camera_optimizer_mode: SO3xR3
  use_bilateral_grid: true
  oom_retry_downscale: 2
export:
  ply_axis: y_up
```

## 明早启动前检查

1. 确认 Insta360 Studio 已彻底完成导出，文件大小不再变化，然后关闭 Studio。目标文件必须是：

   ```text
   D:\Project\3DGS\locations\yanguan-ancient-town-20260822\scenes\night-walk-4k\inputs\stitched\capture-009-4k-equirect.mp4
   ```

   它应为 2:1 的 `3840×1920` 全景视频、29.97 fps、H.265、Rec.709 10bit。不要把 16:9 项目导出或 5.7K 上采样文件改名放进来。

2. 在 PowerShell 中只做只读确认：

   ```powershell
   Set-Location D:\Project\3DGS
   Get-Item .\locations\yanguan-ancient-town-20260822\scenes\night-walk-4k\inputs\stitched\capture-009-4k-equirect.mp4 |
     Select-Object FullName, Length, LastWriteTime
   Get-PSDrive D | Select-Object Name, Used, Free
   ```

3. 预留至少 100 GiB 的 D 盘空间更稳妥；程序的硬边界仍会保留至少 20 GiB 余量。不要在运行期间移动、覆盖或重新导出输入 MP4，也不要修改 E 盘原始 `.insv`。

4. 确认 Windows 接通电源，暂停系统休眠、自动更新重启和计划关机。显示器可以关闭，但 Windows 与 WSL 不能休眠。保持散热通畅。

5. 首次人物分割可能需要联网下载 Torchvision 的模型权重。DeepSeek 默认关闭，因此本地遮罩流程本身不需要 API key。

6. 先运行环境检查；只有所有关键项通过后才继续：

   ```powershell
   .\gsdb.ps1 doctor
   ```

   `doctor` 会检查 WSL 内存（至少 28 GiB）、Torch/Torchvision、gsplat 反向传播、COLMAP 的 `with CUDA` 标识和真实 GPU SIFT、小工具、写入权限及剩余空间。失败时先停下，不要绕过。

## 推荐：一键启动

确认上面的检查全部通过后，明早执行：

```powershell
Set-Location D:\Project\3DGS
# 已按下文设置 DEEPSEEK_API_KEY 时，使用远程遮罩 QA：
.\scripts\run-009-demo.ps1 -EnableVisionQa
```

如果临时决定不把联系表发送到外部服务，则去掉 `-EnableVisionQa`；本地 Mask R-CNN 与确定性遮罩检查仍会执行。

一键脚本会依次执行：

```text
doctor → ingest → preprocess → mask → reconstruct → train → export → qa report → catalog build
```

脚本会从 `preprocess` 输出中自动取得 RunId，并在任一阶段失败时立即停止。默认导出版本为 `v002`，QA 基线固定为 `20260824T022046Z-5679786b`：

```powershell
.\scripts\run-009-demo.ps1 -Version v002
```

不要关闭运行该脚本的 PowerShell 窗口。执行期间可以从另一个窗口查看日志和显卡状态，但不要同时为同一场景启动第二次流程。

## 分阶段启动

需要逐阶段观察时，使用下面的命令。先检查并登记输入：

```powershell
Set-Location D:\Project\3DGS

.\gsdb.ps1 doctor
.\gsdb.ps1 ingest `
  yanguan-ancient-town-20260822 night-walk-4k capture-009-4k --resume
```

创建新运行并预处理。明早的默认方案明确关闭远程视觉 QA，但仍会运行完整的本地人物遮罩：

```powershell
.\gsdb.ps1 preprocess `
  yanguan-ancient-town-20260822 night-walk-4k capture-009-4k `
  --no-vision-qa
```

命令结尾会打印类似：

```text
RUN_ID=20260825T001234Z-1a2b3c4d
```

复制实际值，后续始终使用同一个 RunId：

```powershell
$RunId = '20260825T001234Z-1a2b3c4d'

.\gsdb.ps1 mask `
  yanguan-ancient-town-20260822 night-walk-4k $RunId

.\gsdb.ps1 reconstruct `
  yanguan-ancient-town-20260822 night-walk-4k $RunId

.\gsdb.ps1 train `
  yanguan-ancient-town-20260822 night-walk-4k $RunId

.\gsdb.ps1 export `
  yanguan-ancient-town-20260822 night-walk-4k $RunId --version v002

.\gsdb.ps1 qa report `
  yanguan-ancient-town-20260822 night-walk-4k $RunId `
  --baseline-run-id 20260824T022046Z-5679786b

.\gsdb.ps1 catalog build
```

RunId 由“UTC 创建时间 + 配置哈希前 8 位”组成。运行清单和工作目录分别位于：

```text
locations\yanguan-ancient-town-20260822\scenes\night-walk-4k\runs\<RunId>.yaml
locations\yanguan-ancient-town-20260822\scenes\night-walk-4k\work\<RunId>\
```

修改动态物体置信度、是否启用 DeepSeek、候选帧率、每秒保留数或其他运行配置，都会改变配置哈希，必须从 `preprocess` 创建新 RunId。不要直接编辑已有运行 YAML；加载时会校验哈希并拒绝被篡改的配置。历史 schema v1–v3 run 仍按既有清单与哈希恢复。

## 遮罩阶段实际做什么

`mask` 在 RealityScan 之前完成以下工作：

1. 从选择起点开始按 5 fps 取得候选，每秒按综合质量选前 2；Primary 取 rank 1 并显式投影为 8 个 2048²/120° 视图。
2. 用 `maskrcnn_resnet50_fpn_v2` 检测和分割人物；夜景推理使用固定 gamma，人物边缘执行闭合和 24 px 扩张。
3. 为每张透视图保存同尺寸二值遮罩：白色为可用建筑像素，黑色为人物等忽略区域。
4. 按最终后处理遮罩计算占比；严格大于 5% 的图片/遮罩成对删除，恰好 5% 保留。先原子写 `.mask-filter.pending.json`，完成删除和剩余清单验证后再发布 `mask-filter.json`。
5. 默认自动生成 `mask-final.json` 并继续。传入 `--mask-review-gate` 时才对自动过滤后剩余图片生成审核材料并暂停；人工编辑使遮罩重新超过 5% 时，必须把该图显式标记为排除才能 finalize。

随后 `reconstruct` 只把最终纳入清单送给 RealityScan；使用同盘、唯一文件名的临时硬链接树规避跨视图同名帧冲突，人工排除的图片不会进入该树，也不复制大图。注册率分母、轨迹 QA、Postshot 准备与训练图片数均读取实际纳入清单。

## DeepSeek：默认禁用，设置 Key 后显式启用

不传 `-EnableVisionQa` 时使用默认的 `--no-vision-qa`。这只关闭远程多模态判定，不会关闭本地人物分割和确定性遮罩检查。

启用远程闸门会把两张、共 16 个视图的抽样遮罩联系表发送到 DeepSeek。联系表可能包含可识别的游客或拍摄者，应把它视为对第三方服务的数据外发；确认可以接受后再启用。

一键脚本优先读取 Git 忽略的 `D:\Project\3DGS\env\key.env`（单行原始 key），只在子流程运行期间注入环境变量并在 finally 中清除；也可提前在当前 PowerShell 进程设置：

```powershell
$DeepSeekCredential = Get-Credential -UserName 'deepseek-api-key' -Message '输入 DEEPSEEK_API_KEY'
$env:DEEPSEEK_API_KEY = $DeepSeekCredential.GetNetworkCredential().Password

.\scripts\run-009-demo.ps1 -EnableVisionQa
```

`gsdb.ps1` 和一键脚本只通过 Windows 的 `WSLENV` 转发 `DEEPSEEK_API_KEY` 变量名；密钥值不会成为命令行参数。Base URL 固定为官方 `https://api.deepseek.com`，客户端拒绝 HTTP 和重定向，避免 Bearer key 被转发到其他地址。

远程 QA 会调用 `deepseek-v4-flash-vision-exp`，只发送抽样遮罩联系表，不会发送 `.insv` 或完整视频。图片使用 OpenAI 兼容的 Base64 `image_url` 内容块并设置 `detail: original`；服务端仍会把大图按比例缩放到约 `800×800` 的总像素规模，因此联系表使用大字号标签和高对比度红色遮罩。模型必须返回结构化的 `pass/fail`；只有 `pass`、置信度不低于 0.80、且未报告漏遮或误遮视图时才放行。API 缺少凭据、超时、返回无效 JSON 或其他任一条件不满足时，遮罩阶段直接失败，RealityScan 不会继续。具体输入和限制见 [DeepSeek 官方图像理解文档](https://api-docs.deepseek.com/zh-cn/guides/vision/)。

结束或失败后清除当前会话中的秘密：

```powershell
Remove-Item Env:DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
$DeepSeekCredential = $null
```

API Key 不得写入 Git、README、脚本、YAML、命令参数或日志。`env/` 已整体忽略；仓库只保存环境变量名 `DEEPSEEK_API_KEY`，不会保存变量值。

启用 DeepSeek 会改变运行配置和 RunId。不能给已经以 `--no-vision-qa` 创建的 RunId“中途加开”远程 QA；应新建运行。

## 中断与恢复

需要主动停止时，在前台 PowerShell 中按一次 `Ctrl+C`，等待子进程退出。不要强制关机，也不要直接删除 `work\<RunId>`。

一键脚本重新执行会创建一个新 RunId；它不会自动猜测上一次应从哪里恢复。要恢复已知 RunId，请改用分阶段命令：

```powershell
$RunId = '<上次打印的 RunId>'

# 仅当 preprocess 本身未成功时使用：
.\gsdb.ps1 preprocess `
  yanguan-ancient-town-20260822 night-walk-4k capture-009-4k `
  --run-id $RunId --resume

# 从中断或失败的阶段开始；已成功阶段带 --resume 会直接复用：
.\gsdb.ps1 mask `
  yanguan-ancient-town-20260822 night-walk-4k $RunId --resume
.\gsdb.ps1 reconstruct `
  yanguan-ancient-town-20260822 night-walk-4k $RunId --resume
.\gsdb.ps1 train `
  yanguan-ancient-town-20260822 night-walk-4k $RunId --resume
.\gsdb.ps1 export `
  yanguan-ancient-town-20260822 night-walk-4k $RunId --version v002 --resume
.\gsdb.ps1 qa report `
  yanguan-ancient-town-20260822 night-walk-4k $RunId `
  --baseline-run-id 20260824T022046Z-5679786b --resume
```

`--resume` 只复用可验证的完成结果。人物遮罩逐文件核验；两阶段 `mask-filter` 可继续完成中断的成对删除，RealityScan 只有在 COLMAP 兼容模型、路径映射和最终清单全部一致时才复用。投影目录只生成一部分或图片/遮罩校验失败时会拒绝猜测。

遮罩阶段的续跑只复用可验证的实际输入：原始投影视图、逐图遮罩和联系表文件集合必须齐全。RealityScan 与 Postshot 都不会读取的多级缩小副本不再生成，也不再参与缓存有效性判断。

可在运行 YAML 的 `stages`、`active_stage`、`message` 和 `log_path` 字段中确认停在何处；各阶段日志位于对应的 `work\<RunId>\logs` 下。

## 自动停止边界

流程不会为了“跑出一个结果”而无限调参：

- `doctor`、输入 2:1/解码/哈希或空间检查失败：在预处理前停止。
- 本地遮罩缺失、尺寸错误或非二值：在 RealityScan 前停止；占比严格大于 5% 的单图自动淘汰。
- 启用 DeepSeek 后远程调用失败或判定 `fail`：在 RealityScan 前停止。
- Primary 过滤后不足两个时间桶，或注册率/最大组件覆盖低于 70%：只运行一次每秒 rank 1–2 × 14 的 Fallback。
- RealityScan 工具、许可证或导出错误：直接失败，不触发 Fallback。
- Fallback 仍不足两个时间桶或仍低于 70%：停止，不启动训练，并要求重拍。
- Splatfacto 首次明确 CUDA OOM：只降低一级训练分辨率重试一次；第二次 OOM 或其他训练错误直接停止。
- 任何失败运行都不会覆盖历史成功运行或已有导出。

## 预计耗时与空间

以下旧 90 秒基准只用于估算量级；schema v4 的实际图片数由时长和 5% 过滤结果动态决定，不是承诺值。

| 阶段 | 规划耗时 | 主要空间 |
| --- | ---: | ---: |
| doctor / ingest | 5–20 分钟 | 很小；首次 gsplat/模型缓存另计 |
| preprocess（5 fps 候选、2 fps 保留） | 随时长线性变化 | 完整候选保存在 prepared 缓存 |
| Primary 投影 + 动态遮罩（8 视图/秒） | 随时长线性变化 | 过滤后投影源删除 |
| RealityScan | 随最终纳入图片数变化 | 仅接收最终清单 |
| Splatfacto-big 100k | 1–5 小时 | 约 10–35 GiB |
| export / QA / catalog | 10–40 分钟 | 约 2–10 GiB |
| 训练 checkpoint（每 10k 步一个，共 10 个） | — | 约 10–25 GiB |

Primary 每秒最多 8 张透视图，Fallback 每秒最多 28 张；据此按选择时长线性估算。实际候选数、自动淘汰数、最终输入数、耗时、磁盘峰值和显存峰值都会写入运行清单与 QA 报告。

上面的区间已经计入中间数据改放 WSL ext4（`GSDB_SCRATCH_ROOT`，图像读取实测快 4 倍）、不再生成无人读取的图像/遮罩缩小副本，以及删除了一处对匹配结果没有影响的 COLMAP 数据库 ID 重排——那一步在 1.2 GB 主库上约 18 分钟、2.7 GB 降级库上约 45 分钟，全部是无效开销。

## 只验证流水线时用 `--smoke`

只想确认整条链路能跑通、不看质量时，用缩减档新建运行：

```powershell
.\gsdb.ps1 preprocess `
  yanguan-ancient-town-20260822 night-walk-4k capture-009-4k --smoke
```

它仍使用 5 fps 候选、每秒保留 2 张，但只处理选择区间前 5 秒：默认计数为 25 个候选、10 张保留全景、Primary 5×8=40 张预过滤透视图；投影边长 1024、训练上限 5000 步。配置哈希天然不同，因此会落在自己的 RunId 下，绝不会覆盖或冒充正式运行。**`--smoke` 的产物不能用于质量判断，也不应提交审核。**

因此，明早启动后当天未完成不等于卡死。判断是否仍在工作应查看日志更新时间、CPU/GPU 占用和工作目录增长，不要仅凭控制台一段时间没有新行就中断。

## 人工检查产物时看什么

指标能拦住机器能判断的问题，剩下的必须人眼看。把 `exports\<版本>\splat-yup.ply` 和同目录的
`transforms.json` 一起载入查看器，确认四件事：

1. **场景是正立的**。这是重力恢复是否正确的最终判据。QA 报告里"发布后相机高度跨度占比"
   应该是个很小的数（009 实测 1.58%）；如果场景躺倒或倒置，说明素材的重力稳定有问题。
2. **相机落在场景内部**，沿街道排成一条连续轨迹，而不是飘在外面。
3. **远处的浮点球消失了**，天空与远景区域干净。
4. **街道本体没有被削掉** —— 尤其是街道尽头的远景建筑。如果发现该留的被删了，
   调大 `--cull-distance-factor` 重新发一个版本即可，不需要重训。

QA 报告的"外围高斯剔除"一节会列出剔除总数以及按距离、按尺度各剔除多少。剔除比例通常在 2% 左右；
明显偏高（接近 5% 上限）值得先看一眼再接受。

## 完成判定

一键脚本最终打印类似：

```text
Completed <RunId>. Artifacts remain needs_review; no automatic approval was performed.
```

完成后应存在：

- 场景 `exports002` 下有 `splat-yup.ply`、`preview.mp4`、`thumbnail.jpg`、`artifact.yaml`，以及三个互相能对齐的坐标文件：`transforms.json`（与 PLY 同坐标系）、`transforms-colmap.json`（重建原始）、`dataparser_transforms.json`（Nerfstudio 归一化参数）；Nerfstudio 的原始 PLY 只留在 Git 忽略的 work staging；
- `qa\<RunId>.md`；
- `runs\<RunId>.yaml` 中完整的遮罩、重建、训练、资源和产物记录；
- 从 YAML 原子重建的 `catalog\catalog.sqlite` 记录。

此时运行状态必须保持 `needs_review`。不要在无人值守脚本中追加 `qa approve`。即使由 DeepSeek 完成人物遮罩 QA，它也只负责遮罩闸门，不等于对最终几何、漂浮噪点、夜景曝光伪影、权限或游戏参考价值作最终批准。
