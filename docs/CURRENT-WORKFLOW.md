# Windows 360 重建工作手册

本文件是 APP 唯一的日常操作手册。现行流程为 INSV / 标准全景 / 透视视频输入 → 候选抽帧 → RealityScan → 分段 QA → Postshot Splat ADC 或 Windows gsplat。脚本清单、环境偏离记录和不支持的历史形态见 [维护说明](MAINTENANCE.md)。产品方向是以 GUI 为日常入口的 GS-Studio，见[项目路线图](GS-STUDIO-PROJECT-ROADMAP.md)和[工程路线图](GS-STUDIO-ENGINEERING-ROADMAP.md)；路线图是规划，不取代本手册的实际命令。

新 Run 使用下列默认，已有 Run 按原清单恢复。代码接入、短测试通过和最终画质验收分别记录，不互相替代。

## 1. Windows 环境与检查

准备 Python 3.10、PowerShell、NVIDIA 驱动、FFmpeg/FFprobe、RealityScan 2.2，以及所选后端需要的 Postshot。INSV 另需本机 Insta360 Desktop MediaSDK 和编译完成的 Windows helper。安装脚本不安装商业软件或 SDK。

```powershell
$AppRoot = 'D:\Project\3DGS\APP'   # 改为实际 APP 路径
$DataRoot = 'D:\Project\3DGS\Data' # 已存在的数据根目录
# 仅首次安装运行以下两条，已有正常环境不要重复安装。
& (Join-Path $AppRoot 'scripts\bootstrap-windows.ps1')
& (Join-Path $AppRoot 'scripts\bootstrap-gsplat-windows.ps1')

. (Join-Path $AppRoot 'scripts\session.ps1') -DataRoot $DataRoot
Invoke-Gsdb --help
Invoke-Gsdb doctor --backend all --require mediasdk
```

主环境是 `.venv`；独立训练与统一 PLY 评估环境是 `.venv-gsplat`，固定 PyTorch 2.9.1+cu130、gsplat 1.5.3（源码编译为 sm_120 扩展，缓存在 `wheels/`）。即使使用 Postshot，CLI 训练后的统一评估也需要独立环境。版本对照、构建方式与回退阶梯见 [维护说明](MAINTENANCE.md)。

`session.ps1` 支持任意当前目录，初始化后使用 `Invoke-Gsdb`。绝对路径调用 `gsdb.ps1` 也会复用相同初始化。Data 路径优先级为显式 `-DataRoot`、已有 `GSDB_DATA_ROOT`、APP 同级 Data。已有 SDK/helper/trainer 环境变量保留；不读取或打印密钥值。

按实际安装位置配置；已有正确配置无需重设：

```powershell
$env:GSDB_REALITYSCAN_CLI = 'C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe'
$env:GSDB_REALITYSCAN_EXPORT_PARAMS = Join-Path $AppRoot 'tools\realityscan-setup\colmap-export-params.xml'
$env:GSDB_POSTSHOT_CLI = 'C:\Program Files\Jawset Postshot\bin\postshot-cli.exe'
$env:INSTA360_MEDIA_SDK_ROOT = Join-Path $AppRoot 'sdks\Insta360-MediaSDK'
$env:GSDB_MEDIA_HELPER = Join-Path $AppRoot 'tools\mediasdk-helper\build\Release\gsdb-media-helper.exe'
# CUDA 工具链不在标准位置时设置 GSDB_CUDA_HOME；自定义训练解释器时设置 GSDB_GSPLAT_PYTHON。
```

已有 SDK/helper 不重建。新安装见 [MediaSDK helper](../tools/mediasdk-helper/README.md)，导出设置见 [RealityScan 配置](../tools/realityscan-setup/README.md)。

`doctor --backend postshot` 为默认检查；`--backend gsplat` 检查独立环境 CUDA 前向/反向；`--backend all` 检查两者。公共检查包含主环境 CUDA、分割依赖、RealityScan、可解析的导出 XML、Data 可写性及剩余空间。检查使用工作区 GPU 锁，不运行实际重建或训练。CUDA COLMAP 仅在 `--require colmap` 时检查，现行流程不要求 WSL。

Postshot 版本可用不证明 Studio CLI 训练许可证可用。不能以帮助输出或退出码零判断训练成功；CLI 受限时使用第 4 节 GUI 路线。

## 2. 从原片创建场景和 Capture

原片只读。地点是组织目录，场景是空间连续、可独立重建的街段或院落。同文件的不同选段用不同 Capture ID；已有地点、场景和 Capture 不重复 init，不覆盖清单更改输入。

```powershell
$LocationId = 'my-location'
$SceneId = 'my-scene'
$CaptureId = 'capture-stable'
$Source = 'D:\path\to\capture.insv' # 替换为真实原片
$SceneDir = Join-Path (Join-Path $env:GSDB_DATA_ROOT $LocationId) $SceneId

Invoke-Gsdb location init $LocationId --name '地点名称'
Invoke-Gsdb scene init $LocationId $SceneId --name '场景名称'
Invoke-Gsdb capture init $LocationId $SceneId $CaptureId `
  --source-type insta360_insv --source $Source --camera-model X6 `
  --output-width 3840 --output-height 1920 `
  --start-seconds 0 --end-seconds 45
Invoke-Gsdb media probe $LocationId $SceneId $CaptureId
Invoke-Gsdb ingest $LocationId $SceneId $CaptureId
```

示例创建前 45 秒候选窗口，结束时间必须在原片范围内。正式对照选择最早通过 QA 的连续 30 秒；窗口不足时另建更长 Capture，不降低门槛。全片 Capture 省略 `--end-seconds`，使用探测时长；这不等于应立即全片运行。室内/古建走位、快门与细节补拍见 [古建室内采集 SOP](CAPTURE-SOP.md)（表盘走位法）。

四种源类型，投影语义不同：

| `--source-type` | 输入 | 训练视图 |
| --- | --- | --- |
| `insta360_insv` | Insta360 原片（helper 拼接），`.insv` 一或两个文件 | 每张全景切成 14 个投影视图 |
| `equirect_video` | 标准 2:1 全景视频 | 同上 |
| `equirect_sequence` | 2:1 图片序列，需 `--fps` | 同上 |
| `perspective_video` | 横屏透视视频（如 DJI 航拍） | 1 帧 = 1 视图，不投影 |

INSV 不必先手工导出 MP4，实际型号支持以 helper capabilities 和 probe 为准。不要将 16:9 重构视频当全景。

透视源额外提供 `--horizontal-fov`（默认 84°）作为 RealityScan 焦距初值，训练内参以导出的 COLMAP 模型为准。因为没有 14 倍视图放大，同等时长下训练图数远少于全景源。提高训练图数必须同时提高三个参数：`--candidate-fps`、`--selected-per-second`、`--primary-per-second`。只提高 `--candidate-fps` 不会增加训练图数——初选仍按 `--primary-per-second` 每秒只留最清晰的几张。三者约束为 `primary ≤ selected_per_second ≤ candidate_fps`。

```text
Data/<location>/location.yaml
Data/<location>/<scene>/scene.yaml
  captures/<capture>.yaml
  logs/                            capture 探测协议记录
  <run>/manifest.yaml
  <run>/inputs/primary/<hash>/     候选帧与 dataset.json；Run 独占，内容寻址
  <run>/inputs/repair/             仅按需补抽；合并确认后删除 frames
  <run>/frames-primary/            时序初选
  <run>/reconstruction-primary/    images/、masks/、COLMAP、segments.json
  <run>/repair-new/                补抽新帧的遮罩结果
  <run>/reconstruction-repair/     合并后的补选数据集（仅触发补选时产生）
  <run>/training-data/<segment>/
  <run>/logs/  <run>/cleanup/
  qa/<run>.md
```

## 3. 新 Run、遮罩与重建

```powershell
Invoke-Gsdb preprocess $LocationId $SceneId $CaptureId
$RunId = '<复制实际输出的 RUN_ID>'
$RunDir = Join-Path $SceneDir $RunId
Invoke-Gsdb mask $LocationId $SceneId $RunId
Invoke-Gsdb reconstruct $LocationId $SceneId $RunId
Invoke-Gsdb qa report $LocationId $SceneId $RunId
```

| 设置 | 默认 |
| --- | --- |
| 候选 / 初选 | 1 fps / 每秒一张 |
| 全景投影 | 14 视图、110°、1746 × 1746；中环 yaw 步进 60°、外环 90°，pitch −45°/0°/+45° |
| 视口重叠规则 | 相邻视图球面角距 `sep = arccos(sin²φ + cos²φ·cosΔλ)`，要求 `FOV − sep ≥ 15°`；14×110° 默认满足（环内相邻约 20°–50°+）。Insta360 原片为整球，重叠由投影切分决定，不由相机硬件决定 |
| 底部裁切 | 15% |
| 透视源 | 不投影，`--horizontal-fov` 仅作焦距初值 |
| 遮罩整图剔除 | 严格大于 0.005（0.5%）；等于保留 |
| 补选 | 薄弱区间按需 2 fps 补抽，最多一次，前后各 2 秒窗口，不放宽阈值 |
| 中间文件 | 默认按阶段清理；`--keep-intermediates` 保留调试输入 |
| 正式训练外观 | SH 3 |
| 默认后端 | Postshot Splat ADC |

人物遮罩默认 Mask R-CNN：置信度 0.25、像素阈值 0.50、推理 gamma 0.75、膨胀 24 px、闭合 7 px。推理 gamma 只用于人物检测，不是整体压暗训练图像。人工遮罩门禁和外部 vision QA 默认关闭；自动 mask-final 不代表人工审核。向外部服务发送图片须有明确授权。

全景源先切视图再遮罩：人站在 360 机身旁只占球面一小片，却占整幅等距柱状图很大比例，按整图比例剔除会把每张都判掉，切分到 14 个视图后才定位到少数几个。透视源本身就是视图。

RealityScan 白色保留、黑色忽略，实际输入为 `图像名.jpg.mask.png` 图层和 `inpMaskOpts=1`。`alignment-masks.json` 核验逐图对应、哈希和启用命令，不声称读取了软件内部全部被屏蔽特征。刚性 rig 未启用；另做同全景中心一致性 QA。

初次对齐后，对少于三个剩余视图、未注册采样、超过两秒的缺口及异常区间，合并前后各两秒的窗口并裁到 Capture 范围。以 Capture 起点为基准，在 2 fps 网格上从原片补抽尚未处理的帧，按源帧索引去重。只投影和检测新增图片，复用 primary 已过滤的图片与遮罩；补抽计划先落盘，恢复不增加第二轮。原轮与 repair 独立 QA，选择有效覆盖更好的模型，不拼接不同坐标系。

新 Run 的候选目录归该 Run 独占。初次候选量、投影量按上表估算，不代表覆盖或画质已经验收。`--smoke` 仅为前五秒、较小投影的链路检查，不自动训练，也不作为正式画质对照。改配置应新建 Run，恢复不能更换输入或阈值。

## 4. 分段 QA 与训练

`segments.json` 位于选中的 reconstruction 目录。`manifest.yaml` 的 `selected_dataset` 和 `metrics.selected_attempt` 是实际选择依据，不能假设总是 primary。

三个结果独立解释：

- 数据完整性：缺文件、相机错配、非有限位姿、错误时间戳阻断。
- 分段适用性：至少十个时间采样、三个不同位置，分段实际输入图像注册率至少 70%；未通过分段隔离。
- 路线覆盖：首尾缺失、未注册区间和长缺口单独报告。存在可训练分段不等于全路线完整。

速度按真实时间间隔计算，阈值取十倍中位速度与中位速度加十倍稳健离散度中的较大值。长缺口切段但不判瞬移；旋转先消除投影朝向。异常通过切段、隔离处理，不用轨迹平滑掩盖。

选择报告中实际 passed 的分段：

```powershell
$SegmentId = 'segment-001' # 替换为实际通过的 ID
$NativeOutput = Join-Path $RunDir 'experiments\gsplat-photo-on'
$PostshotOutput = Join-Path $RunDir 'experiments\postshot-photo-on'

# 实际原生训练；先完成代表片段检查。
Invoke-Gsdb train $LocationId $SceneId $RunId --segment $SegmentId `
  --backend gsplat --output $NativeOutput

# 只准备 Postshot 输入、适配包和命令。
Invoke-Gsdb train $LocationId $SceneId $RunId --segment $SegmentId `
  --backend postshot --output $PostshotOutput --dry-run
# 有 Studio CLI 许可后，去掉 --dry-run 执行相同命令。
```

默认启用光度补偿，关闭用 `--no-photo-comp`；不同配置用不同输出目录。gsplat 默认 Bilateral Grid 匀光与 Sparse Depth 深度锚（`--no-use-bilateral-grid` / `--no-use-sparse-depth` 可关）；包内无 `sparse_depth.npz` 时自动跳过深度锚。默认预算 `max(30000, 30 × 训练图片数)`；Postshot 按千步向上取整，以保存的实际命令为准。较长有效段可加 `--duration-seconds 30`，选完全位于通过分段内、重新核验注册率的最早窗口；不足时拒绝。

两后端共享图片、遮罩、相机和初始化点。训练/验证按采样分组；初始化颜色只用未遮罩训练观测。gsplat 按需加载图像，遮罩排除 L1/SSIM 观测，评估处理遮罩边界，不把人物区域训练成黑色。曝光/白平衡按全景共享参数，固定参考并约束时间连续性，导出标准 SH 3。

### Postshot GUI

准备成功后使用 `$PostshotOutput\postshot-input`：

1. 导入 `images` 与 `colmap`，Camera Poses=Import，Image Selection=Use All。
2. `masks` 加入 Image Masks，选择 Remove Occluders：白色忽略，与 RealityScan 极性相反。
3. 确认图像、遮罩及相机匹配，数量等于共享包训练组；不额外导入验证图片。
4. 使用 Splat ADC、SH 3，额外抗锯齿关闭；光度补偿按本次实验开关设置。已有位姿不重新 tracking。
5. 保存新的 `.psht` 并导出 PLY，记录设置。GUI 成功不自动更新 CLI `dispatch.json`，不得手改为 succeeded 冒充可复现实验。

## 5. 恢复、结果与故障

恢复前确认没有同一 Run 的活动进程，检查 manifest、阶段日志和过滤记录。MediaSDK、遮罩、重建及训练使用跨进程 GPU 锁，等待消息不代表失败。不要绕过锁另启 GPU 作业或重复提交同一 Run。

```powershell
Get-Content -LiteralPath (Join-Path $RunDir 'manifest.yaml')
# 根据实际失败阶段选择一条，不要无条件全部执行：
Invoke-Gsdb preprocess $LocationId $SceneId $CaptureId --run-id $RunId --resume
Invoke-Gsdb mask $LocationId $SceneId $RunId --resume
Invoke-Gsdb reconstruct $LocationId $SceneId $RunId --resume
Invoke-Gsdb qa report $LocationId $SceneId $RunId --resume

# 同输入、同配置的原生检查点恢复：
Invoke-Gsdb train $LocationId $SceneId $RunId --segment $SegmentId `
  --backend gsplat --output $NativeOutput --resume
```

改变分段时长、补偿或步数应使用新实验目录；Postshot CLI 不支持该 `--resume` 分支，需按保存项目另行继续。

| 文件或目录 | 含义 |
| --- | --- |
| Run 的 `manifest.yaml` | 阶段状态与配置的权威记录 |
| `mask-filter.json` / `mask-final.json` | 剔除与最终输入清单，勿删除以绕过检查 |
| `repair-plan.json` | 唯一一次补选身份 |
| `training-data/<segment>/dataset.json` | 共享包、分组、相机和文件哈希 |
| 实验 `dispatch.json` / `train.log` | 准备或实际执行状态、命令与日志 |
| 原生 `checkpoint.pt` / `failure.json` | 检查点 / 失败和恢复说明 |
| 原生 `training.json` / `runtime*.json` | 训练与运行环境记录 |
| `model.ply` / `model.psht` | 高斯模型 / Postshot 项目，须结合状态检查 |
| `evaluation/metrics.json` 及 PNG | 验证指标、预测/参考图与角度扫描 |

OOM 保留可恢复状态，不自动降分辨率或 SH。没有检查点的初始化失败按 failure.json 使用同输入新目录重试。哈希变化应调查原因，不重写哈希通过校验。训练资源含每图平均采样次数、点数、显存和耗时；恢复报告应辨明累计或本次运行计时。

Postshot 返回零但无模型时查看 Studio license 日志；导入包存在只表示准备成功。源模型或输入已清理后，历史报告无法代替可恢复数据。

### 阶段清理与磁盘占用

默认 `retention.mode=minimal`。遮罩过滤及最终确认通过、清单保存后，删除候选帧与初选全景；repair 合并输入通过后删除补抽全景。对齐、分段 QA 和选择结果完成后，删除临时导入图片、可识别的 Run 内项目中间文件及未选中尝试的大体积输入。有效图片、遮罩、COLMAP、位姿、检查点、模型、日志及来源清单保留。

同盘训练包与后端图片使用硬链接，共享内容不得原地修改。不同极性的遮罩单独保存；不支持硬链接或跨盘时校验后复制，`storage.json` 报告复制占用。资源管理器将多个目录大小相加会重复计算硬链接，清理报告分别列出 `logical_bytes` 和预计可回收的 `reclaimable_bytes`。

```powershell
# 只预览，不删除数据。
Invoke-Gsdb cleanup $LocationId $SceneId $RunId
# 重试已满足依赖的清理，自动清理通常无需手动调用。
Invoke-Gsdb cleanup $LocationId $SceneId $RunId --apply
# 调试时在新建 Run 的 preprocess 命令追加 --keep-intermediates。
```

每次清理在 `cleanup/<hash>.json` 中先写删除计划，再逐项保存结果，记录文件哈希、大小和消费者验证依据。锁定当前 Run，拒绝路径越界及目录连接；不清理全局缓存、其他 Run、原片、外部实验目录或不认识的文件。清理失败保留 `pending`，计算阶段已验证成功的状态不变；下次恢复或 `cleanup --apply` 重试。

`retention-retired.json` 标记未选中尝试仅保留审计记录，不能把这些记录视为可直接恢复的输入。选中重建保留完整过滤输入，仍可重跑 QA、选择其他有效分段及准备两后端对照。训练成功不删除这些输入或检查点，也不代表已通过人工验收。已按计划清理的候选不会因恢复已完成阶段而自动重新生成；有效输入缺失或哈希不符仍然阻断。

人工遮罩确认开启时，先完成对应确认，再继续：

```powershell
Invoke-Gsdb mask-finalize $LocationId $SceneId $RunId --attempt primary
# 仅出现 repair 人工确认门禁时执行：
Invoke-Gsdb mask-finalize $LocationId $SceneId $RunId --attempt repair
Invoke-Gsdb reconstruct $LocationId $SceneId $RunId --resume
```

## 6. 对照、高亮诊断与全量条件

通用对照入口不依赖预设地点、日期或旧 Run：

```powershell
$ComparisonOutput = Join-Path $RunDir 'experiments\backend-comparison'
& $GsdbPython (Join-Path $AppRoot 'scripts\compare-backends-v5.py') `
  --run-dir $RunDir --segment $SegmentId --output $ComparisonOutput --dry-run
if ($LASTEXITCODE -ne 0) { throw '检查 comparison.json 中的失败原因' }
```

去掉 `--dry-run` 会实际训练两后端的补偿开/关组合。`--backends gsplat` 或 `postshot` 限定后端，`--photo-comp both|on|off` 选择组合；`--resume` 恢复原生检查点并核验复用成功模型。失败或许可证阻断时返回非零，结果保存在 `comparison.json`；不会自动启动全场景。

使用同一分组、共享数据和验证相机，比较 PSNR、SSIM、LPIPS、覆盖、人物残留、细节、高亮及资源消耗。高亮诊断顺序：

1. 固定相机、显示曝光及色彩设置，对比 Postshot 内部和 PLY 渲染。
2. 同模型对比正式 SH 3 与诊断 SH 0，查看未截断亮度和角度扫描。
3. 对照曝光补偿、异常输入位姿、源图饱和高光和弱观测区域。
4. 保留正式 SH 3，不默认整体压暗、统一删亮点或将真实灯光反射视作错误。

默认角度扫描只覆盖前三个验证相机的 ±30°，不保证命中全部异常视角；特定异常诊断需要对应模型、相机和源图。CUDA 冒烟及短训练只能验证链路，不能接受细节画质。

自动测试与分段检查通过后，先完成人工人物、高亮和细节对照，再执行各场景完整的新 Capture/Run。gsplat 不自动成为默认。人工验收应保存与具体模型哈希对应的记录。

## 7. 独立选帧工作台（select-sharp）

`Invoke-Gsdb select-sharp` 在 Run 流水线之外，用同一套抽帧、打分、投影和遮罩代码从**显式路径**的原片里挑最清晰帧并剔除人物遮挡严重的视图，用来在正式登记之前判断一段素材值不值得做。不写 Capture/Run 清单，不训练，不是画质验收路径。

```powershell
Invoke-Gsdb select-sharp --source $Source --out 'D:\work\pick' `
  --window 0..30 --candidate-fps 5 --mask-threshold 0.01
```

- `--source` 可重复；源类型按扩展名和画幅推断（`.insv` → INSV，2:1 → 全景视频，横屏 → 透视视频），也可用 `--kind` 强制。
- `--window START..END` 作用于每个源并按各自时长截断；不同源需要不同窗口时分两次执行到不同 `--out`。
- `--views/--view-fov/--view-size/--crop-bottom` 决定全景切分，默认与第 3 节表格一致；透视源不投影。
- `--target auto|flat|planar|all` 决定遮罩作用在整图还是投影视图上，auto 在有投影视图时按视图判。
- 分阶段执行用 `--stage probe|extract|select|project|mask|filter|all`。每步幂等，中断后重复同一命令即从断点继续：候选按内容哈希校验，投影跳过尺寸已对的视图，遮罩复用已有结果，`kept/` `rejected/` 按当前阈值重新对账而不是叠加。
- 默认在选帧完成后删除候选帧（`kept`/`selected` 是硬链接，数据不丢）；需要保留加 `--keep-candidates`。

产物为 `<out>/<源名>/{prepared,selected,planar,kept,rejected}` 与 `<out>/summary.json`（含各源哈希、窗口、保留/剔除计数与被剔除帧的遮挡比例）。
