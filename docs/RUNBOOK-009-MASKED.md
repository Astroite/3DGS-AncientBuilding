# 009 前 90 秒遮罩版 Demo：明早启动手册

> **现在不要执行本页任何启动命令。** 本页用于今晚完成配置审核；请等到明早准备出门、确认电脑可以持续通电且不会休眠后再启动。本文中的命令会实际解码视频、下载/加载分割模型、生成数千张视图、运行 CPU COLMAP 和 GPU 训练。

本次运行固定使用：

- 地点：`yanguan-ancient-town-20260822`
- 场景：`night-walk-4k`
- 采集：`capture-009-4k`
- 片段：009 视频的 `0–90` 秒
- 主流程：270 张全景帧 × 每帧 8 个透视视图，共 2160 张
- 有界降级：主重建未达门槛时，才运行 180 × 14，共 2520 张
- 人物处理：本地 Mask R-CNN 分割、边缘闭合与扩张；同一黑白遮罩同时用于 COLMAP 和 Nerfstudio
- 最终状态：`needs_review`，脚本不会自动批准结果

本次新 RunId 会把以下关键配置完整写入运行 YAML 并参与配置哈希：

```yaml
preprocess:
  target_frames: 270
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
  max_masked_fraction: 0.45
vision_qa:
  enabled: false            # 获正式授权后才用 --vision-qa 新建运行
  provider: mimo
  model: mimo-v2.5
  minimum_confidence: 0.80
reconstruction:
  registration_threshold: 0.70
  primary:  {frame_count: 270, images_per_equirect: 8,  crop_bottom: 0.20}
  fallback: {frame_count: 180, images_per_equirect: 14, crop_bottom: 0.15}
train:
  method: splatfacto
  max_iterations: 30000
  oom_retry_downscale: 2
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

5. 首次人物分割可能需要联网下载 Torchvision 的模型权重。MiMo 默认关闭，因此本地遮罩流程本身不需要 MiMo API。

6. 先运行环境检查；只有所有关键项通过后才继续：

   ```powershell
   .\gsdb.ps1 doctor
   ```

   `doctor` 会检查 WSL GPU、CUDA、Torch/Torchvision、gsplat 小型反向传播、FFmpeg、CPU COLMAP、写入权限和剩余空间。失败时先停下，不要绕过。

## 推荐：一键启动

确认上面的检查全部通过后，明早执行：

```powershell
Set-Location D:\Project\3DGS
.\scripts\run-009-demo.ps1
```

一键脚本会依次执行：

```text
doctor → ingest → preprocess → mask → reconstruct → train → export → qa report → catalog build
```

脚本会从 `preprocess` 输出中自动取得 RunId，并在任一阶段失败时立即停止。默认导出版本为 `v001`；如果该版本已经存在，启动前明确指定一个尚未使用的版本，例如：

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
  yanguan-ancient-town-20260822 night-walk-4k $RunId --version v001

.\gsdb.ps1 qa report `
  yanguan-ancient-town-20260822 night-walk-4k $RunId

.\gsdb.ps1 catalog build
```

RunId 由“UTC 创建时间 + 配置哈希前 8 位”组成。运行清单和工作目录分别位于：

```text
locations\yanguan-ancient-town-20260822\scenes\night-walk-4k\runs\<RunId>.yaml
locations\yanguan-ancient-town-20260822\scenes\night-walk-4k\work\<RunId>\
```

修改人物置信度、是否启用 MiMo、抽帧数量或其他运行配置，都会改变配置哈希，必须从 `preprocess` 创建新 RunId。不要直接编辑已有运行 YAML；加载时会校验哈希并拒绝被篡改的配置。

## 遮罩阶段实际做什么

`mask` 在 COLMAP 之前完成以下工作：

1. 把 270 张等距柱状全景投影为 2160 张透视图，并生成两个图像下采样层级。
2. 用 `maskrcnn_resnet50_fpn_v2` 检测和分割人物；夜景推理使用固定 gamma，人物边缘执行闭合和 24 px 扩张。
3. 为每张透视图保存同尺寸二值遮罩：白色为可用建筑像素，黑色为人物等忽略区域。
4. 校验图片/遮罩一一对应、尺寸一致、只包含 0/255，并拒绝单张遮掉超过 45% 的异常结果。
5. 生成抽样联系表。MiMo 未启用时只记录“远程 QA 已禁用”，不会发送图片到外部服务。

随后 `reconstruct` 把同一组遮罩通过 COLMAP 的 `ImageReader.mask_path` 排除出特征提取，并把 `mask_path` 写进 Nerfstudio 的 `transforms.json`，训练时继续排除这些像素。

## MiMo：默认禁用，获授权后才启用

明早建议先使用默认的 `--no-vision-qa`。这只关闭远程多模态判定，不会关闭本地人物分割和确定性遮罩检查。

只有在小米客服明确确认你的账号、套餐和目标接口允许“自动化调用 MiMo v2.5 对人物遮罩联系表做多模态 QA”之后，才启用远程闸门。不要猜 Token Plan 的接口地址，也不要把编码工具的非公开接口当作通用推理 API。

获授权后，从客服或正式控制台取得 OpenAI 兼容的 Base URL，在**当前 PowerShell 进程**临时设置环境变量：

```powershell
$MiMoCredential = Get-Credential -UserName 'mimo-token' -Message '输入 MIMO_API_KEY'
$env:MIMO_API_KEY = $MiMoCredential.GetNetworkCredential().Password
$env:MIMO_BASE_URL = 'https://<客服确认的正式地址>/v1'

.\scripts\run-009-demo.ps1 -EnableVisionQa
```

`gsdb.ps1` 和一键脚本只通过 Windows 的 `WSLENV` 转发这两个变量名；密钥值不会成为命令行参数。Base URL 必须是服务方给出的最终 HTTPS 地址，客户端拒绝 HTTP 和重定向，避免 Bearer key 被转发到其他地址。

远程 QA 会调用 `mimo-v2.5`，只发送抽样遮罩联系表，不会发送 `.insv` 或完整视频；但联系表仍可能包含可识别的游客或拍摄者人像，应按外发数据处理并确认服务方的传输、留存和删除条款。模型必须返回结构化的 `pass/fail`；只有 `pass`、置信度不低于 0.80、且未报告漏遮或误遮视图时才放行。API 缺少凭据、超时、返回无效 JSON 或其他任一条件不满足时，遮罩阶段直接失败，COLMAP 不会继续。

结束或失败后清除当前会话中的秘密：

```powershell
Remove-Item Env:MIMO_API_KEY, Env:MIMO_BASE_URL -ErrorAction SilentlyContinue
$MiMoCredential = $null
```

API Key 不得写入 Git、README、脚本、YAML、`.env`、命令参数或日志。仓库只保存环境变量名 `MIMO_API_KEY` / `MIMO_BASE_URL`，不会保存变量值。

启用 MiMo 会改变运行配置和 RunId。不能给已经以 `--no-vision-qa` 创建的 RunId“中途加开”远程 QA；应新建运行。

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
  yanguan-ancient-town-20260822 night-walk-4k $RunId --version v001 --resume
.\gsdb.ps1 qa report `
  yanguan-ancient-town-20260822 night-walk-4k $RunId --resume
```

`--resume` 只复用可验证的完成结果。人物遮罩会逐文件核验后续作；COLMAP 数据库若没有完整阶段标记，会保留旧数据库并在同一 RunId 下创建新的 `attempt-NNN`。投影目录只生成了一部分，或现有图片/遮罩的数量、尺寸、二值范围不一致时，程序会故意拒绝猜测；遇到这种提示应保留旧运行用于排错，从 `preprocess` 创建新 RunId，不要手动拼接、覆盖或删除中间产物。

可在运行 YAML 的 `stages`、`active_stage`、`message` 和 `log_path` 字段中确认停在何处；各阶段日志位于对应的 `work\<RunId>\logs` 下。

## 自动停止边界

流程不会为了“跑出一个结果”而无限调参：

- `doctor`、输入 2:1/解码/哈希或空间检查失败：在预处理前停止。
- 本地遮罩缺失、尺寸错误、非二值或单图遮挡超过 45%：在 COLMAP 前停止。
- 启用 MiMo 后远程调用失败或判定 `fail`：在 COLMAP 前停止。
- 主 COLMAP 的注册率或最大连通模型覆盖低于 70%：只运行一次 180 × 14 降级流程。
- 降级 COLMAP 仍低于 70%：停止，不启动训练，并保留失败分析数据。
- Splatfacto 首次明确 CUDA OOM：只降低一级训练分辨率重试一次；第二次 OOM 或其他训练错误直接停止。
- 任何失败运行都不会覆盖历史成功运行或已有导出。

## 预计耗时与空间

以下是 RTX 4070 Ti SUPER 16 GiB + CPU COLMAP 对本次夜景 4K Demo 的规划区间，不是承诺值。COLMAP 的耗时对纹理、拖影、人物数量、CPU 核数和是否触发降级最敏感。

| 阶段 | 规划耗时 | 主要空间 |
| --- | ---: | ---: |
| doctor / ingest | 5–20 分钟 | 很小；首次 gsplat/模型缓存另计 |
| preprocess（270 张全景） | 10–30 分钟 | 约 2–6 GiB |
| 投影 + 人物遮罩 + 金字塔 | 30–90 分钟 | 约 8–25 GiB |
| 主 COLMAP（2160 图，CPU） | 2–8 小时 | 约 5–20 GiB |
| Splatfacto 30k | 2–6 小时 | 约 5–25 GiB |
| export / QA / catalog | 10–40 分钟 | 约 2–10 GiB |

主流程通常按 **5–15 小时、峰值约 30–80 GiB** 规划。若触发 2520 图的唯一降级重建，整体可能达到 **10–24 小时、峰值约 50–100 GiB**。首次下载权重、CPU 较慢或夜景匹配困难时还会更久。实际耗时、磁盘峰值和显存峰值会写入运行清单与 QA 报告。

因此，明早启动后当天未完成不等于卡死。判断是否仍在工作应查看日志更新时间、CPU/GPU 占用和工作目录增长，不要仅凭控制台一段时间没有新行就中断。

## 完成判定

一键脚本最终打印类似：

```text
Completed <RunId>. Artifacts remain needs_review; no automatic approval was performed.
```

完成后应存在：

- 场景 `exports\<version>` 下的 Gaussian PLY、Nerfstudio 配置、坐标变换、缩略图和预览视频；
- `qa\<RunId>.md`；
- `runs\<RunId>.yaml` 中完整的遮罩、重建、训练、资源和产物记录；
- 从 YAML 原子重建的 `catalog\catalog.sqlite` 记录。

此时运行状态必须保持 `needs_review`。不要在无人值守脚本中追加 `qa approve`。即使后续由 MiMo 完成人物遮罩 QA，它也只负责遮罩闸门，不等于对最终几何、漂浮噪点、夜景曝光伪影、权限或游戏参考价值作最终批准。
