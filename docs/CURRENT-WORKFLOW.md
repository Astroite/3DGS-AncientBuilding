# 当前全流程执行手册：Windows → RealityScan → Postshot

**2026-09-12 增补：** 新 Run 已升级 schema 5：14 视图首选、0.5% 遮罩阈值、一次局部补选、分段 QA，以及 gsplat/Postshot 共用训练包。新功能和命令见 [Schema 5 执行说明](PIPELINE-V5.md)。下文 schema 4 参数保留用于历史 Run 恢复；不代表新默认。全量运行仍须先完成人工片段对照验收。

核对日期：2026-09-09。适用工具版本 0.2.0、schema 2 采集 / schema 4 新 Run。
基于 APP HEAD `2cae79784911e5f105b8805e7a5cadb1c401c63d` **及当前未提交修复**，不是声称该提交已经包含全部修复。保留工作区改动，不 reset/checkout 回历史版本。

本页是当前操作入口。旧 README 的 WSL、`locations/.../scenes/.../inputs`、强制手工 MP4、`run-009-demo.ps1` 与 `train → export` 示例属于历史流程，不应套用本次 Windows/Postshot 路线。当前路线已实际跑到重建质量失败、独立 Postshot 包成功和 CLI 许可证失败；**尚未验证 Postshot 训练成功或正式发布闭环**。

## 1. 任意目录初始化 PowerShell 会话

只修改第一行安装位置，后续命令不依赖当前工作目录。路径迁移时重新初始化；ID 为参数，不硬编码到程序。建议使用 PowerShell 7。

```powershell
$AppRoot = 'D:\Project\3DGS\APP' # 唯一安装路径；换机器时修改
. (Join-Path $AppRoot 'scripts\session.ps1')
# 自定义 Data 根目录：. (Join-Path $AppRoot 'scripts\session.ps1') -DataRoot 'F:\GSDB-Data'
$LocationId = 'yanguan-ancient-town-20260822'
$SceneId = 'night-walk-4k'
$CaptureId = 'capture-009-4k'
$SceneDir = Join-Path (Join-Path $env:GSDB_DATA_ROOT $LocationId) $SceneId
Invoke-Gsdb --help
```

`session.ps1` 设置绝对 Data 路径、找到 APP 自己的 `.venv` 与本地 MediaSDK helper；`Invoke-Gsdb` 临时进入 APP、执行后恢复原目录，失败抛错。它不自动启动任何阶段。不要从裸 `python`、旧 WSL conda 或任意 PATH 上的 gsdb 启动。

已有环境不重装。全新机器先安装 Python 3.10、NVIDIA 驱动及本机工具，再运行：

```powershell
& (Join-Path $AppRoot 'scripts\bootstrap-windows.ps1')
```

本次使用 RealityScan 2.2、Postshot 1.1.69、MediaSDK 3.1.5 / helper 0.2.0、Insta360 X6。必要时显式设置（不要覆盖已配置的其他有效安装）：

```powershell
$env:GSDB_REALITYSCAN_CLI = 'C:\Program Files\Epic Games\RealityScan_2.2\RealityScan.exe'
$env:GSDB_REALITYSCAN_EXPORT_PARAMS = Join-Path $AppRoot 'tools\realityscan-setup\colmap-export-params.xml'
$env:GSDB_POSTSHOT_CLI = 'C:\Program Files\Jawset Postshot\bin\postshot-cli.exe'
$env:GSDB_MEDIA_HELPER = Join-Path $AppRoot 'tools\mediasdk-helper\build\Release\gsdb-media-helper.exe'
Invoke-Gsdb doctor
& $env:GSDB_POSTSHOT_CLI --help
nvidia-smi --query-gpu=name,memory.free,utilization.gpu --format=csv,noheader
Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free
```

RealityScan XML 见 [导出参数说明](../tools/realityscan-setup/README.md)。导出任务文件用 `registration.txt`，不能用保留名 `images.txt`；需要完整 cameras/images/points3D 三件套，退出码为 0 不能替代文件检查。不要凭空编造 XML 或改变相机参数。

**先确认 Postshot CLI 的 Studio 许可证。** `--help` 成功只证明程序存在，不能证明有 CLI 权限。2026-09-09 实测 CLI 报 `Postshot Studio license required`，仍返回 0；必须同时查日志及 `.psht` 是否生成。没有 CLI 权限时直接走 GUI 导入，避免耗时准备后反复重试许可证错误。不要购买、登录他人账户或绕过许可证。

## 2. 目录与新采集

```text
<workspace>/APP/                          源码、原生 .venv、工具、本文
<workspace>/Data/<location>/location.yaml
<workspace>/Data/<location>/<scene>/scene.yaml
  captures/<capture>.yaml                 源文件、哈希、选段、拼接设置
  prepared/<capture>/<hash>/              可重建缓存
  <run>/manifest.yaml                    Run 状态与配置的权威记录
  <run>/reconstruction-primary|fallback/  图像、遮罩、模型、轨迹报告
  <run>/postshot/                         正式导入包
  <run>/postshot-training/                Postshot 项目及训练日志
```

已有地点、场景、采集跳过 init；已有 Run 恢复直接去第 4 节，不重新 preprocess 新建 Run。

```powershell
Invoke-Gsdb location init $LocationId --name '地点名称'
Invoke-Gsdb scene init $LocationId $SceneId --name '场景名称'
$Source = 'E:\path\capture.insv' # 明确选定的只读源文件
Invoke-Gsdb capture init $LocationId $SceneId $CaptureId --source-type insta360_insv --source $Source --camera-model X6 --output-width 3840 --output-height 1920
Invoke-Gsdb media probe $LocationId $SceneId $CaptureId
Invoke-Gsdb ingest $LocationId $SceneId $CaptureId
```

实际命令名/选项以 `Invoke-Gsdb capture init --help` 为准。X6 可用获批 helper 直接解码，不必先用 Studio 导出 MP4。标准 2:1 全景视频改用 `--source-type equirect_video`，图片序列用 `equirect_sequence` 并明确 `--fps`；多源用重复 `--source`，不要把 16:9 重构视频当全景。要限定选段，在 capture init 增加 `--start-seconds`、`--end-seconds`；不传结束时间使用探测到的完整时长。不修改源文件或覆盖已有采集清单以变更选段，应创建新 CaptureId。

## 3. 新 Run：预处理、遮罩与重建

```powershell
Invoke-Gsdb preprocess $LocationId $SceneId $CaptureId --candidate-fps 5 --selected-per-second 2 --primary-per-second 1 --fallback-per-second 2 --mask-discard-threshold 0.05 --no-mask-review-gate --no-vision-qa
$RunId = '<复制 preprocess 实际输出的 Run ID>'
$RunDir = Join-Path $SceneDir $RunId
Invoke-Gsdb mask $LocationId $SceneId $RunId
Invoke-Gsdb reconstruct $LocationId $SceneId $RunId
```

小样试跑在**新建** preprocess 时增加 `--smoke`：只取前 5 秒、缩小投影，不等于本次全长运行。配置变化创建新 Run，`--resume` 不用于偷偷更改阈值、模型或分辨率。

当前默认：5 fps 候选 / 每秒选 2 张；Primary rank 1 × 8 视图、2048、120°、底裁 20%；Fallback rank 1–2 × 14 视图、1746、110°、底裁 15%。Mask R-CNN 默认人物类别、置信度 0.25、像素阈值 0.50、gamma 0.75、膨胀 24 px、闭合 7 px。最终遮罩占比**严格大于 5%**整对剔除，等于 5% 保留。源遮罩黑色忽略；Postshot 包转换为白色忽略。

自动流程仍需完成 mask-final。人工审核和外部 vision QA 默认关闭；需要时只在新 Run 明确开启相应选项，先查看 `mask-review --help` / `mask-finalize --help`。不得把自动生成 final 当成人工审核完成。开启外部 vision QA 涉及发送抽样图像，需有用户授权。

重建按最终实际图片清单计算注册率及最大组件覆盖率；Primary 不达 70% 才尝试 Fallback（过滤后时间桶不足也可触发）。工具、导出或许可证错误直接失败。Fallback 仍不达门槛或轨迹 QA 有阻断项时停止，不能将“有模型文件”等同于“重建成功”。RealityScan 本次未启用刚性 rig；不要把配置的 use_rig 或导入说明模板当作已实现的后端约束。

## 4. 恢复既有 Run：先检查，再备份，最后续跑

```powershell
$RunId = '20260906T164624Z-97ab61db' # 换成目标 Run
$RunDir = Join-Path $SceneDir $RunId
Get-Content (Join-Path $RunDir 'manifest.yaml')
Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'python|RealityScan|postshot|pwsh|powershell' -and
    ($_.CommandLine -like "*$RunId*" -or $_.Name -match 'RealityScan|postshot')
} | Select-Object ProcessId,ParentProcessId,Name,CommandLine
```

确认不存在同一 Run 的活动进程，尤其工具子进程不一定包含 Run ID。不要因看到其他 Postshot GUI 就将其结束。命令行不可读或进程归属不明时继续核实；不要启动第二个相同重建。

以下备份清单与指标，不复制几十 GB 图片：

```powershell
$Stamp = Get-Date -Format 'yyyyMMddTHHmmssfff'
$Backup = Join-Path $RunDir "recovery-backups\$Stamp"
New-Item -ItemType Directory -Path $Backup -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $RunDir 'manifest.yaml') -Destination $Backup
Get-ChildItem -LiteralPath $RunDir -File -Filter '*metrics*' | Copy-Item -Destination $Backup
foreach ($Attempt in @('primary','fallback')) {
    $Src = Join-Path $RunDir "reconstruction-$Attempt"
    $Dst = Join-Path $Backup $Attempt
    New-Item -ItemType Directory -Path $Dst -Force | Out-Null
    foreach ($Name in @('mask-filter.json','.mask-filter.pending.json','mask-final.json','trajectory-qa.json')) {
        $File = Join-Path $Src $Name
        if (Test-Path -LiteralPath $File) { Copy-Item -LiteralPath $File -Destination $Dst }
    }
}
$LogDir = Join-Path $RunDir 'logs'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Log = Join-Path $LogDir "reconstruct-resume-$Stamp.log"
Invoke-Gsdb reconstruct $LocationId $SceneId $RunId --resume *> $Log
# 长任务用持久终端运行；另一终端执行下一条观察。不要重复提交 reconstruct。
Get-Content -LiteralPath $Log -Tail 12 | ForEach-Object {
    if ($_.Length -gt 500) { $_.Substring(0,500) + ' ... [truncated]' } else { $_ }
}
```

预处理本身未完成时用 `Invoke-Gsdb preprocess $LocationId $SceneId $CaptureId --run-id $RunId --resume`；遮罩阶段中断用 `Invoke-Gsdb mask $LocationId $SceneId $RunId --resume`。选择实际失败阶段，不盲目从头跑。没有进度输出时结合日志时间、CPU、子进程确认存活；图像解码核验可能持续数分钟。

本次修复后的恢复顺序：投影/分割**之前**恢复 completed 或 pending 过滤记录；完整核验结构、阈值、路径、保留文件哈希及额外文件集合后，才删除属于已有 rejected 记录的额外文件。删除可中断续作；不重新建立可信清单。未知文件、保留文件缺失、非人工审核允许的哈希变化或阈值冲突均在删除前报具体路径并停止。不要手工删除 filter/final 或改哈希“修好”错误。

run 指标缺失不等于遮罩未完成。恢复统计优先用 filter 的 `original_image_count` 和已有指标；检测统计缺失标记未知。`equirect-fallback` 已清理不要求重新投影；先看可信记录。严格校验、schema 1–3 和人工审核语义均保留。

## 5. 正式 Postshot 路线：只接收成功重建

```powershell
Invoke-Gsdb postshot-prepare $LocationId $SceneId $RunId --resume
$Dataset = Join-Path $RunDir 'postshot'
Get-Content (Join-Path $Dataset 'IMPORT.md')
```

prepare 要求 reconstruct=succeeded 且 selected_dataset 存在，并继续检查轨迹与来源；当前失败 Run 不能直接用此命令。包中只含已注册图像、同步改名的 COLMAP 二进制模型、转换遮罩、image-map.csv、dataset.json 和 IMPORT.md。可用 `--output` 指定绝对路径；中断续作 `--resume`。逐张解码、哈希核验可能需数十分钟，不能把耗时当成挂起。

### GUI（本机 CLI 无 Studio 权限时的操作路径）

1. 打开 Postshot，同时将包内 `images` 与 `colmap` 拖入导入窗口。
2. 将 `masks` 加入 Image Masks，选择 Remove Occluders（白色忽略）。
3. Camera Poses=Import，Image Selection=Use All；确认匹配数量等于 dataset.json 的 images 数。
4. 检查相机和点云，开始训练并保存 `.psht`。已有外部位姿，不再执行 camera tracking。
5. 需要交换资产时从 Postshot GUI 导出支持的 PLY/SPZ；该操作未在本次验证，导出后检查实际文件、坐标与可视效果。

### 有 Studio CLI 权限后的正式短跑/训练

```powershell
$Project = Join-Path $RunDir 'postshot-training\smoke-1000.psht'
Invoke-Gsdb postshot-train $LocationId $SceneId $RunId --dataset $Dataset --output $Project --profile Splat3 --ksteps 1 --max-splats 1000 --dry-run
# dry-run 不验证许可证。实际启动：
Invoke-Gsdb postshot-train $LocationId $SceneId $RunId --dataset $Dataset --output $Project --profile Splat3 --ksteps 1 --max-splats 1000
```

`ksteps=1` 是 1000 步；max-splats 单位为千，这里上限 100 万。短跑只验证链路，不是质量验收。默认保留训练上下文。已有输出不能覆盖，另取新项目名；项目存在不等于完成，读取对应 `.training.json` / `.postshot-train.log`。

正式训练改用新文件名、移除短跑限制，让 Postshot 采用其默认步数/点数，或使用用户选定参数。要同时导出，给 `postshot-train` 增加 `--export-ply <绝对路径.ply>` **或** `--export-spz <绝对路径.spz>`，两者不能同传。

**当前集成边界：** CLI 没有旧的 `train` 命令。现有 `gsdb export` 仍读取 Nerfstudio 的 `metrics.train.config_path`，不能拿 Postshot `.psht` 直接接它；不自动串 `export → qa report → qa approve`。Postshot 产物的版本发布、QA 清单联动尚待接通，人工看过结果也不能伪造已有 Run 的正式 artifacts。`Invoke-Gsdb catalog build` 只重建索引，不批准成果。

## 6. 用户明确要求跳过轨迹 QA 的独立实验

默认不走此分支。用户已明确授权本次实验时，直接执行，无需为相同授权重复询问。新场景不能继承本次授权。本脚本不修改 CLI/schema/正式门槛，不写回 Run；只在私有内存副本适配准备接口，并跳过 trajectory 校验。文件、遮罩和来源完整性仍检查，审核文件哈希在前后核对。只支持已具备 filter/final/trajectory 报告的 schema 4。

```powershell
$Label = 'trajectory-test-01' # 新实验用新标签
& $GsdbPython (Join-Path $AppRoot 'scripts\prepare-postshot-experiment.py') --run-dir $RunDir --attempt fallback --label $Label --allow-failed-trajectory --reason '用户明确要求跳过轨迹 QA，测试 Postshot 导入' --resume
if ($LASTEXITCODE -ne 0) { throw 'Experiment preparation failed; inspect audit' }
$Dataset = Join-Path $RunDir "postshot-experiments\$Label\dataset"
```

此脚本只准备，不训练。实验目录有逐次 audit 和 TEST-ONLY.txt；dataset.json 的 validation=passed 仅表示文件结构/完整性，**不表示轨迹通过**。不要直接运行先前 `.repair-runs/.../run_test.py`：那是固定本次 ID、会自动训练的一次性脚本。

实验包 GUI 按第 5 节导入。有 CLI Studio 权限、且用户授权测试训练时，直接使用 Postshot 原生命令（正式 GSDB 入口仍会正确阻断失败 Run）：

```powershell
$ExperimentDir = Split-Path $Dataset -Parent
$Project = Join-Path $ExperimentDir 'smoke-1000.psht'
if (Test-Path -LiteralPath $Project) { throw 'Choose a new output project name' }
$CliLog = Join-Path $ExperimentDir ('postshot-' + (Get-Date -Format 'yyyyMMddTHHmmssfff') + '.log')
$CommandArgs = @('train','--import',(Join-Path $Dataset 'images'),(Join-Path $Dataset 'colmap'),
    '--import-masks',(Join-Path $Dataset 'masks'),'--profile','Splat3','--image-select','all',
    '--max-image-size','0','--mask-mode','occluders','--gpu','0','--output',$Project,
    '--store-training-context','--train-steps-limit','1','--max-num-splats','1000')
$CommandArgs | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ExperimentDir 'command.json')
& $env:GSDB_POSTSHOT_CLI @CommandArgs *> $CliLog
$Code = $LASTEXITCODE
if ($Code -ne 0 -or -not (Test-Path -LiteralPath $Project)) {
    Get-Content -LiteralPath $CliLog -Tail 12
    throw "Postshot failed or produced no project (exit=$Code)"
}
Get-FileHash -LiteralPath $Project -Algorithm SHA256
```

## 7. 本次 Run 的接手状态（无需重新分析）

`yanguan-ancient-town-20260822 / night-walk-4k / 20260906T164624Z-97ab61db`，capture-009-4k，X6 完整 293.86 秒，不是旧手册前 90 秒。采集清单源路径目前在 Data 地点目录，实际来源以 captures YAML 为准。

- 遮罩 resume 修复已落地：原 8232 对，保留 6140 对，剔除 2092 对；独立核验保留哈希，清理重新出现的剔除对，filter 哈希保持不变；132 项相关测试通过。
- 续跑 Primary 677/1730=39.13%；Fallback **4496/6140=73.2248%**，通过 70% 注册门槛。
- 2026-09-09 01:21:29（上海时间）因轨迹 QA 3 项阻断结束：588 个预期时间采样中缺 121 个位姿；112→171、530→581 的位移分别约为中位步长 70.73、45.45 倍。跨越缺帧区间，不能声称是真实瞬移；模型单位未标定为米。
- Run failed、selected_dataset=null。模型保留在 reconstruction-fallback/colmap，transforms.json 有 4496 帧；约 526 万稀疏点不是训练好的高斯点。
- 用户随后明确授权跳过 QA 测试。**现成包**：`<RunDir>/postshot-test-qa-skipped`，4496 图像/遮罩/相机、5,259,955 稀疏点，文件校验通过。可直接 GUI 导入，无需重复打包。
- CLI 短训练未实际进行：Postshot 1.1.69 提示 Studio license required，退出 0 但没有 `.psht`；未自动重试，原 Run 清单不变。GUI 导入训练尚未验证。
- 修复记录：`<workspace>/.repair-runs/mask-resume-20260908/REPORT.md`；测试审计：`<workspace>/.repair-runs/postshot-test-20260909/result.json`。训练失败日志：`<RunDir>/postshot-test-training/smoke-1000.postshot-train.log`。

后续对话可直接粘贴：

> 先读 APP/AGENTS.md 和 APP/docs/CURRENT-WORKFLOW.md，使用 scripts/session.ps1 初始化原生 Windows 环境。保留未提交改动。目标 Run 为 20260906T164624Z-97ab61db；resume bug 已修复，重建因轨迹 QA 失败，Postshot 测试包已生成，CLI 因 Studio 许可证失败。不要重新投影、分割、重建或打包。按我这次明确指定的下一步操作；未指定时先报告现成成果与阻断。

## 8. 验证与常见故障

```powershell
Push-Location -LiteralPath $AppRoot
try {
    & $GsdbPython -m pytest tests/test_masking.py tests/test_pipeline.py tests/test_mask_finalize.py tests/test_reconstruction.py tests/test_reconstruction_realityscan.py tests/test_mask_review.py tests/test_vision_qa.py
    if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed' }
    git diff --check
} finally { Pop-Location }
```

只改文档无需重跑昂贵重建。涉及遮罩恢复/重建代码才运行相关回归；运行完成后以结果为准，历史 132 passed 不代表后续变更自动通过。

| 现象 | 下一步 |
|---|---|
| Python `Unable to create process`，指向本机 Python310 | 确認 APP venv 的基础解释器存在；本次是执行沙箱限制，使用工具的执行权限升级，不重建/删除环境 |
| `rg` 不存在 | PowerShell `Select-String` / `Get-ChildItem`，不用为查文件安装工具 |
| mask-filter 清单与文件数不同 | 修复后的 `--resume` 恢复；未知文件/哈希变化先排查，禁止删可信记录重建 |
| equirect-fallback 不存在 | 可能是正常缓存回收；已有过滤记录和注册模型优先恢复 |
| reconstruct 有 colmap 文件但 failed | 看 manifest 与 trajectory-qa.json；物理产物存在不代表阶段成功 |
| 日志一行非常长 | Nerfstudio 转换可能打印全部相机字典；按上面限制单行长度，优先读 JSON/YAML 摘要 |
| Postshot 返回 0 但无 psht | 读日志，尤其 Studio license；不报训练成功、不因退出 0 自动进入导出 |
| Postshot prepare 没有百分比输出很久 | 源图像/遮罩核验和来源校验多次逐张解码；监测 CPU/进程，勿启动重复进程 |
| 自动审批拒绝操作 | 保留已完成成果，报告被拒动作及原因，先做可执行部分，不用其他方式绕过 |
