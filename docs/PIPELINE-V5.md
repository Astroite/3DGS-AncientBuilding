# Schema 5：360 分段重建与 Windows 原生训练

2026-09-12 实施入口。历史 schema 1–4 的解析、配置和 Run 保留原样；新采集通过 `preprocess` 创建 schema 5。本页记录新功能和验收边界，实际实验状态见 `.repair-runs/pipeline-v5-20260912`。

本轮已执行证据及尚未完成的验收见 [阶段验证记录](PIPELINE-V5-VALIDATION-20260912.md)。MediaSDK、遮罩、重建和训练 GPU 阶段使用跨进程串行锁；不要同时从绕过这些入口的独立脚本启动 GPU 作业。

## 新默认与一次补选

候选 5 fps，每秒初选两张全景；每张投影 14 视图，110°、1746 × 1746、底部裁切 15%。人物遮罩占比 **严格大于 0.005** 时剔除整张投影视图，等于时保留。保留图的黑色忽略/白色保留像素遮罩以 `gsdb_000000.jpg.mask.png` 图层命名送入 RealityScan，并显式设置 `inpMaskOpts=1`。

`alignment-masks.json` 保存逐图名字、图像/遮罩哈希、极性及对齐完成状态。这里的核验是输入配对和 CLI 设置核验；没有把它表述为读取 RealityScan 内部特征点来证明每个被忽略像素均被排除。刚性 rig 未启用；另做同全景相机中心离散 QA。

初次对齐之后，对剩余视图不足三个、未注册采样、长缺口及 QA 异常区间前后两秒，补入尚未使用的 5 fps 候选。`repair-plan.json` 固定补选身份，只对齐一次 repair。阈值不放宽。初轮和 repair 独立做 QA，按通过分段的覆盖时长、有效图片数选择结果，不拼接不同坐标系。

## QA 与训练数据

`segments.json` 分别给出完整性、分段训练适用性和路线覆盖。长于两秒的缺口切段，速度使用实际时间间隔；阈值取 10 倍中位速度和中位速度加 10 倍稳健离散度的较大值。消除投影朝向后比较旋转，检查同全景中心离散；异常边界切段，轨迹不平滑。默认分段至少十个采样、三个不同位置，按该时间范围实际输入图像计注册率，至少 70%。首尾缺失及缺失原因另列，部分分段通过不代表整条路线通过。

统一训练包在 Run 的 `training-data/<segment>` 内：相同图片、像素遮罩、COLMAP 位姿及初始化点集供两个后端使用。验证组按全景采样每八组留一组，避免同一全景泄漏。初始化点仅保留至少两次未遮罩训练观测，最多确定性采样一百万点。图像、相机、位姿及来源哈希不一致时阻断。

```powershell
$AppRoot = 'D:\Project\3DGS\APP'
. (Join-Path $AppRoot 'scripts\session.ps1')
& (Join-Path $AppRoot 'scripts\bootstrap-gsplat-windows.ps1')

# 先用 preprocess 创建新 Run，再 mask、reconstruct；使用实际通过的 segment ID。
Invoke-Gsdb train $LocationId $SceneId $RunId --segment segment-001 `
  --backend gsplat --output (Join-Path $SceneDir "$RunId\experiments\gsplat-photo-on")
Invoke-Gsdb train $LocationId $SceneId $RunId --segment segment-001 `
  --backend postshot --no-photo-comp --output (Join-Path $SceneDir "$RunId\experiments\postshot-photo-off")
```

后端默认仍是 Postshot；其 profile 固定 `Splat ADC`，SH 3、抗锯齿关闭、禁止点集重心变换，另明确设置光度补偿。`--dry-run` 准备可检查的输入和命令，之后可在同一输出目录执行相同命令。已执行目录不能无意覆盖；gsplat 用 `--resume` 恢复最后完成的检查点。实验状态记录在 `dispatch.json` 和新 Run 的 `metrics.training_experiments`，不替代整条路线的人工验收。

报告逻辑升级之后，可用 `Invoke-Gsdb qa segments $LocationId $SceneId $RunId --attempt primary` 重算新 Run 的报告。历史 schema 4 回放必须显式提供位于历史 Run 外的 `--output`；命令不会把重建阶段改成成功。

## Windows gsplat 与诊断

独立 `.venv-gsplat` 固定 PyTorch 2.1.2/cu118 和 gsplat 1.4.0 的官方 Windows 预编译轮子 `1.4.0+pt21cu118`。优先使用该轮子，源码构建才需要完整 CUDA 11.8 工具链；不改既有主环境。使用 `DefaultStrategy` ADC 增密，正式输出 SH 3。

图片按需加载，L1 排除遮罩像素，SSIM 排除接触遮罩的窗口；LPIPS 排除接触遮罩边界的感受野。光度补偿按全景共享 RGB 增益/偏置，固定首组参考曝光，带恒等与时间连续约束。导出的是参考曝光下的标准 SH 模型。没有将遮罩区域涂黑作为观测目标。

默认步数 `max(30000, 30 × 训练图片数)`，增密周期也按图片数调整。记录平均每图采样次数、点数、显存和耗时。OOM 输出可恢复状态，不自动降低尺寸或 SH。每 1000 步原子保存，失败后最多损失未完成的检查点区间。

`evaluation` 含统一验证相机的预测/参考 PNG、PSNR/SSIM/LPIPS、原始渲染最大值/分位数/超 1 比例。前三个验证相机做 ±30° 角度扫描，保存 SH 0 与 SH 3 图及未截断最亮像素值。`diagnostic-sh0.ply` 仅诊断用。人物残留、真实灯光反射和细节退步仍需视觉对照，不能仅靠超 1 阈值删点。

## 实验与验收

```powershell
& $GsdbPython (Join-Path $AppRoot 'scripts\replay-segments-v5.py') --help
& $GsdbPython (Join-Path $AppRoot 'scripts\benchmark-v5.py') `
  --case yanguan-stable --output (Join-Path $AppRoot '..\.repair-runs\pipeline-v5-20260912\benchmark')
# 其他 case：yunxiu-stable、yanguan-gap、yunxiu-pose、yanguan-stable-45；使用独立 capture/Run，可恢复。
```

历史回放用单独输出报告，不修改历史 transforms、QA 或失败状态。旧 primary/fallback 回放是历史基线，不能冒充在相同裁切范围重新对齐的公平实验。

在同一 case 完成新配置的预处理后，给 `benchmark-v5.py` 增加 `--variant old-primary` 或 `--variant old-fallback`，会创建使用相同候选数据哈希的独立对照 Run。它们读取历史投影和遮罩阈值，不启用历史流程未实际执行的 rig 或像素遮罩，也不执行新 repair；所有结果使用同一分段 QA 解释。

`scripts/compare-backends-v5.py --case-file <case.json> --segment <通过的ID> --output <新目录>` 可复现两个后端的补偿开/关对照；`--dry-run` 仅准备数据和命令。分辨率和 SH 不会随测试预算自动降低。该脚本不启动全量重建。

较长有效段可加 `--duration-seconds 30`，从中选择最早通过窗口注册率检查的连续 30 秒，输出独立训练包；`train` 命令也支持该参数。`--backends gsplat --resume` 可单独运行原生对照。复用成功实验前会核验数据包身份、训练预算和模型哈希。

`scripts/continue-validation-v5.py --output <本轮实验根目录> --train` 串行完成预设的剩余代表片段、历史配置对照和完整预算原生补偿开/关实验。`validation-progress.json`、`queue-logs` 和各实验 `comparison.json` 记录进度及失败；它不会启动全场景 Run。

Postshot CLI 需 Studio 许可证。许可证失败即记录失败并保留 GUI 导入包；退出码零不代表训练成功。不能据此声称两后端画质对照已完成。Postshot 内部与 PLY 的显示对比、曝光补偿开关比较，需要真实训练成果。

顺序：自动测试 → 两场景稳定段和异常区间 → 同分段两后端训练/渲染 → 人工确认人物、高亮和细节 → 两场景各一次全量新 Run。**没有人工对照验收时不启动全量。** gsplat 评估验收之后才讨论成为默认后端。
