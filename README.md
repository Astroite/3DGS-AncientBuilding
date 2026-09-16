# GSDB：360 视频到 Gaussian Splatting

Windows 原生流水线，按地点、场景、Capture 和 Run 管理全景输入、人物遮罩、RealityScan 重建、分段 QA 与训练成果。

**唯一操作入口：[Windows 360 重建工作手册](docs/CURRENT-WORKFLOW.md)**
开发、脚本用途和历史归档见 [维护说明](docs/MAINTENANCE.md)。

本地可视化训练与基础编辑入口：`scripts/studio.ps1`。支持外部 COLMAP / 已通过 QA 的共享包、训练预览和快照编辑，使用现有 `.venv-gsplat` 环境。操作与验证边界见 [GS Studio 使用说明](docs/STUDIO.md)。

新 Run 使用 schema 6：1 fps 候选、每秒一张全景、14 视图、110°、1746²、底裁 15%、严格大于 0.5% 遮罩占比剔除，以及一次按需 2 fps 局部补抽。阶段验证成功后自动清理可丢弃中间文件，保留一份有效数据。正式外观保留 SH 3；Postshot Splat ADC 仍为默认后端，gsplat 使用独立 Windows 环境。有效分段不代表路线覆盖完整。

## 快速入口

已有环境下，在任意 PowerShell 目录执行：

```powershell
$AppRoot = 'D:\Project\3DGS\APP' # 修改为实际路径
. (Join-Path $AppRoot 'scripts\session.ps1')
Invoke-Gsdb --help
Invoke-Gsdb doctor --backend all --require mediasdk
```

首次安装、原片初始化、代表片段、恢复和 GUI 导入均见工作手册。需要 Python 3.10、NVIDIA 驱动、FFmpeg/FFprobe、RealityScan；INSV 另需 MediaSDK/helper，Postshot CLI 训练需要对应许可证。主环境为 `.venv`，共享训练/评估环境为 `.venv-gsplat`。

原片只读，配置和清单记录来源；历史 schema 继续按原规则解析。不要用旧实验报告替代当前文件检查，也不要以短训练或退出码零认定画质通过。代表片段人工验收前不自动运行全场景。
