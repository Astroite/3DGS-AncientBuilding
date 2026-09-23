# GS-Studio（建设中）

APP 保留现有仓库和目录，正在升级为以 GUI 为日常入口的 GS-Studio。目标覆盖地点、场景、Capture、Run、RealityScan 重建、分段 QA、本地训练、模型编辑与导出。现行 Windows 流水线（360 / 透视视频 → Gaussian Splatting）仍主要通过 `gsdb` CLI 操作；正式全流程 GUI 尚未实现或验收。

**当前操作依据：[Windows 360 重建工作手册](docs/CURRENT-WORKFLOW.md)。** 项目阶段与需求追踪见 [GS-Studio 项目路线图](docs/GS-STUDIO-PROJECT-ROADMAP.md)，入口整合、兼容和决策关口见 [工程路线图](docs/GS-STUDIO-ENGINEERING-ROADMAP.md)，实施前技术决策见 [架构决策记录](docs/GS-STUDIO-DECISIONS.md)。开发、脚本用途、环境版本和不支持的历史形态见 [维护说明](docs/MAINTENANCE.md)；已通过评审的设计和早期技术评估保留在 [GS-Studio-Plan 历史资料](docs/GS-Studio-Plan/README.md)。

当前本地可视化训练与基础编辑原型入口为 `scripts/studio.ps1`。支持外部 COLMAP / 已通过 QA 的共享包、训练预览和快照编辑，使用现有 `.venv-gsplat` 环境。原型能力和验证边界见 [GS-Studio 原型使用说明](docs/STUDIO.md)。

流水线按地点、场景、Capture 和 Run 管理输入、人物遮罩、RealityScan 重建、分段 QA 与训练成果；只有一条流水线和一套 Run 配置，旧 Run 不再可读、不可恢复。新 Run 默认 1 fps 候选、每秒一张、全景 14 视图 / 110° / 1746² / 底裁 15%，严格大于 0.5% 遮罩占比剔除，薄弱区间按需 2 fps 局部补抽一次。透视视频源（如航拍）不投影，1 帧 = 1 视图。阶段验证成功后自动清理可丢弃中间文件，保留一份有效数据。正式外观保留 SH 3；**当前 CLI** 的默认训练后端仍为 Postshot Splat ADC，gsplat 使用独立 Windows 环境。**未来新 GUI 实验**预选 gsplat，并可选择 Postshot；此规划不改变当前命令默认值。有效分段不代表路线覆盖完整。

## 快速入口

已有环境下，在任意 PowerShell 目录执行：

```powershell
$AppRoot = 'D:\Project\3DGS\APP' # 修改为实际路径
. (Join-Path $AppRoot 'scripts\session.ps1')
Invoke-Gsdb --help
Invoke-Gsdb doctor --backend all --require mediasdk
```

首次安装、原片初始化、代表片段、恢复和 GUI 导入均见工作手册。需要 Python 3.10、NVIDIA 驱动、FFmpeg/FFprobe、RealityScan；INSV 另需 MediaSDK/helper，Postshot CLI 训练需要对应许可证。主环境为 `.venv`，共享训练/评估环境为 `.venv-gsplat`。

原片只读，配置和清单记录来源；旧 Run 不再可读可恢复。不要用旧实验报告替代当前文件检查，也不要以短训练或退出码零认定画质通过。代表片段人工验收前不自动运行全场景。
