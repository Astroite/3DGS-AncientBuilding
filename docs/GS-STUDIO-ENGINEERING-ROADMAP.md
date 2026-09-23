# GS-Studio 工程路线图

更新：2026-09-22。本文规划 APP 原位升级的工程顺序，不是已经实施的架构或运行说明。项目阶段、需求与验收见[项目路线图](GS-STUDIO-PROJECT-ROADMAP.md)；已定技术决策见[架构决策记录](GS-STUDIO-DECISIONS.md)；现行命令和边界以[工作手册](CURRENT-WORKFLOW.md)及[维护说明](MAINTENANCE.md)为准。

## 入口和业务边界

正式 GUI 以已选 [A 工作台设计](GS-Studio-Plan/design/specification-a/README.md)为基线，先补齐采集、Run、重建和 QA 页面。Python/gsdb 流水线与本地 gsplat 训练是主线；现有 Tk Studio 只作可复用功能线索。GUI 和 CLI 逐步调用同一业务服务，不通过解析 CLI 终端输出判定状态。正式 UI 框架已定为 PySide6（Qt 6），部署与资源协作方案见 [D-12](GS-STUDIO-DECISIONS.md)；服务边界见 [D-13](GS-STUDIO-DECISIONS.md)。

`APP` 路径、`gsdb` Python 包、`gsdb` 命令和 `GSDB_*` 环境变量暂保留。用户可见标题、启动方式和文档入口按 P1 改为 GS-Studio；旧入口持续可用。GUI 项目浏览只从现行清单和证据读取状态，不能靠目录或 PLY 是否存在判断阶段成功。

## 现行入口到 GUI 的归属

此表是功能覆盖清单，列出未来页面，不表示页面已经实现。

| 现行命令/入口 | GUI 归属与显示重点 | 边界与约束 |
| --- | --- | --- |
| `location init`、`scene init`、`capture init` | 项目浏览与 Capture 创建；已有对象、来源及不可覆盖提示 | CLI 保留；沿用清单身份与原片只读约束 |
| `media probe`、`ingest` | Capture 检查；探测结果、缺项和可执行状态 | 不在界面中推断未知源能力 |
| `preprocess` | Run 创建与阶段面板；新建/恢复、配置摘要、Run ID 和日志 | 只支持现行单路径 Run；旧 Run 不可读、不可恢复 |
| `mask`、`mask-review`、`mask-finalize` | 遮罩阶段与人工审核；primary/repair 和已确认输入分开显示 | 审核绑定当前遮罩身份；不能跳过门禁 |
| `reconstruct`、`qa report`、`qa segments` | RealityScan 与 QA 面板；选中尝试、分段适用性、路线覆盖与失败原因 | 受一次 repair、阶段状态和来源证据约束 |
| `train --segment --backend` | 实验创建、控制和结果；合格分段、后端、配置、检查点与评估 | 新 GUI 实验预选 `gsplat`，可选 Postshot；CLI 默认仍为 Postshot |
| `cleanup` | Run 存储面板；可回收空间、预览、确认、执行与 pending 重试 | 使用现行白名单、消费者校验和 Run 锁 |
| `catalog build` | 项目索引维护与刷新状态 | 索引不得取代源清单作为权威状态 |
| `doctor` | 依赖检查与故障定位 | 可在 GUI 提供入口；首次安装与低层诊断脚本仍保留 |
| `scripts/studio.ps1` | 训练、查看与编辑功能原型 | 正式 GUI 以 A 设计重做，不把原型截图当验收 |
| `postshot-review` | 训练输入检查；确认共享包、遮罩极性与命令就绪 | 输入准备成功不等于训练成功 |
| `qa approve`/`qa reject` | 人工验收结论记录 | 只记录人工结论，不改阶段状态或完整性校验 |
| `select-sharp` | 素材试选工作台；抽帧、清晰度打分、投影与遮罩筛选 | 不写 Capture/Run 清单、不训练，不作为正式画质对照 |

`postshot-prepare`、`postshot-train`、旧 `export` 与 `clean` 已随单路径整理删除，不再保留历史入口；GUI 不为它们补回界面。

Postshot 的 GUI 可选路径须区分“输入准备成功”“Studio GUI 已手动完成”和“CLI 实际训练成功”；许可证不足时不能以进程退出码零或导入包存在宣称训练完成。模型编辑、项目保存及 PLY 导出使用正式 GUI 工作台，并保留原始训练成果与编辑快照的身份区别。

## 实施顺序与共同约束

1. **P1：入口与服务边界。** 盘点 Typer 命令中的参数解析、业务调用和展示逻辑；将可共享的操作、状态读取和错误分类整理为 GUI/CLI 共用服务。先交付只读项目浏览、依赖与日志视图，再接写入动作。具体进程、线程和 UI 框架由 P1 的架构决策记录确定（已定：[D-12](GS-STUDIO-DECISIONS.md)、[D-13](GS-STUDIO-DECISIONS.md)）。只读服务层与 `gsdb status` 已交付；GUI 壳与写入动作未开工。
2. **P2：上游操作。** 按现行阶段前置条件接入 Capture、Run、遮罩、重建、QA、恢复和清理。界面操作从清单、哈希、阶段日志和锁判断可用性；repair、人工审核与清理仍调用原有受控路径，不另写捷径。
3. **P3：实验与预览。** 使用现有合格分段与共享训练包进入本地 gsplat；GUI 为新实验显式写入后端选择，Postshot 是可选分支。显示实际选中数据集、分组、配置、训练/验证状态、检查点和错误。预览质量设置不得改变训练配置；不自动把分段可训练解释为全路线完成。
4. **P4：编辑与导出。** 将独立快照、手柄＋数值输入、撤销重做、保存重开和标准 PLY 交付接入 A 工作台。活动训练、原始模型和编辑版本各有明确身份；不以导出文件代替可续训检查点。
5. **P5：验收后切换日常入口。** 使用目标版本的实际证据完成导入、恢复、画质、资源、DPI、离线和导出检查；更新操作手册，使 GUI 成为日常操作入口。此前现行工作手册继续描述实际 CLI 流程。

状态来源须维持现行语义：Run `manifest.yaml` 与配置/输入哈希、遮罩最终清单、`selected_dataset`、分段 QA、实验 dispatch/训练记录/评估及清理日志共同决定可显示的结果。错误状态保留，不能手改清单或绕过锁、校验、QA、retention 消费者检查。只有一条流水线和一套 Run 配置：旧 Run 不可读、不可恢复，新默认只作用于新建 Run，GUI 不再补历史分支。

## 实施前决策与验证关口

决策的完整记录见[架构决策记录](GS-STUDIO-DECISIONS.md)。

| 关口 | 最迟阶段 | 必须定下的内容 | 状态 |
| --- | --- | --- | --- |
| UI 技术选型与部署 | P1 编码前 | A 设计的视口、缩放、中文路径、Windows 本地打包及资源协作方案 | 已定：[D-12](GS-STUDIO-DECISIONS.md) |
| D-05 项目存储 | P1 写入项目前 | 外部源引用/可选归档、重定位与身份校验，避免原型隐式全量复制成为默认 | 待决策 |
| D-06 分组来源与兼容 | P3 导入/训练前 | 已有可信分组的保留和普通 COLMAP 缺失分组时的可见行为 | 待决策 |
| D-07 拾取语义 | P4 编辑前 | 非穿透可见性的用户可观察规则；沿用已接受的手柄＋数值交互 | 规则待决策 |
| D-08/D-10 质量目标与预设 | P3/P5 前 | 训练预设、代表素材、画质和资源判据；不得把原型默认或短训当正式结论 | 待决策 |

本轮只做静态路线规划。上述关口不是数据格式变更授权，也不解除[暂停运行验证的既有约束](GS-Studio-Plan/planning/DECISIONS-AND-RISKS.md)。
