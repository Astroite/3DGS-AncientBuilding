# GS Studio 工程结构

本仓库是单个 Python 工程。正式安装包为 `gsstudio`；`gsstudio` 提供 CLI，`gs-studio` 启动 PySide6 工作台。独立 `../Data` 保存 Location、Scene、Capture、Run、实验和编辑成果；本次工程重组没有修改其清单 schema、目录或文件名。

仓库根目录只放打包元数据、README、贡献约束和 `gsstudio.ps1` 便捷入口。正式代码在 `src/`，可执行脚本在 `scripts/`，外部工具在 `tools/`，测试在 `tests/`，文档在 `docs/`，Tk 参考实现位于 `prototypes/`。`.venv/`、`.pytest_cache/` 和 `.claude/` 是本机环境或配置目录，不属于发布包；保留原位以免破坏现有环境。

## 源码边界

| 目录 | 职责 | 依赖约束 |
| --- | --- | --- |
| `src/gsstudio/domain/` | 清单模型、配置、Run 状态转换与已准备输入身份 | 不导入界面、进程或文件系统适配 |
| `src/gsstudio/application/` | GUI/CLI 共用查询、创建、审核、Run、训练和交付操作；请求、结果、进度及错误类型 | 编排领域、流水线和基础设施；不依赖 Qt/Typer |
| `src/gsstudio/pipeline/` | 素材准备、遮罩、重建、QA、训练、PLY 编辑与各阶段编排 | 使用领域规则和基础设施适配；不依赖界面 |
| `src/gsstudio/infrastructure/` | 清单持久化、路径与哈希、Run/GPU 锁、外部命令与 MediaSDK/RealityScan/Postshot 适配 | 仅依赖领域层和第三方库 |
| `src/gsstudio/interfaces/` | `cli/`、`desktop/` 与 `worker/` 三个入口 | 业务写操作调用 application；呈现层不得从输出文件推断 Run 成功 |
| `src/gsstudio/resources/` | 随 Python 包发布的遮罩审核静态页 | 通过包内资源路径读取 |

`application/operations.py` 和 `pipeline/stages.py` 是新包内的稳定导入面，只重新导出分模块实现，不包含第二套业务逻辑。CLI 和独立 worker 共享操作与事件契约；GUI 不解析 CLI 终端文本。Run 清单、输入哈希、阶段证据、QA 与实验记录共同决定状态。

## 功能子目录

| 子目录 | 内容 |
| --- | --- |
| `pipeline/input/` | 原片探测、准备、预处理与独立选帧 |
| `pipeline/masks/` | 遮罩推理、审核、固定与阶段执行 |
| `pipeline/reconstruction/` | 投影、COLMAP 产物与 RealityScan 阶段 |
| `pipeline/quality/` | 分段 QA、视觉 QA 与结论材料 |
| `pipeline/training/` | 训练包、gsplat 算法、光度和深度优化、后端对照 |
| `pipeline/editor/` | PLY 查看、编辑快照与导出 |
| `infrastructure/adapters/` | MediaSDK、RealityScan、Postshot、gsplat 与媒体命令适配 |
| `infrastructure/persistence/` | 清单、索引、准备输入和文件持久化 |
| `infrastructure/runtime/` | Run/GPU 锁与外部进程执行 |

`pipeline/stages.py`、`stage_common.py` 和 `retention.py` 跨阶段使用，因此保留在 `pipeline/` 顶层；`infrastructure/paths.py` 和 `path_safety.py` 为各适配层共用。

## 工程配套

- `tests/unit/` 放领域与处理算法测试，`tests/contract/` 放 CLI、worker 与桌面契约测试，`tests/manual/` 放真机与 GPU 验收脚本。`tests/factories.py` 提供隔离 Data 树。
- `scripts/` 放正式启动与环境构建脚本；`tools/` 放单独构建的 MediaSDK helper 和 RealityScan 导出参数。
- `prototypes/tk_studio/` 保留尚未迁入 Qt 的交互线索，不作为安装入口或正式验收依据。
- `docs/user/` 是操作说明，`docs/development/` 是维护说明，`docs/architecture/` 是现行路线与决策，`docs/archive/` 保存历史设计和实验记录。

## 本轮迁移与验证边界

迁移前工作区已有未提交的 GUI、服务层、训练与文档改动，重组时直接保留这些文件内容，没有重置或覆盖 Data。旧 `gsdb` Python 包、命令及 `GSDB_*` 环境变量不提供别名；启动脚本、测试导入、helper 名称和文档已同步切换到 `gsstudio` / `GSSTUDIO_*`。

D-02 暂停运行验证继续生效。本轮只执行静态解析、导入与依赖检查、引用核对、文档链接检查及差异检查；单元测试、Qt 离屏测试、真实素材、RealityScan、训练和 PLY 回读仍待用户明确恢复相应验证后执行。
