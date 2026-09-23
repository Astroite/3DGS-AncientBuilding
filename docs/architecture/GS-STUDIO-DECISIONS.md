# GS-Studio 架构决策记录

更新：2026-09-22。本文记录 GS-Studio 实施前必须定下的技术决策及其状态，格式沿用[决策模板](../archive/GS-Studio-Plan/templates/DECISION.md)。阶段、需求与验收见[项目路线图](GS-STUDIO-PROJECT-ROADMAP.md)，工程顺序与关口见[工程路线图](GS-STUDIO-ENGINEERING-ROADMAP.md)；现行命令仍以[工作手册](../user/CURRENT-WORKFLOW.md)为准。历史决策（D-01～D-04、D-09、D-11）保留在 [GS-Studio-Plan 决策表](../archive/GS-Studio-Plan/planning/DECISIONS-AND-RISKS.md)。

技术或设计决定不解除 [D-02 暂停运行验证](../archive/GS-Studio-Plan/planning/DECISIONS-AND-RISKS.md)：本轮没有启动 CPU/GPU 测试以外的训练、重建或应用验收。

## 状态一览

| ID | 主题 | 状态 | 最迟阶段 |
| --- | --- | --- | --- |
| D-12 | UI 技术选型、部署与资源协作 | 已接受（2026-09-22） | P1 编码前 |
| D-13 | GUI/CLI 服务边界与错误分类 | 已接受（2026-09-22） | P1 |
| D-05 | 项目存储与源文件定位 | 已接受（2026-09-23） | P1 写入项目前 |
| D-14 | 工程目录与命名统一 | 已接受（2026-09-23） | 工程重组 |
| D-06 | 分组来源与兼容 | 待决策 | P3 导入/训练前 |
| D-07 | 拾取与可见性规则（交互已定） | 算法待决策 | P4 编辑前 |
| D-08/D-10 | 质量目标、代表素材与训练预设 | 待决策 | P3/P5 前 |

## D-12 UI 技术选型、部署与资源协作

- 日期：2026-09-22
- 状态：已接受
- 决定者与依据：用户在 P1 编码前选定；对齐 [A 深色专业工作台](../archive/GS-Studio-Plan/design/specification-a/README.md)的视口、缩放、中文路径与本地打包要求
- 关联需求和任务：UI、G-01、O-01/O-03、AC-13、AC-20；工程路线图 P1
- 替代的既有决定（如有）：无（Tk Studio 原型继续作为功能线索，不是正式 UI 基线）

### 要解决的问题

正式 GUI 需要以三维视口为中心的多面板工作台，要求 Windows 高 DPI、中文路径与目录、本地离线打包，以及与训练/重建争抢 GPU 时的可预期行为（R-05）。

### 可行选项

| 选项 | 用户价值 | 成本与限制 |
| --- | --- | --- |
| PySide6（Qt 6） | 停靠面板、表格、树、日志等组件成熟，DPI 与 Unicode 路径处理完善，桌面工具常见 | 新增 Qt 运行时体积；需在 `studio` extra 固定版本并处理打包插件 |
| Dear PyGui | 体积小、自带三维视口、与工作线程协作简单 | 停靠布局与密集表单/表格弱于 Qt，中文字体需显式加载，A 设计多栏信息架构成本高 |
| Tk（沿用原型） | 零新增依赖，原型视口与编辑代码可直接迁移 | 停靠多面板、工具栏与 DPI 需手工拼装，A 设计还原度差 |
| 本地 WebView + three.js | 视觉最贴近 SVG 稿，样式自由度高 | 引入前端构建链与 WebView2 运行时，易触碰「无浏览器服务器、账户、CDN」边界，离线校验更复杂 |

以上均为静态比较，未做运行验收，不作为画质、性能结论。

### 决定与理由

正式 GUI 采用 **PySide6（Qt 6）**，范围覆盖 P1～P4 的全部工作台界面：

- **视口**：由原生 gsplat 离屏渲染结果贴到 Qt 控件显示，交互只改变相机参数并请求重渲染；不引入 WebGL/three.js，训练与预览仍走同一 CUDA 栈。
- **缩放**：Qt 高 DPI 缩放为基础，按 1280×800 与 1920×1080 两套布局做收缩规则；比例、字号以 A 规范为准。
- **中文路径**：界面与文件对话框走 Qt 原生 Unicode 路径，磁盘读写继续用现有 `pathlib` 与 `paths.host_path`，不做编码转换层。
- **Windows 本地打包**：PyInstaller onedir，随包带 Qt 插件；完全离线，无浏览器服务器、无账户、无 CDN。首次安装、底层诊断与应急恢复仍用脚本和 CLI。
- **资源协作**：训练、重建与预览渲染在工作线程或子进程进行，继续通过 `gpu_lock.gpu_session` 串行占用 GPU；UI 线程只做显示与输入，不持有 GPU 锁做重渲染。等待 GPU 与训练中状态分开显示。
- **依赖**：PySide6 与 Pillow 一起放进 `studio` extra，CLI 主环境不引入 Qt；入口脚本与 `prototypes/tk_studio/studio.ps1` 的关系在 P1 GUI 壳落地时同步。

### 后续影响

已按本决定落地：`pyproject.toml` 的 `studio` extra 现含 Pillow 与 PySide6（6.x）；启动入口为 `gs-studio`（`gsstudio.interfaces.desktop.app:main`）与 `scripts/gs-studio.ps1`（优先 `GSSTUDIO_GSPLAT_PYTHON` / `.venv-gsplat`，缺省回退 `.venv`）。Tk 原型现位于 `prototypes/tk_studio/`，不参与安装。P5 发布时补 PyInstaller onedir 打包脚本与 DPI 实测证据。本决定只定框架与部署形态，不构成任何画质、性能或兼容结论，也不解除运行验证暂停：界面的真机 DPI、字体与布局尚未目视验收。

## D-13 GUI/CLI 服务边界与错误分类

- 日期：2026-09-22
- 状态：已接受
- 决定者与依据：工程路线图 P1「入口与服务边界」；已交付 `src/gsstudio/application/`（只读部分）
- 关联需求和任务：G-01、C-01、R-01、Q-01、AC-20；工程路线图 P1、P2
- 替代的既有决定（如有）：无

### 要解决的问题

GUI 与 CLI 需要对「有哪些对象、处于什么状态、哪里坏了」给出同一答案，不能靠解析对方终端输出，也不能由目录或导出文件反推阶段成功。

### 可行选项

| 选项 | 说明 | 限制 |
| --- | --- | --- |
| GUI 解析 CLI 文本输出 | 实现最快 | 输出一改就碎，错误无法稳定分类，已被工程路线图排除 |
| 每个界面直接读清单 | 无中间层 | 状态判定规则会散落在两个入口，容易出现两套口径 |
| 共用服务层（选定） | 操作、状态读取与错误分类只实现一次 | 需要划清边界，避免服务层变成第二条流水线 |

### 决定与理由

`src/gsstudio/application/` 是 GUI 与 CLI 的共用面；2026-09-22 初次交付只读部分，2026-09-23 扩展写操作：

| 模块 | 职责 |
| --- | --- |
| `application/browse.py`、`run_status.py` | Location / Scene / Capture / Run、阶段、证据、日志与实验的查询 |
| `application/deps.py`、`errors.py` | 依赖检查视图和错误分类 |
| `application/contracts.py`、`operations.py` | 请求、结果、事件契约和共用操作导入面 |
| `application/projects.py`～`delivery.py` | 创建、素材、Run、审核、实验与交付的单一实现 |
| `interfaces/worker/main.py` | Qt 独立进程调用的结构化事件入口 |

边界规则：

1. **状态只来自清单与证据。** 目录、PLY、报告文件的存在都不表示阶段成功；证据视图只报告 `present`，成功与否仍由 `RunManifest.stages` 决定。
2. **缺陷可见而不是静默跳过。** 缺源文件、缺准备输入、配置哈希不符、历史 schema、丢失清单的 Run 目录都以 `Problem` 返回，并带上可打开的清单/日志/结果路径。
3. **错误分类只做标注。** 不改变哪个操作失败、不改变报错文本主体、不改变退出码，也不改失败与恢复语义；现有模块继续抛出原异常类型，`classify()` 只为两个前端提供稳定代码。
4. **写操作复用现有 pipeline。** 操作层负责参数、身份与入口契约；阶段状态、锁、QA 和 retention 仍由现有模块执行，不另写捷径。独立 worker 不解析 CLI 输出。
5. **只读入口已开给 CLI：** `gsstudio status`（可 `--json`）是该服务的命令行面，供脚本与恢复使用。

### 后续影响

P1 的 GUI 项目浏览、依赖与日志视图直接调用上述模块。2026-09-23 增加共用写操作、独立 worker 及 Qt 操作入口；随后按 D-14 拆分模块并统一命名。CLI 的主要写命令调用同一操作层。事件只报告阶段进度，成功仍以原有清单、训练记录和模型哈希为准。代码尚未运行验收。

## D-14 工程目录与命名统一

- 日期：2026-09-23
- 状态：已接受
- 决定者与依据：用户要求按常规软件工程结构重组，并确认统一改名；范围仅覆盖代码工程

正式包与 CLI 为 `gsstudio`，GUI 命令为 `gs-studio`，环境变量使用 `GSSTUDIO_*`。旧 `gsdb` 导入、命令和环境变量不保留兼容别名。源码分为 `domain`、`application`、`pipeline`、`infrastructure`、`interfaces`、`resources`；测试、脚本、工具、文档和原型分目录管理，完整边界见[工程结构](../development/STRUCTURE.md)。

独立 Data 的 Location/Scene/Capture/Run 清单、Run schema、实验身份和磁盘布局保持现状。结构重组只做静态核对，D-02 暂停运行验证仍然生效。

## D-05 项目存储与源文件定位

- 日期：2026-09-23
- 状态：已接受
- 决定者与依据：用户确认借鉴 ooosplat 的桌面流程，但选择外部只读源引用，避免原片在每个项目里重复复制
- 关联需求：C-01/02、R-01、O-01/03、AC-17/18

Capture 清单保留原始视频或图片序列的绝对路径、字节数和 SHA256；Run 自有内容寻址的候选帧及后续派生数据。创建 Run、探测与 ingest 前核验源身份。原片移动后，通过 GUI「重新定位原片」或 `gsstudio capture relink` 按原顺序提交新路径，只有文件数、大小和 SHA256 全部匹配才更新 Capture 清单。训练模型、检查点和编辑状态保持各自身份；编辑导出不可覆盖原始 PLY。当前不提供隐式全量归档或旧项目迁移。

代码已按此方向实现，尚未执行真实素材、搬移、恢复与磁盘占用验收；D-02 的验证暂停继续生效。
