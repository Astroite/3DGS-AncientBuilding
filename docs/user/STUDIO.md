# GS-Studio：正式工作台与训练原型

## 当前正式工作台（代码已实现，运行未验收）

入口为 `scripts/gs-studio.ps1`。项目树的「操作」菜单依次提供地点、场景、Capture 创建；Capture 支持探测、ingest、外部原片重新定位和创建 Run。新 GUI Run 默认要求人工遮罩审核，工作进程在阶段边界自动推进，遇到遮罩、QA、失败或分段选择时停下。菜单的「阶段后停止」只在当前阶段完成后生效，关闭运行中的窗口会先请求停止。意外退出且 Run 不再持锁时显示「待检查」；继续前重新核验证据，不能凭残留文件显示成功。

遮罩审核使用 Qt 原生窗口，审核记录绑定当前遮罩哈希；选择当前 primary 或 repair attempt，完成后执行「固定当前遮罩」，再由「继续到下一审核点」推进。Run 页分别显示分段可训练状态与全路线覆盖；QA 接受后可从通过的分段创建 gsplat 实验，Postshot 需要显式选择。失败或中断的 gsplat 实验可通过「检查并恢复训练」选择原实验，核对训练包、配置和调度记录后续跑；完整产物可回填 Run 记录。训练日志中的实际 step/loss/Gaussian 数量显示在「活动」页；gsplat 每 500 步写一次固定验证相机的源图/渲染预览，与训练配置分开。训练成功且 PLY 哈希匹配后，使用「打开训练模型」进入 Qt 查看与编辑页；编辑状态保存在实验的 `edits/` 下，导出必须选择新 PLY，原始 `model.ply` 不被覆盖。

GUI/CLI 共用 `gsstudio.application.operations`；长任务运行于独立 worker 进程，界面通过结构化事件更新而不解析 CLI 输出。原片记录外部路径、字节数和 SHA256；重新定位必须按原顺序提交完全一致的文件。当前继续沿用单一路径 Run schema，不提供历史 Run 迁移。

**验证状态：**本轮仅做静态代码和差异检查。此前 D-02 对 CPU/GPU 测试及应用验收的暂停仍生效。真实 Capture、RealityScan、训练、DPI、模型预览、编辑导出和故障恢复均待明确恢复运行验证后验收。正式 Qt 工作台尚缺可交互的训练中视角/源图对照及拖拽变换手柄；下文 Tk 原型入口仍供这些能力的实现参考，不能作为正式界面验收依据。

## 旧训练与编辑原型

本文说明 GSStudio 现有 Tk 原型的实际入口和限制。全流程 GUI 的产品目标与阶段见[项目路线图](../architecture/GS-STUDIO-PROJECT-ROADMAP.md)，工程整合见[工程路线图](../architecture/GS-STUDIO-ENGINEERING-ROADMAP.md)；原型不代表正式界面或产品验收。PySide6 正式工作台入口为 `scripts/gs-studio.ps1`，见[架构决策记录 D-12](../architecture/GS-STUDIO-DECISIONS.md)；下文只记录尚未被 Qt 完整替代的原型细节。

此原型首版实现 RealityScan 输出 → 可视化导入 → 原生 gsplat 训练/预览 → 快照编辑 → PLY 导出。原型自身不调用 Postshot、不重新对齐相机，也不处理视频；正式工作台的流程和后端选择见上文。

本节界面为历史功能原型。[UI 风格探索](../architecture/STUDIO-UI-PLAN.md)记录正式 UI 的静态方案。按 D-02，CPU/GPU 测试及应用验证继续暂停；下文验证命令仅作历史记录，未经用户另行明确要求不执行。

## 启动

使用已配置的 Windows Python 3.10、NVIDIA 和 `.venv-gsplat` 环境：

```powershell
& 'E:\Projects\3DGS\GSStudio\prototypes\tk_studio\studio.ps1'
# 仅预填数据集，不自动导入或训练：
& 'E:\Projects\3DGS\GSStudio\prototypes\tk_studio\studio.ps1' -Dataset 'D:\path\to\dataset'
# 恢复保存的项目：
& 'E:\Projects\3DGS\GSStudio\prototypes\tk_studio\studio.ps1' -Project 'D:\path\to\studio-project\project.json'
```

入口尊重 `GSSTUDIO_GSPLAT_PYTHON`。GUI 使用 Tk 和 Pillow，无浏览器服务器、账户或 CDN；Pillow 已在现有训练环境中。新环境先按 `scripts/bootstrap-gsplat-windows.ps1` 配置训练依赖，必要时安装本项目的 `studio` extra。不要在普通主环境中启动 GPU 训练。

## 导入与准备

1. 拖入根目录，或选择数据集、images 和 COLMAP 目录。COLMAP 支持 BIN/TXT 三件套、`colmap` 与 `sparse/<模型>` 结构；多个模型必须明确选择。
2. 首版相机支持 PINHOLE、SIMPLE_PINHOLE。畸变相机必须在前置软件导出去畸变图片和匹配参数；界面会阻断不支持的相机。图片支持 JPG/PNG、中文路径与子目录。
3. 可选 masks 目录，必须逐图唯一匹配、尺寸相同、0/255 二值。默认白色保留，可改为白色忽略。源图红色覆盖区域表示不参与训练；不把被遮挡区域训练成黑色。
4. 检查摘要与错误列表。未匹配图片会列为未参与训练，不静默忽略；坏图、缺图、非法相机和遮罩错误阻断训练。
5. 在图片列表点击相机，或在视口双击相机位置，检查输入图和对应视角。

已有工作流应导入 `training-data/<segment>` 的 `dataset.json` 共享包，保留训练/验证全景分组、遮罩和初始化点。带分段 QA / mask-final / Run 清单的原始重建根目录会被拒绝，防止替代原有门禁。普通外部 COLMAP 数据不被声称通过了 GS Studio 路线 QA。

普通 COLMAP 输入按 `frame_数字` 分组，每八组留一组验证；少于八组留最后一组。无此分组命名时按独立图片处理，时间是序号。如果这些图片实际来自同一全景，请使用带正确分组的共享包。初始化颜色只从未遮罩训练观测计算，至少保留三个具有两次训练观测的点。

## 训练与观察

- 选择“正式训练”或“快速预览”。快速预览默认 1,000 步、最长边 800，不能作为画质验收。正式训练使用原分辨率、SH3、光度补偿；步数可调整。
- 新实验自动生成唯一项目目录，复制并规范化不可变训练数据；原始图像不修改。开始前检查 CUDA、输出目录，并在获得工作区 GPU 锁后显示可用显存。
- 支持暂停、继续、保存检查点和提前结束。等待 GPU 时也可结束；关闭训练窗口会请求保存成果后退出。
- 重新打开项目，保持原配置后点击“继续 / 恢复”。更改关键配置必须新建实验。故障只从最近完整检查点恢复，不自动降低分辨率或 SH。
- 训练中由原生 gsplat 渲染真实高斯预览；损失、步数、高斯数量、速度、耗时和显存持续更新。剩余时间是估算，包含暂停等经过时间的影响。
- 自由视角支持左键旋转、右键平移、滚轮缩放、WASD/QE 漫游。输入相机对照模式下，左键拖动和滚轮共同平移/放大源图与渲染；右键进入自由视角。
- 预览边长和刷新频率只影响查看器，不更改训练设置。训练补偿与显示区分：当前显示固定为 0 EV，无自动色调映射。

GUI 不自动运行 LPIPS 或下载评估权重；记录 `evaluation_status=not_run`。训练完成不代表通过人工画质验收。

## 模型编辑和项目

切换右侧“模型编辑”页。编辑针对完成或提前结束后的独立快照；活动训练期间只能观察、暂停或结束，不能直接修改优化器里的模型。

- 矩形选择支持反选、清空、删除和高亮。穿透选择选取矩形内所有高斯中心；关闭时保留每个像素最近的中心。这是中心拾取，不是完整透明度可见性求解。
- 三维裁剪框通过六个坐标调整，预览显示框体并高亮将删除的高斯，支持保留框内/框外。首版没有拖拽变换手柄。
- 平移、XYZ 旋转、统一缩放围绕世界原点进行；提供 Z-up → Y-up 方向预设。旋转会同步更新高斯四元数和 SH 系数。
- 删除、裁剪、变换支持撤销重做，且历史可随项目保存。原始训练快照不修改，导出必须选择新文件。
- Gaussian PLY 使用标准二进制 SH3 属性布局；较低阶输出保留相同字段布局、未使用系数为零。首版不接受任意属性布局、压缩 PLY、SPZ/SOG。
- 项目包含训练数据副本和相对路径，因此整个目录搬移后直接打开 `project.json`，不依赖原图位置。原始路径只作为溯源记录。不要单独搬走项目中的 dataset 或模型文件；校验失败会拒绝恢复。

项目主要记录：`project.json`、`dataset/dataset.json`、`dataset/source.json`、`training/config.json`、`checkpoint.pt`、`studio-progress.jsonl`、`training.json`、失败说明、原始快照和编辑历史。文件存在不等于训练成功；观察 `training.json` 和项目状态。普通 PLY 导入不会加载外部训练检查点。

## 验证

```powershell
Set-Location 'E:\Projects\3DGS\GSStudio'
& .venv\Scripts\python.exe -m pytest -ra
# 每次指定一个尚不存在的目录：
& .venv-gsplat\Scripts\python.exe tests/manual/check-studio.py --output ..\studio-validation\gui-new
& .venv-gsplat\Scripts\python.exe tests/manual/check-studio.py --output ..\studio-validation\gpu-new --gpu
```

GUI 检查只截图测试窗口本身。GPU 检查用合成数据验证第 2 步暂停、第 8 步提前结束、恢复至第 12 步、实时预览、PLY 导出与项目重开；它遵守工作区 GPU 锁，30 秒仍占用即退出。

2026-09-13 已完成的检查：CPU/既有回归、窗口布局检查、真实 segment-007 共享包导入（256 张图片、226 张训练图、224,480 点，无导入错误）。GPU 集成尝试因现有工作区任务占锁超时，尚未通过；真实训练画质及独立查看器横评仍待 GPU 可用后验收。性能没有以百万级模型实测定级。

P1 的 ROI 训练、批量队列、跨实验对比、Agent 自动化接口和压缩格式未实现。
