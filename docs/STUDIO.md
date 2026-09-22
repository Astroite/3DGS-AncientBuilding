# GS-Studio：当前训练与编辑原型

本文说明 APP 现有 Tk 原型的实际入口和限制。全流程 GUI 的产品目标与阶段见[项目路线图](GS-STUDIO-PROJECT-ROADMAP.md)，工程整合见[工程路线图](GS-STUDIO-ENGINEERING-ROADMAP.md)；原型不代表正式界面或产品验收。

首版实现 RealityScan 输出 → 可视化导入 → 原生 gsplat 训练/预览 → 快照编辑 → PLY 导出。不调用 Postshot，不重新对齐相机，也不处理视频。现有 GSDB CLI 和默认后端不变。

当前界面为功能原型。正式 UI 开发前新增 P0 [UI 风格探索](STUDIO-UI-PLAN.md)：先比较三套静态视觉方案，再确定布局与设计规范。按用户 2026-09-13 的补充要求，暂停 CPU/GPU 测试及应用验证，不干扰其他会话任务；下文验证命令仅作记录，未经用户另行明确要求不执行。

## 启动

使用已配置的 Windows Python 3.10、NVIDIA 和 `.venv-gsplat` 环境：

```powershell
& 'D:\Project\3DGS\APP\scripts\studio.ps1'
# 仅预填数据集，不自动导入或训练：
& 'D:\Project\3DGS\APP\scripts\studio.ps1' -Dataset 'D:\path\to\dataset'
# 恢复保存的项目：
& 'D:\Project\3DGS\APP\scripts\studio.ps1' -Project 'D:\path\to\studio-project\project.json'
```

入口尊重 `GSDB_GSPLAT_PYTHON`。GUI 使用 Tk 和 Pillow，无浏览器服务器、账户或 CDN；Pillow 已在现有训练环境中。新环境先按 `scripts/bootstrap-gsplat-windows.ps1` 配置训练依赖，必要时安装本项目的 `studio` extra。不要在普通主环境中启动 GPU 训练。

## 导入与准备

1. 拖入根目录，或选择数据集、images 和 COLMAP 目录。COLMAP 支持 BIN/TXT 三件套、`colmap` 与 `sparse/<模型>` 结构；多个模型必须明确选择。
2. 首版相机支持 PINHOLE、SIMPLE_PINHOLE。畸变相机必须在前置软件导出去畸变图片和匹配参数；界面会阻断不支持的相机。图片支持 JPG/PNG、中文路径与子目录。
3. 可选 masks 目录，必须逐图唯一匹配、尺寸相同、0/255 二值。默认白色保留，可改为白色忽略。源图红色覆盖区域表示不参与训练；不把被遮挡区域训练成黑色。
4. 检查摘要与错误列表。未匹配图片会列为未参与训练，不静默忽略；坏图、缺图、非法相机和遮罩错误阻断训练。
5. 在图片列表点击相机，或在视口双击相机位置，检查输入图和对应视角。

已有工作流应导入 `training-data/<segment>` 的 `dataset.json` 共享包，保留训练/验证全景分组、遮罩和初始化点。带分段 QA / mask-final / Run 清单的原始重建根目录会被拒绝，防止替代原有门禁。普通外部 COLMAP 数据不被声称通过了 GSDB 路线 QA。

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
Set-Location 'D:\Project\3DGS\APP'
& .venv\Scripts\python.exe -m pytest -ra
# 每次指定一个尚不存在的目录：
& .venv-gsplat\Scripts\python.exe tests/manual/check-studio.py --output ..\studio-validation\gui-new
& .venv-gsplat\Scripts\python.exe tests/manual/check-studio.py --output ..\studio-validation\gpu-new --gpu
```

GUI 检查只截图测试窗口本身。GPU 检查用合成数据验证第 2 步暂停、第 8 步提前结束、恢复至第 12 步、实时预览、PLY 导出与项目重开；它遵守工作区 GPU 锁，30 秒仍占用即退出。

2026-09-13 已完成的检查：CPU/既有回归、窗口布局检查、真实 segment-007 共享包导入（256 张图片、226 张训练图、224,480 点，无导入错误）。GPU 集成尝试因现有工作区任务占锁超时，尚未通过；真实训练画质及独立查看器横评仍待 GPU 可用后验收。性能没有以百万级模型实测定级。

P1 的 ROI 训练、批量队列、跨实验对比、Agent 自动化接口和压缩格式未实现。
