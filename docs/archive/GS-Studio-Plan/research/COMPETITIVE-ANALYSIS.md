# 竞品能力与需求取舍

初次调研：2026-09-12；资料复核：2026-09-13。以下是官方资料核验，不是实际横评。链接与证据级别见 [来源登记](SOURCES.md)。

## 工具定位

| 工具 | 已核实的相关能力 | 本项目借鉴 | 不随之扩大的范围 |
| --- | --- | --- | --- |
| Postshot | 外部位姿与稀疏点导入、训练实时预览、遮罩、光度补偿、编辑、ROI 训练与模型导出 | 连贯的导入到训练流程、参数说明、状态反馈 | 视频能力、引擎插件、完整渲染时间线 |
| LichtFeld Studio | COLMAP 训练、检查点恢复、实时检查、选区与变换、撤销重做、PLY/SPZ/SOG 导出和自动化 | 完整本地工作台、训练与编辑的衔接 | 插件生态、MCP、更多算法先列后续 |
| Brush | COLMAP/Nerfstudio 数据输入、实时观察训练、源图对照、遮罩、PLY 查看与 CLI | 轻量训练体验、源图与训练结果对比 | 跨全部设备运行不作为首版要求 |
| SuperSplat | 清理、裁剪、颜色调整，以及查看和发布工具 | 选区、编辑反馈、成果交付体验 | 在线发布、动画、碰撞等不纳入首版 |

以上对应 [Postshot 官网](https://www.jawset.com/)、[LichtFeld 官方仓库](https://github.com/MrNeRF/LichtFeld-Studio)、[Brush 官方仓库](https://github.com/ArthurBrussee/brush) 与 [SuperSplat 官方文档](https://developer.playcanvas.com/user-manual/supersplat/)。

## 对 Postshot 的具体观察

- COLMAP BIN/TXT 支持相机、图片位姿和点三件套。导入外部位姿时跳过选图/跟踪；官方提醒未匹配图片或位姿可能被忽略。本项目应明确列出它们。[导入说明](https://activation.jawset.com/docs/d/Postshot%2BUser%2BGuide/Importing%2BImages)
- 遮罩“移除遮挡物”与“移除背景”语义不同；首版只要求忽略遮挡观测，不能把两者混淆。光度补偿涉及曝光、白平衡等变化。当前文档推荐 Splat3，并把 ADC 列为旧配置，因而本项目不以复制算法名称作为验收标准。[训练配置](https://activation.jawset.com/docs/d/Postshot%2BUser%2BGuide/Interface/Training%2BConfiguration)
- 价格页列出 Free 的导入/训练能力、Indie 的 PLY/SPZ 导出、Studio 的 CLI。当前抓取文本中的金额出现动态占位值，因此不把具体价格写入预算或收益计算。[官方价格页](https://www.jawset.com/shop/)

## 从调研转成产品要求

| 观察 | 本项目选择 | 需求 |
| --- | --- | --- |
| 相机与图片错配直接影响模型 | 导入阶段呈现完整摘要、错误和未匹配项 | I-02～I-05 |
| 用户已有 RealityScan | 不重复实现 SfM；直接训练 | T-01 |
| 仅看训练数字不足以判断质量 | 训练中可自由观察，并与同相机源图对照 | V-01～V-03 |
| 真实数据常有曝光差和遮挡 | 遮罩极性可视化，光度补偿可控 | I-07、T-03 |
| 训练是长任务 | 暂停、继续、持久化恢复及明确失败状态 | T-04～T-06 |
| 成果往往需要清理 | 首版保留基础选区、裁剪、变换和撤销 | E-01～E-04 |
| 导出是交付闭环 | 标准 Gaussian PLY、回读与来源记录 | O-01～O-03 |
| 用户不认可当前 UI | 先探索三种布局与视觉方向，不仅换配色 | UI-01～UI-04 |

这些是基于使用场景的产品判断，不是竞品性能结论。

## 尚不能下的结论

- 没有在相同数据、预算与视角下实测，不能给出训练速度、画质或显存排名。
- 功能覆盖不等于可直接复用源代码；许可证、分发成本和维护负担在未来技术评估阶段单独确认。
- 原型可复用不代表必须自研；LichtFeld/Brush 可作为未来“使用现成工具、二次开发、自研”的比较对象，但本轮不做该决策。
