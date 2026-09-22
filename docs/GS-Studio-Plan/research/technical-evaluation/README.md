# DEV-05：画质优先的静态技术评估

> **2026-09-13 历史评估，不是现行选型。** 当时的 LichtFeld 主推荐保留原文供追溯；用户于 2026-09-22 选择 Python/gsdb/gsplat 为主线，见 [D-03](../../planning/decisions/D-03-GS-STUDIO-ROUTE.md)和 [APP 工程路线图](../../../GS-STUDIO-ENGINEERING-ROADMAP.md)。下文“待决策”均指本评估交付时的状态。

2026-09-13 · **技术评估已交付待决策** · D-03 为建议，尚未接受

## 结论

**主推荐：以 LichtFeld Studio 二次开发实现 A 工作台。备选：保留现有 Python/gsplat 数据与训练基础，替换正式 UI 并改造运行协调。** Brush 保留为参考候选，当前不作为首选或备选。

这是按“画质能力优先，其次可靠性、开发量和维护成本”作出的带前提建议，不是画质排名。LichtFeld 的源码同时提供可选外观建模、增密策略、训练状态序列化和编辑交互；相较于从原型逐项补齐，更适合作为优先验证的完整基础。实际画质、速度、显存和在本机的可构建性均未知。[LF02](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/strategies/strategy_factory.cpp) [LF04](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/bilateral_grid.cpp) [LF05](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/ppisp.cpp) [LF08](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/checkpoint.cpp) [LF13](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/gizmo_transform.cpp)

用户已认可 DEV-04，记录为**设计评审通过、M2 完成**。本轮不改变 A 设计、不修改 APP/Data、不安装或执行候选，不创建新 API、项目格式或通信协议。AC-13 正式应用验收仍未完成。

## 为什么这样选

| 顺序 | 选择依据 | 对结论的限制 |
| --- | --- | --- |
| 画质机制 | LichtFeld 已有 MCMC 等策略、可选 mip 过滤、bilateral grid 与 PPISP；外观补偿参数及优化状态有保存实现 | 可选功能不能当作已启用，更不能换算为 Postshot 之上的画质；正式配置另评 |
| 可靠性 | 模型、策略优化器、grid 与 PPISP 状态有序列化；比只导出 PLY 更接近完整续训 | 随机状态、采样顺序、损坏回退、OOM 和配置一致性仍需验证 |
| A 工作台 | 已有渲染、编辑变换和 RmlUi 页面入口 | 仍需重新组织 A 页面、保存语义、独立快照和小窗口；不是简单换色 |
| 备选价值 | 本地原型已有可信包分组、遮罩边界损失、全景共享补偿和较完整检查点 | Tk、共享训练/预览线程、回传帧、复制型项目与编辑时机需要重做 |

逐项证据见[能力矩阵](CANDIDATE-COMPARISON.md)及[原型审计](PROTOTYPE-AUDIT.md)。

## 主推荐的采用前提

1. **输入合同可落实。** 禁止覆盖已有训练/验证组；不得按图片序号重分组、静默跳图或用随机点替代缺失稀疏点。初始化颜色只取未遮罩训练观测；保留 RealityScan 提供的相机与图片对应关系。LichtFeld 当前 eval 分支会按间隔重设 split，缺 points 路径可退回随机初始化，必须改造。[LF09](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/training_setup.cpp) [LF10](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/io/loaders/colmap_loader.cpp)
2. **外观与交付一致。** 训练补偿、查看器显示曝光和可交付 PLY 外观分别核验。PPISP 有逐相机/帧参数，不能直接等同本地全景共享分组。标准 PLY 导出不能被当作携带全部外观模型的检查点；必须在独立查看器验证未依赖查看器补偿的结果。[LF05](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/ppisp.cpp) [LF11](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/io/formats/ply.cpp) [LF12](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/rendering/viewport_appearance_correction.cpp)
3. **恢复及编辑满足产品语义。** 保存项目不等于保存新检查点；较早有效点可回退；训练实验与编辑快照隔离。已有 snapshot 服务只是复用线索，不能证明 DEV-04 的完整独立编辑流程已满足。[LF08](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/checkpoint.cpp) [LF14](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/training_snapshot_service.cpp)
4. **将来允许运行后，采用隔离环境验证工具链和资源协作。** 当前 Windows 原型使用 cu118，而 LichtFeld 文档要求 CUDA 12.8+、驱动 570+。不能为此升级正在承担其他任务的现有环境。[LF01](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/README.md)

## 切换条件与可能推翻推荐的证据

| 未来证据或评审结论 | 处理 |
| --- | --- |
| 相同输入与显示条件下，LichtFeld 的可交付 PLY 在建筑细节、新视角或高亮上持续明显弱于备选，且合理配置不能解决 | 改以 Python/gsplat 为主；保留对照材料，不按算法名称辩护 |
| 分组/遮罩/初始化颜色的严格语义需要侵入性大改，破坏上游训练能力或无法可靠维护 | 转向现有数据与训练基础；这部分是备选最明确的优势 |
| 主路线无法保证有效恢复、独立编辑不污染训练或导出 SH/方向一致，而备选可以补齐并验证 | 切换备选，优先完成正确性闭环 |
| 所需驱动/工具链无法在用户允许的隔离条件下使用，或持续构建失败且无可接受修复 | 切换备选；不擅自改变本机环境 |
| 仅发现 UI 改造多、编译慢或功能数量不同 | 不单独推翻推荐；重新评估维护负担，画质优先原则继续生效 |
| Brush 后续增加完整续训、外观补偿和独立编辑，或取得明显更好的实测画质 | 重开候选评估，不在本轮自动提升其优先级 |

## 首个实现里程碑与推进顺序

**首个实现里程碑建议：A 工作台的可信导入与只读空间预览。** 路线确认、DEV-06 行为边界及后续架构说明完成后，再开工正式 UI、目录/模型选择、完整配对与分组清单、遮罩预览、相机/稀疏点联动、项目保存重开及源文件重新定位。关联 DEV-07～10，I-01～07、O-01/O-03。256 / 226 / 30 是既有设计示例规模，不是本轮重新导入结果。

在全面扩展训练与编辑页面前，优先安排[关键假设验证](VALIDATION-HANDOFF.md)中的输入正确性、质量与恢复关口；只有用户另行允许相应运行才执行。确认这些关口后再推进 DEV-11～17，最后 DEV-19～22 验收与交付。当前仅提交路线建议和验证材料，不启动该里程碑。

## 给未决项的输入

| 决策 | 建议输入，均待确认 |
| --- | --- |
| D-03 技术路线 | 主推荐 LichtFeld 二次开发；备选 Python/gsplat＋正式 UI；不锁死具体算法预设 |
| D-05 项目存储 | 优先外部不可变源数据引用，项目管理自己的实验/检查点/编辑版本；移动后重新定位并核验身份。归档副本可另议；不能照搬原型全量复制 |
| D-06 分组来源 | 已有可信分组优先且不可重写；普通 COLMAP 不包含本项目完整分组语义，必须显示来源并确认缺失信息。不能默默按序号或文件名决定 |
| D-07 拾取规则 | 手柄＋数值交互已接受。非穿透选择需要明确可见性定义；高斯中心近似不可冒充透明贡献判定。LichtFeld 选区与深度筛选是技术输入，不代表规则已定 |
| D-10 训练预设 | SH3、启用补偿保留；具体策略、抗锯齿和外观组合留待同条件比较，不凭默认值冻结 |

## 交付索引

- [PROTOTYPE-AUDIT.md](PROTOTYPE-AUDIT.md)：本地模块保留／改造／替换、代码位置与需求。
- [CANDIDATE-COMPARISON.md](CANDIDATE-COMPARISON.md)：三路线能力、画质机制、A 适配与工程负担。
- [SOURCES-AND-DEPENDENCIES.md](SOURCES-AND-DEPENDENCIES.md)：固定提交、许可证、环境元数据、工作树状态及证据限制。
- [VALIDATION-HANDOFF.md](VALIDATION-HANDOFF.md)：未来验证场景、判据与记录要求，全部未执行。

本次只读源码和依赖元数据、联网阅读官方材料并核对文档链接。未编译、安装、启动应用、运行测试/训练或后台任务；没有新增运行验证证据。
