# 三条技术路线比较

2026-09-13 · [路线建议](README.md) · [版本与证据](SOURCES-AND-DEPENDENCIES.md)

矩阵统一标签：**可复用**＝已读实现可作为基础；**需改造**＝有基础但不满足本需求；**缺失**＝已检查的目标路径没有该能力；**证据不足**＝本次阅读无法确认；**需实测**＝静态证据无法判定结果。任何“可复用”都不等于正式验收通过。

本地列评价现有 1.4.0 基础；gsplat 当前上游的新能力单列说明，不能算成本机已有功能。候选固定提交见来源登记。Postshot 只作为功能与未来画质对照；SuperSplat 的编辑经验不构成训练路线。

## 能力矩阵

| 能力 / 需求 | Python/gsplat＋正式 UI | LichtFeld 二次开发 | Brush 扩展 |
| --- | --- | --- | --- |
| 外部相机/稀疏点初始化 I-02、T-01 | 可复用：本地 Dataset/initialize；支持范围窄 | 需改造：COLMAP 导入存在，但缺点云可随机初始化 [LF10](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/io/loaders/colmap_loader.cpp) | 需改造：COLMAP 已有；自动挑选注册图最多的模型 [B07](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-dataset/src/formats/colmap.rs) |
| 分组和完整性 I-04、NFR-02 | 需改造：共享包可信；普通模型启发式分组 | 需改造：eval 分支按 test_every 重设 split [LF09](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/training_setup.cpp) | 需改造：周期验证划分；不支持的相机存在跳过路径 [B07](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-dataset/src/formats/colmap.rs) |
| 遮罩忽略观测 I-07 | 可复用：masked L1/有效 SSIM 窗口；极性归一 | 需改造：Ignore/Segment 等模式不同，需只映射忽略语义 [LF07](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/trainer.cpp) [LF22](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/losses/mask_loss.cpp) | 需改造：Masked alpha 及反转可用，但错尺寸会缩放 [B08](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-dataset/src/load_image.rs) |
| 曝光/白平衡 T-03 | 可复用：全景共享 RGB gain/offset；更复杂外观需改造 | 可复用：bilateral grid、PPISP；分组语义需改造 [LF04](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/bilateral_grid.cpp) [LF05](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/ppisp.cpp) | 证据不足：已读训练配置/主循环未找到等价训练补偿 [B04](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/train.rs) [B05](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/config.rs) |
| 增密/裁除、SH、抗锯齿 | 可复用：DefaultStrategy 与 SH；AA 为可选渲染参数 | 可复用：多策略、SH、可选 mip_filter [LF02](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/strategies/strategy_factory.cpp) [LF03](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/strategies/mcmc.cpp) [LF06](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/core/include/core/parameters.hpp) | 可复用：生长/裁除、SH、MipSplat 路径 [B04](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/train.rs) |
| 跨进程完整续训 T-05 | 需改造：状态较全；只有一个滚动检查点 | 可复用：模型/策略/外观优化状态；完整恢复一致性需实测 [LF08](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/checkpoint.cpp) | 缺失：已读 export_checkpoint 只写 PLY，不保存 SplatTrainer 优化状态 [B06](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-process/src/train_stream.rs) |
| 暂停/继续/提前结束 T-04 | 需改造：控制标志已有，共享 worker 与资源语义待改 | 需改造：训练暂停已有，A 状态及等待取消须接入 [LF07](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/trainer.cpp) | 需改造：内存暂停已有，不代表关闭后恢复 [B10](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/apps/brush-app/src/ui/training_panel.rs) |
| OOM、损坏点回退 T-06 | 需改造：失败记录已有，缺历史有效点回退 | 证据不足：检查点校验不等于全故障恢复完成 [LF08](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/checkpoint.cpp) | 需改造：先补完整续训，再定义回退链 [B06](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-process/src/train_stream.rs) |
| 自由查看及同视角对照 V-01/V-03 | 需改造：CPU 帧回传与 Tk 对照需重做 | 需改造：视口/外观渲染基础可用，A 同步规则需落实 [LF12](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/rendering/viewport_appearance_correction.cpp) [LF16](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/rmlui/resources/resume_checkpoint_panel.rml) | 可复用：scene/training UI；A 同步与小窗口需改造 [B09](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/apps/brush-app/src/ui/scene.rs) [B10](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/apps/brush-app/src/ui/training_panel.rs) |
| 选择/裁剪/变换/撤销 E-01～04 | 需改造：数学基础已有，手柄/拾取/枢轴待补 | 可复用：选区及变换入口；独立快照隔离需改造 [LF13](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/gizmo_transform.cpp) [LF15](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/selection/selection_service.cpp) | 证据不足：已读 app 页面无法确认完整基础编辑与版本历史 |
| 训练中的独立快照编辑 E-04 | 缺失：当前仅训练返回后创建 Editor | 证据不足：snapshot 服务不直接证明编辑分支与训练隔离 [LF14](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/training_snapshot_service.cpp) | 缺失：已读主训练/导出链没有本项目需要的快照编辑流程 [B06](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-process/src/train_stream.rs) |
| 项目与源数据重定位 O-01 | 需改造：复制型项目，缺外部源重定位 | 需改造：checkpoint 路径重选界面已有，项目/哈希/历史语义仍需对齐 [LF16](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/rmlui/resources/resume_checkpoint_panel.rml) | 证据不足：模型/运行导出不等于完整项目持久化 [B06](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-process/src/train_stream.rs) |
| Gaussian PLY O-02 | 需改造：SH3 导出可用；重载支持布局窄 | 可复用：丰富 PLY 读写；外观模型及变换后颜色需实测 [LF11](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/io/formats/ply.cpp) [LF12](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/rendering/viewport_appearance_correction.cpp) | 可复用：训练导出 PLY；编辑变换后 SH 一致性需实测 [B06](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-process/src/train_stream.rs) |
| 中文路径、DPI、离线完整流程 | 需实测：有路径读取基础，不是端到端证据 | 需实测：官方有 Windows 支持，不代表中文/离线全流程通过 | 需实测：桌面支持不代表所有文件对话框和字体路径通过 |
| 同输入下画质、速度、显存 | 需实测 | 需实测 | 需实测 |

本地证据均在[原型模块审计](PROTOTYPE-AUDIT.md)链接到代码，不使用旧测试替代当前判断。

## 1. 画质机制：实现、配置与未知

### 初始化、增密和裁除

本地模型用导入稀疏点初始化，并在准备阶段限制初始化颜色来源。DefaultStrategy 依据梯度与尺度增密、按透明度等裁除，SH 随训练逐级启用。默认并未启用 antialiased；不能把上游 absgrad、MCMC、外观插件都算作当前已接入功能。本地代码为 native_train.py:123、142、209；上游默认策略见 [G03](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/gsplat/strategy/default.py)。

LichtFeld factory 注册 mcmc、mrnf、improved_gs_plus；已具体阅读 MCMC 的重定位、噪声、裁除及优化状态保存。其他策略本轮只确认注册入口，没有审完所有算法，不能据名称作质量判断。mip_filter 默认 false，外观选项也需实际配置开启。[LF02](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/strategies/strategy_factory.cpp) [LF03](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/strategies/mcmc.cpp) [LF06](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/core/include/core/parameters.hpp)

Brush 主训练循环存在尺度过滤、生长/分裂、低透明度与异常高斯清理，以及 MipSplat 路径。它有实际质量机制，不能因工具轻巧就判画质差；未知的是相同街景和预算下这些机制的结果。[B04](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/train.rs) [B05](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/config.rs)

### 遮罩与初始化污染

“忽略”必须是不让遮挡像素影响优化，不能自动等同挖透明洞或把目标涂黑。本地 masked_losses 明确处理 SSIM 窗口有效性。LichtFeld 的 Ignore 与 Segment/SegmentAndIgnore 有不同的损失和 alpha 约束，接入时不能仅按“有 mask”选择模式；边界窗口如何传播梯度仍待验证。[LF07](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/trainer.cpp) [LF22](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/losses/mask_loss.cpp)

Brush 将 mask 放入 alpha，支持反转；但尺寸不一致会 resize，不能直接满足 I-07。需在进入训练之前严格报告错尺寸，而不是让加载器替用户修正。[B08](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-dataset/src/load_image.rs)

上游稀疏点自带 RGB 不自动满足“只用未遮罩训练观测计算初始化颜色”。LichtFeld、Brush 都需要专门落实或证明该限制；验证图可用于前置 RealityScan 重建，不等于允许它进入训练损失或用于本项目的初始化着色。

### 外观建模与可交付画质

本地 PanoramaExposure 用全景共享参数拟合 RGB 增益/偏移，固定参考并约束时间连续性。LichtFeld bilateral grid 提供图像相关的空间颜色校正；PPISP 保存曝光、暗角、颜色和响应参数，按相机/帧登记映射。二者模型能力更多，但需要检查全景切面的一致性与新视角外观。[LF04](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/bilateral_grid.cpp) [LF05](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/ppisp.cpp)

LichtFeld 视口还可应用额外外观校正，PLY 写入路径与视口后处理不同。未来必须分别记录“带训练补偿的已知相机拟合”“无额外补偿的标准 PLY”“自由新视角”；不能把前者截图直接当作独立查看器交付效果。[LF11](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/io/formats/ply.cpp) [LF12](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/rendering/viewport_appearance_correction.cpp)

gsplat 固定上游提交的 simple_trainer 支持可选 bilateral_grid/ppisp 和 appearance optimization；这是备选继续提升机制的来源。它们不在已安装 1.4.0 原型中自动生效，迁移需要依赖、遮罩、分组和检查点评估。[G02](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/examples/simple_trainer.py)

Brush 配置及训练循环未找到等价曝光/白平衡训练参数。本轮标“证据不足”，后续若确认没有则需要新增；查看器显示曝光不能替代 T-03。[B04](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/train.rs) [B05](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/config.rs)

## 2. 可靠性与恢复

| 状态内容 | 本地原型 | LichtFeld | Brush |
| --- | --- | --- | --- |
| 高斯参数及步数 | 已读保存/恢复实现 | 已读 checkpoint 实现 | PLY 可保存模型；不构成完整训练状态 |
| 优化器与增密状态 | 已保存 optimizer、strategy | checkpoint 调用策略序列化；MCMC 保存优化器/调度器 | trainer 内有 optimizer/refine/rng，已读 checkpoint 导出未保存它们 |
| 外观参数及优化状态 | exposure 与 exposure_optimizer | grid 保存矩阵与 Adam 矩；PPISP 保存参数、矩、步数/学习率/ID 映射 | 等价外观训练状态证据不足 |
| 随机状态及采样一致性 | 显式 CPU/CUDA RNG；实际恢复仍待测 | 本轮未证明全部 RNG/采样游标覆盖 | 完整磁盘续训未建立 |
| 损坏回退及原子提交 | 临时替换单点；缺更早有效点 | 格式/状态校验存在；多代回退闭环证据不足 | 需先建立可续训检查点 |

证据：本地 native_train.py:249～275；[LF08](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/checkpoint.cpp) [LF03](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/strategies/mcmc.cpp) [LF04](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/bilateral_grid.cpp) [LF05](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/ppisp.cpp) [B04](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/train.rs) [B06](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-process/src/train_stream.rs)。

LichtFeld 的检查点读取包含魔数/版本等校验，不能由此推出 OOM 后总能保存、恢复必然等价或输入变化已被完整阻断。MCMC 路径已读的序列化较完整，但本轮未穷尽每种策略、controller 和项目章节的恢复路径。主推荐仍以恢复验证为硬条件。

gsplat 官方示例保存 splats、可选位姿/外观模型，却未在该保存处写训练优化器；加载 ckpt 后走 eval。Python 备选应保留本地完整续训基础，不能简单用新版示例整体替换。[G02](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/examples/simple_trainer.py)

## 3. 数据、相机和导出正确性

- **多模型及未匹配输入**：本地模型选择基础可保留。LichtFeld 需在 A 导入摘要中明确选定模型并禁止缺稀疏点随机降级。Brush select_colmap_model 选择注册图最多的模型，且存在相机错误后跳过图像的路径，应改成明确选择/问题清单。[LF10](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/io/loaders/colmap_loader.cpp) [B07](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-dataset/src/formats/colmap.rs)
- **相机模型**：本地只支持 PINHOLE/SIMPLE_PINHOLE。Brush 已有多种模型转换，LichtFeld 有加载和畸变路径，但本轮没有逐个确认全部模型与训练后端组合。首版候选允许列表仍由 DEV-06 制定，不能声称支持任意 COLMAP。
- **分组**：LichtFeld training_setup 的 eval 分支按 i % test_every 重置 split；Brush 用 split_eval_every。已有可信 manifest 必须优先于这些默认行为；frame 相机身份、训练/验证、曝光共享组是不同概念。[LF09](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/training_setup.cpp) [B07](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-dataset/src/formats/colmap.rs)
- **路径与文件完整性**：中文、空格、子目录、同名文件、大小写歧义需跨加载器/对话框/保存恢复验证。本地 np.fromfile + imdecode 只能证明一段读取实现，不能证明所有路径通过。
- **PLY**：三路线均有导出基础，但通用交付须保留位置、尺度、旋转、透明度、DC 与高阶 SH。整体旋转不能只改 xyz；本地有 SH 旋转实现，上游编辑→导出链仍需实际方向相关颜色检查。外观网络或逐图补偿不是标准 PLY 的通用字段。[LF11](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/io/formats/ply.cpp) [B06](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-process/src/train_stream.rs)

## 4. A 界面可实现性与差距

| A 页面/交互 | Python/gsplat | LichtFeld | Brush |
| --- | --- | --- | --- |
| 三栏主工作台、底部指标与阶段状态 | 替换 Tk 正式布局；运行事件需改造 | RmlUi 页面/样式及 C++ 绑定是改造入口；状态与面板重新组织 | egui/eframe/egui_tiles 可重组布局，训练面板已有 |
| 等大对照、同步缩放平移与恢复自由视角 | 现有 CPU 图像管线需重做体验 | 现有视口及相机外观入口可用；A 对照动作与状态保存需改造 | scene 页面可复用；A 相机切换重置与同步细节需落实 |
| 快照身份、选择、裁剪、世界/局部变换 | 保留编辑数学，新增手柄/枢轴与隔离流程 | gizmo_transform/selection_service 可复用；限制到独立快照，统一缩放 | 完整基础编辑证据不足，开发负担高 |
| 1280/853 逻辑宽度、150% 缩放 | 按 DEV-04 重做 | 将现有固定面板调整成紧凑/抽屉布局 | 需要相应 panel/tile 自适应策略 |
| 中文字体、离线资源 | Windows 字体回退需正式处理 | 有本地 RML/RCSS，已读 fallback 是 Inter，不能当作完整中文字体方案 | 系统中文字体加载、离线打包未确认 |

实现入口证据：[LF13](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/gizmo_transform.cpp) [LF15](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/selection/selection_service.cpp) [LF16](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/rmlui/resources/resume_checkpoint_panel.rml) [LF23](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/rmlui/resources/font_fallback.rcss) [B02](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/Cargo.toml) [B09](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/apps/brush-app/src/ui/scene.rs) [B10](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/apps/brush-app/src/ui/training_panel.rs)。界面能力是可行性判断，不是最终 UI 框架决策或真实 DPI 证据。

LichtFeld selection_service 包含节点范围、裁剪和深度过滤等操作。此处“深度过滤”不能直接推导为 D-07 的像素透明贡献拾取；需要专门确认选择规则，不能把 renderer 中出现 visible mask 名称当作已满足非穿透语义。[LF15](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/selection/selection_service.cpp)

## 5. 工程负担与维护

| 维度 | Python/gsplat＋正式 UI | LichtFeld 二次开发 | Brush 扩展 |
| --- | --- | --- | --- |
| 获取与许可证 | gsplat Apache-2.0；本地 APP 未核实可分发许可声明 | GPL-3.0-or-later 源码；个人自用不以闭源分发要求排除；Windows 预编译获取为付费入口 | Apache-2.0 源码；桌面分发入口可参考官方 |
| Windows 工具链 | 现有 PyTorch 2.1/cu118 wheel 基础；新 gsplat 功能不能保证同依赖可用 | C++23、CUDA 12.8+、CMake>=3.30、MSVC/vcpkg、Vulkan/SDL3/RmlUi 等，工具链跨度大 | Rust 文档要求 1.88+，wgpu/Burn/egui；当前依赖清单含 git 分支和补丁，Cargo.lock 固定其提交 |
| 打包 | Python、GPU wheel、正式 UI 资源的离线封装需做 | 原项目带 USD/FFmpeg/Python 等，本需求不使用视频也不意味着默认构建已移除依赖 | 桌面原生方向较轻；锁文件、着色器及中文字体仍需核对 |
| 预计开发负担 | 数据语义最接近；正式 UI、交互、训练预览与项目改造较多 | 输入合同、A 布局、快照语义和资源协调是重点；算法与编辑基础可借用 | 输入可改造，但续训、外观、项目及编辑缺口叠加 |
| 上游维护 | 升级渲染接口/策略必须检查遮罩和 checkpoint 兼容 | 同时涉及训练、渲染、项目和 UI，跟进面较大 | Burn/wgpu 等联合变更和 git 依赖增加锁定维护要求 |

来源：[G01](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/LICENSE) [LF01](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/README.md) [LF17](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/CMakeLists.txt) [LF18](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/vcpkg.json) [LF19](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/docs/docs/development/build.md) [LF20](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/LICENSE) [LF21](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/THIRD_PARTY_LICENSES.md) [B01](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/README.md) [B02](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/Cargo.toml) [B03](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/LICENSE)。许可证为源码登记，不把获取方式当作技术障碍，也不展开商业分发法律意见。

Brush 的 Burn/wgpu git 依赖已有 Cargo.lock 固定提交，不能把“manifest 指向分支”误写成不可复现；维护风险在更新锁文件和后端补丁时。gsplat 当前核心 setup 要求 torch>=2.7，示例 requirements 则固定 torch 2.9.1 / torchvision 0.24.1 / numpy>=2，均与本地基础不同。[B11](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/Cargo.lock) [G04](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/setup.py) [G05](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/examples/requirements.txt)

本轮不虚报工期、性能或打包体积。LichtFeld 的复用广度支持条件式主推荐；Python/gsplat 的现有数据正确性和状态保存支持备选。可推翻这一判断的运行证据及判据见[验证交接](VALIDATION-HANDOFF.md)。
