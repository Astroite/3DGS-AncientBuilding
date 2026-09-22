# 现有原型静态审计

2026-09-13 · 关联 I/T/V/E/O、DEV-07～17 · [结论](README.md)

审计对象是 APP 当前工作树，不是已提交版本或通过验收的产品。HEAD、未提交状态与源码 SHA256 见[证据登记](SOURCES-AND-DEPENDENCIES.md)。CodeGraph 未定位到当前未跟踪的 studio 文件后，回退到直接源码阅读；未更新索引。本轮未导入 Python 模块或执行其代码。

“保留／改造／替换”是工程处置；能力标签仍使用“可复用／需改造／缺失／证据不足／需实测”。跨语言路线中的“保留”通常指保留规则和已有证据，不表示将 Python 文件直接嵌进 LichtFeld。

## 模块处置

| 模块与位置 | 处置 / 能力 | 可保留部分 | 必须改变或明确的边界 | 关联需求 |
| --- | --- | --- | --- | --- |
| [studio_data.py](../../../../src/gsdb/studio_data.py)，read_model:78、Dataset:165 | 保留并改造 / 需改造 | BIN/TXT、显式模型列表、相机/图像检查、中文路径读取、哈希核验、共享包分组 | 普通 COLMAP 只接受 PINHOLE/SIMPLE_PINHOLE；文件名/序号推断不能成为可信分组 | I-01～06、NFR-02 |
| 同文件 keep:277、prepare:289 | 保留语义、改造准备流程 / 需改造 | 极性归一、遮罩尺寸核查；非包输入初始化颜色只用未遮罩训练观测，要求至少两次观测 | prepare 重新写入所有图片及 masks；分辨率改变时内参同步，但不能当作外部引用方案 | I-07、T-01、O-01 |
| [native_train.py](../../../../src/gsdb/native_train.py)，masked_losses:54、PanoramaExposure:72 | 保留 / 可复用 | masked L1；SSIM 仅使用全部有效的窗口；按全景分组的 RGB 增益/偏移、参考约束、时间平滑 | 只是已读代码，不等于新版本损失及画质已通过；不是 bilateral grid 或完整成像管线 | T-03、I-07 |
| 同文件 initialize:142、train:209 | 保留并改造 / 需改造 | 稀疏点初始化、Adam、DefaultStrategy、SH 阶数渐进；实际配置与版本记录 | 新策略和外观能力需独立迁移评估；参数暴露须符合 A 的预设/高级设置 | T-01～03、O-03 |
| 同文件 save:265、恢复分支 | 保留并改造 / 需改造 | 参数、优化器、策略、曝光参数及优化器、CPU/CUDA 随机状态、步数、配置身份 | 单一 checkpoint.pt 覆盖保存，没有多代有效点回退；成功与失败状态须按实际提交确认 | T-04～06 |
| [studio_runtime.py](../../../../src/gsdb/studio_runtime.py)，_loop:60、_observer:133、start:171 | 改造 / 需改造 | 工作区 GPU 锁、等待取消、暂停/停止标志、进度与错误事件 | 训练/预览共享 worker，暂停仍处于 gpu_session；不能把暂停解释为资源已释放 | T-04～06、V-01/05 |
| 同文件 _preview:119 | 替换正式预览接入 / 需改造 | 相机请求、预览频率与修订号可作行为参考 | GPU 渲染→CPU numpy→8 位图回传 Tk；同步 observer 会阻塞下一训练步，资源代价未知 | V-01～05 |
| 同文件 save_project:261、open_project:292、save_as:323 | 改造 / 需改造 | 保存数据、配置、编辑状态和模型引用；来源校验 | 复制型项目、另存再次复制；缺完整外部重定位、历史检查点选择与精确恢复状态组织 | O-01/O-03 |
| [studio_edit.py](../../../../src/gsdb/studio_edit.py)，Editor:21 | 保留数学与历史基础，改造交互 / 需改造 | keep 集合与矩阵历史、撤销重做、裁剪预览、独立 PLY 原始文件保护 | 最近中心拾取、轴对齐裁剪与世界原点变换不足以直接覆盖 A 手柄/枢轴/局部坐标 | E-01～04 |
| 同文件 materialize:120；[ply.py](../../../../src/gsdb/ply.py)，_sh_rotation_matrix:160 | 保留 / 可复用，结果需实测 | 同时处理高斯方向和 SH 旋转；统一缩放、非有限值检查 | 内部 SH 排列、旋转约定与独立查看器外观仍需目标版本验证 | E-03、O-02 |
| native_train.py export_ply:89、load_ply:108；studio_edit.py export:138 | 保留并改造 / 需改造 | 标准颜色/SH 属性、临时文件写入、保护来源 | 编辑器限制 62 float 的指定二进制 SH3 布局；不是任意 Gaussian PLY 兼容实现 | O-02/O-03 |
| [studio.py](../../../../src/gsdb/studio.py) | 替换 / 需改造 | 操作入口、已有文案与功能映射可参考 | Tk/Pillow 功能原型不是 A 正式工作台；布局、组件、同步对照、抽屉与手柄需要正式实现 | UI-02～04、V-03、E-01～03 |
| [bootstrap-gsplat-windows.ps1](../../../../scripts/bootstrap-gsplat-windows.ps1)、native_train.py configure_windows_cuda:25 | 保留隔离原则、改造环境适配 / 需改造 | 现有独立训练环境及固定 wheel 来源 | CUDA/MSVC/架构硬编码不能推广为正式安装策略；候选升级不在当前环境执行 | NFR-03、NFR-06 |

## 训练、预览、复制与快照的关键链路

源码显示 start 在单个 worker 中调用 train；每轮 observer 同步完成进度、检查点请求及预览。_preview 把 GPU 图像转到 CPU，再量化为 uint8 放入事件队列，Tk/Pillow 使用该图像。这说明界面主线程与 worker 分开，却不说明训练和预览独立。没有实测帧率或耗时，不能给出“足够流畅”的结论。定位：studio_runtime.py:21、60、119、133、215；native_train.py:278。

暂停循环位于 observer，外层 gpu_session 尚未退出。它仍允许按视角变化渲染，也可能继续持有训练张量和工作区锁。因此 A 的“已暂停”“保存完成”“可让出资源”必须分别定义，不能通过按钮名称暗示资源已经释放。等待可取消的已有行为值得保留。

闲置模型查看时，_loop 对编辑模型 materialize 后创建整份 CUDA tensor；渲染完成清空 params 并释放缓存，下次视角变化可能再次物化/上传。此设计用于与其他工作区任务协作，代价是潜在全量数据转换和传输。不要在未知规模下宣称成本很低，也不要为了性能擅自取消资源协作。

train 返回之后才复制 model.ply、构造 Editor；edit 在没有 Editor 时拒绝操作（studio_runtime.py:218～255）。所以原型有独立模型编辑，却**缺失训练中选择独立快照并持续编辑的完整流程**。需要明确快照捕获步数、原始模型身份、版本和失败行为；本轮不规定实现协议。

## 保存和恢复边界

native_train.py 的保存内容比上游 simple_trainer 示例更接近续训需要：优化器与曝光优化器、增密状态、随机状态均有代码。但同一个 checkpoint.pt 被临时文件替换，只保护提交过程，不能提供“最新点损坏时恢复更早有效点”。OOM 不保证还能新存一次；界面应指向已经有效保存的版本与实际步数。

配置比对与包身份校验已经存在，不能为方便恢复而删除。还需检查恢复后采样与优化状态是否一致、异常保存是否被标成成功、项目保存失败时如何反馈。上游 gsplat 示例的 ckpt 分支是评估路径，并未保存通用训练优化器；不能用它直接替换本地续训代码。[G02](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/examples/simple_trainer.py)

## 数据与分组边界

共享训练包保留既有 frame、split 与初始化规则；普通 COLMAP 则利用 frame_ 文件名和周期划分，未知组使用图片 ID/序号构造。Dataset 还要求训练及验证两侧都有图片。后者是原型限制，不是已经决定的通用产品规则；需要 DEV-06 明确只有训练图时的允许行为和告知方式。

prepare 为每张输入输出编号 PNG、归一后的 mask 和调整后的相机参数，原文件不覆盖，但新增项目空间成本可能很高；另存项目也有复制行为。D-05 应决定外部引用与可选归档的边界，保留身份校验与完整清单，不继续靠隐式全复制解决路径问题。

## 本地依赖偏差

仅读取现有 .venv-gsplat 的 pyvenv.cfg 与包 METADATA：Python 3.10.8、torch 2.1.2+cu118、gsplat 1.4.0+pt21cu118；Pillow 元数据为 12.3.0，而 APP studio extra 声明 Pillow>=10,<12。这是**配置与元数据不一致**，不是本轮运行故障结论。未运行 import、pip 或安装修复；后续环境整理单独处理。完整版本与证据范围见[依赖登记](SOURCES-AND-DEPENDENCIES.md)。

## 审计结论

优先保留输入完整性、分组、遮罩损失、全景补偿、训练状态保存和 SH 变换规则；正式 UI 与预览/训练协调需要实质性改造。选择 LichtFeld 时将这些规则作为迁移验收条件；选择备选时也不能以已有原型为由跳过快照隔离、恢复和 A 布局实现。
