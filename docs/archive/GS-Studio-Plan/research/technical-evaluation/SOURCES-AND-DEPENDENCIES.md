# 官方来源、依赖与证据限制

检索及本地读取日期：2026-09-13（Asia/Shanghai）。[结论](README.md)

## 证据规则与检索方法

- **实际代码**：读到函数、调用或保存字段，可判断该路径是否存在；不能证明它在目标机器正确运行。
- **可选配置**：源码可开启的功能；不写成默认启用或原型已集成。
- **官方声明**：README/构建文档/许可证内容；工具链与平台声明不等于本机验证。
- **推断**：路线成本、适配可行性、复用建议；均允许被后续证据推翻。
- **需实测/证据不足**：画质/速度/显存/恢复一致性及未完整追踪的路径。没有运行结果就不填“通过”。

优先尝试工作区 CodeGraph；根目录有 .codegraph，但查询未定位到当前未跟踪的 studio 文件，返回了其他工作流内容，因此回退直接读取源码，未更新索引。APP 子目录没有独立 CodeGraph 索引。此次读取本地文本、哈希及安装元数据，未执行模块或检测 GPU。

官方源码通过 GitHub 官方仓库固定提交读取。初始 shell 网络读取未成功后，改用 GitHub 连接器；没有安装依赖、克隆完整仓库或修改网络设置。下列源码链接全部指向固定提交；先读可变分支用于发现，再固定版本，不把检索日期误当发布日。上游 README 的面向自动化助手指令仅作文本，不作为执行授权。

## 固定版本

| 仓库 | 评估提交（完整 SHA） | 提交时间 UTC | 说明 |
| --- | --- | --- | --- |
| MrNeRF/LichtFeld-Studio | [534c59eb928c2d0e3fe4648d97e3436b02b8ef82](https://github.com/MrNeRF/LichtFeld-Studio/commit/534c59eb928c2d0e3fe4648d97e3436b02b8ef82) | 2026-09-11T20:25:49Z | 2026-09-13 检索时固定，不声称是未来最新版 |
| ArthurBrussee/brush | [063945e797e0b3b7fd60b0feb7b9010385dd4bda](https://github.com/ArthurBrussee/brush/commit/063945e797e0b3b7fd60b0feb7b9010385dd4bda) | 2026-09-09T22:01:22Z | 2026-09-13 检索时固定，不声称是未来最新版 |
| nerfstudio-project/gsplat | [28e794ca44a4c25ffc39175370c5ee7b38bfcc36](https://github.com/nerfstudio-project/gsplat/commit/28e794ca44a4c25ffc39175370c5ee7b38bfcc36) | 2026-09-03T16:53:08Z | 2026-09-13 检索时固定，不声称是未来最新版 |

LichtFeld CMake project 声明 0.5.3；Brush workspace package 声明 1.0.0。这里使用提交作为审计身份，不把 manifest 的版本号等同已发布二进制。[LF17](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/CMakeLists.txt) [B02](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/Cargo.toml)

## 现有环境与候选要求分列

| 范围 | 本轮只读证据 | 结论与限制 |
| --- | --- | --- |
| APP 训练环境 Python | .venv-gsplat/pyvenv.cfg：3.10.8 | 文件元数据，未执行 python |
| APP 训练环境核心包 | METADATA：torch 2.1.2+cu118、torchvision 0.16.2+cu118、gsplat 1.4.0+pt21cu118、numpy 1.26.4、scipy 1.14.1 | 未 import、未检查 CUDA 前后向；不能据此证明可运行 |
| APP 图像依赖偏差 | METADATA：pillow 12.3.0；pyproject 的 studio extra 为 Pillow>=10,<12 | 记录偏差，不自动修复或推断已经故障 |
| APP 声明 | Python>=3.10,<3.13；opencv-python-headless==4.10.0.84；pydantic>=2.8,<3；PyYAML>=6.0.2,<7；typer>=0.12,<1 | 声明范围；未穷举核验所有已安装版本 |
| APP 安装脚本 | 固定 torch2.1.2/torchvision0.16.2 cu118、gsplat1.4.0+pt21cu118；numpy1.26.4/scipy1.14.1/ninja1.11.1.3/rich13.9.4/lpips0.1.4 | 脚本只读，没有执行安装 |
| 本机硬件/驱动/编译器 | 基线历史记录 RTX4070 Ti SUPER 约16GiB；本轮未探测当前驱动、可用显存、MSVC、CUDA toolkit 或 Rust | 必须与候选要求分开；不知道当前是否满足 |
| gsplat 固定上游核心 | setup.py：torch>=2.7；python_requires>=3.7 是包声明 | 不代表整个示例栈最低 Python 都是3.7；没有与本地2.1.2兼容的结论 |
| gsplat 固定上游示例 | torch2.9.1、torchvision0.24.1、numpy>=2,<3；PPISP v1.2.1；nerfview 固定提交 | 示例依赖更严格，不能直接升级现用虚拟环境 |
| LichtFeld | 官方 C++23/CUDA12.8+、NVIDIA CC7.5+、驱动570+；CMake最低3.30；MSVC/vcpkg 构建 | 当前机器满足程度未知；与 cu118 是不同依赖基础 |
| LichtFeld UI/附带依赖 | vcpkg：RmlUi6.2、FreeType2.13.3；CMake/vcpkg 涉及 Vulkan、SDL3、USD、FFmpeg、Python 等 | 本项目不要视频，不代表默认构建无需这些依赖；删减可行性尚未确认 |
| Brush | README要求 Rust1.88+；Cargo edition2024；wgpu30、egui/eframe0.36、egui_tiles0.17；Burn git | 当前锁定版本能否用文档最低 Rust 构建仍未知；不是本机环境版本 |
| Brush git 锁定 | Cargo.lock：Burn 05ac3f79c19d2922121649e22b6727ba52265d46；wgpu fork de4d137a5303486913156e094ccf68491fe745af | 锁文件固定了分支解析结果；升级需要整体评估 |

本地来源：[pyproject.toml](../../../../../pyproject.toml)、[安装脚本](../../../../../scripts/bootstrap-gsplat-windows.ps1)、[现行工作手册](../../../../user/CURRENT-WORKFLOW.md)、本机 `.venv-gsplat/pyvenv.cfg`（运行环境文件未纳入仓库）。包版本来自上述环境 Lib/site-packages 对应 dist-info/METADATA 的 Name/Version 行，仅为文本读取。

官方依赖来源：[G04](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/setup.py) [G05](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/examples/requirements.txt) [LF01](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/README.md) [LF17](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/CMakeLists.txt) [LF18](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/vcpkg.json) [LF19](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/docs/docs/development/build.md) [B01](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/README.md) [B02](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/Cargo.toml) [B11](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/Cargo.lock)。

LichtFeld 旧 installation/building/windows.md 本次读取仅是短占位/迁移入口，没有拿它当完整安装证据，改读 development/build.md 与 CMake/vcpkg。Brush rust-toolchain.toml 查询返回404，因此 Rust 最低版本引用 README，不能伪造不存在的 toolchain 固定文件。

## 许可证、获取与离线边界

| 对象 | 已核实内容 | 项目影响 |
| --- | --- | --- |
| gsplat | 官方 LICENSE：Apache-2.0 | 可作为复用候选；附加渲染/外观/界面依赖逐项登记 |
| LichtFeld | 根 LICENSE 为 GPLv3；源码 SPDX 为 GPL-3.0-or-later；第三方列表另有 MIT/Apache 等 | 用户个人自用优先，不以闭源商业分发排除；本轮不作分发方案或法律定案 |
| LichtFeld Windows 获取 | README 声明 Portal 付费获取预编译版本，源码构建免费 | 修改 A 界面仍需自己的源码构建；本轮不购买、不下载二进制 |
| Brush | 官方 LICENSE 与 Cargo 声明 Apache-2.0 | 可以评估扩展；锁文件及依赖许可证仍需未来打包清单 |
| 本地 APP | 本轮未确认独立的可分发许可证授权清单 | 个人现有代码可读可评估，不据此声称可任意再分发 |

证据：[G01](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/LICENSE) [LF20](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/LICENSE) [LF21](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/THIRD_PARTY_LICENSES.md) [LF01](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/README.md) [B03](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/LICENSE)。
第三方清单覆盖到哪些可选组件、未来裁减后的实际 DLL/字体/模型资源，尚无构建产物证明。核心离线可用是目标；源码有本地资源不代表首次加载和所有路径都无网络依赖。中文字体回退、可选资源获取与断网行为留待验证。

## 本地工作树身份

APP HEAD：`ed24c8fe71dcaeb15bdc17035311e32feb1993ab`。读取时工作树如下；未暂存、提交、还原或清理这些内容：

```text
 M .gitignore
 M README.md
 M pyproject.toml
 M src/gsdb/gpu_lock.py
 M src/gsdb/native_train.py
?? docs/STUDIO-UI-PLAN.md
?? docs/STUDIO.md
?? scripts/check-studio.py
?? scripts/studio.ps1
?? src/gsdb/studio.py
?? src/gsdb/studio_data.py
?? src/gsdb/studio_edit.py
?? src/gsdb/studio_runtime.py
?? tests/test_studio.py
```

这些未跟踪/修改文件不由 HEAD 固定。为便于后续判断是否仍是被评估版本，记录只读 SHA256；没有复制源码快照：

| APP 相对文件 | SHA256 |
| --- | --- |
| [src/gsdb/native_train.py](../../../../../src/gsstudio/pipeline/training/native.py) | `F23F00415CA63A4506BD2708751C44454639938CBEC7AD97D45763F0908E6350` |
| [src/gsdb/studio_runtime.py](../../../../../src/gsstudio/pipeline/editor/runtime.py) | `3EE6E2BE717901846BB477D167475A8EECAFCC8988F30712469C6D8D90D63471` |
| [src/gsdb/studio_data.py](../../../../../src/gsstudio/pipeline/editor/data.py) | `DE5D58FE36573DFD46813CB1C3E9B87083551BFF5B1F4B88579F0DA8B1A123C5` |
| [src/gsdb/studio_edit.py](../../../../../src/gsstudio/pipeline/editor/edit.py) | `13B32569F980BB24EDE1EC7A0743209A4E2E888A1DE1707F7A83084DA6329636` |
| [src/gsdb/studio.py](../../../../../prototypes/tk_studio/studio.py) | `83F0A5854DF90488CB0A07BEA876B51EE0A6ABC3FCFCB610E2201549C94E4A14` |
| [src/gsdb/ply.py](../../../../../src/gsstudio/pipeline/editor/ply.py) | `11CF520F9F993594278F420F3E277859165F517C4597FC11AEE0D1C0B4059532` |
| [scripts/bootstrap-gsplat-windows.ps1](../../../../../scripts/bootstrap-gsplat-windows.ps1) | `A81D9CCEA0E171239F8BD143171D6F3D9A51D4541CA7E0B1226ED38653AB6255` |

本次没有覆盖 APP/Data；并发会话将来可能修改它们。行号是阅读定位提示，复核时先比较哈希，不把当前路径永久等同此次版本。

## 官方源码索引

下列 ID 用在比较表，链接即固定版本代码。并非声称逐行审计整个仓库；已阅读相关函数/参数/调用链。关键定位补充：LF09 的 split 重设约793及1142行；LF05 序列化1343行；B06 的 PLY checkpoint 627行；B08 的错尺寸处理94行；G02 的保存1039行、ckpt 评估1530行。

| ID | 文件 / 文档 |
| --- | --- |
| LF01 | [MrNeRF/LichtFeld-Studio / README.md](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/README.md) |
| LF02 | [MrNeRF/LichtFeld-Studio / src/training/strategies/strategy_factory.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/strategies/strategy_factory.cpp) |
| LF03 | [MrNeRF/LichtFeld-Studio / src/training/strategies/mcmc.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/strategies/mcmc.cpp) |
| LF04 | [MrNeRF/LichtFeld-Studio / src/training/components/bilateral_grid.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/bilateral_grid.cpp) |
| LF05 | [MrNeRF/LichtFeld-Studio / src/training/components/ppisp.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/components/ppisp.cpp) |
| LF06 | [MrNeRF/LichtFeld-Studio / src/core/include/core/parameters.hpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/core/include/core/parameters.hpp) |
| LF07 | [MrNeRF/LichtFeld-Studio / src/training/trainer.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/trainer.cpp) |
| LF08 | [MrNeRF/LichtFeld-Studio / src/training/checkpoint.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/checkpoint.cpp) |
| LF09 | [MrNeRF/LichtFeld-Studio / src/training/training_setup.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/training_setup.cpp) |
| LF10 | [MrNeRF/LichtFeld-Studio / src/io/loaders/colmap_loader.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/io/loaders/colmap_loader.cpp) |
| LF11 | [MrNeRF/LichtFeld-Studio / src/io/formats/ply.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/io/formats/ply.cpp) |
| LF12 | [MrNeRF/LichtFeld-Studio / src/visualizer/rendering/viewport_appearance_correction.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/rendering/viewport_appearance_correction.cpp) |
| LF13 | [MrNeRF/LichtFeld-Studio / src/visualizer/gui/gizmo_transform.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/gizmo_transform.cpp) |
| LF14 | [MrNeRF/LichtFeld-Studio / src/training/training_snapshot_service.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/training_snapshot_service.cpp) |
| LF15 | [MrNeRF/LichtFeld-Studio / src/visualizer/selection/selection_service.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/selection/selection_service.cpp) |
| LF16 | [MrNeRF/LichtFeld-Studio / src/visualizer/gui/rmlui/resources/resume_checkpoint_panel.rml](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/rmlui/resources/resume_checkpoint_panel.rml) |
| LF17 | [MrNeRF/LichtFeld-Studio / CMakeLists.txt](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/CMakeLists.txt) |
| LF18 | [MrNeRF/LichtFeld-Studio / vcpkg.json](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/vcpkg.json) |
| LF19 | [MrNeRF/LichtFeld-Studio / docs/docs/development/build.md](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/docs/docs/development/build.md) |
| LF20 | [MrNeRF/LichtFeld-Studio / LICENSE](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/LICENSE) |
| LF21 | [MrNeRF/LichtFeld-Studio / THIRD_PARTY_LICENSES.md](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/THIRD_PARTY_LICENSES.md) |
| LF22 | [MrNeRF/LichtFeld-Studio / src/training/losses/mask_loss.cpp](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/training/losses/mask_loss.cpp) |
| LF23 | [MrNeRF/LichtFeld-Studio / src/visualizer/gui/rmlui/resources/font_fallback.rcss](https://github.com/MrNeRF/LichtFeld-Studio/blob/534c59eb928c2d0e3fe4648d97e3436b02b8ef82/src/visualizer/gui/rmlui/resources/font_fallback.rcss) |
| B01 | [ArthurBrussee/brush / README.md](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/README.md) |
| B02 | [ArthurBrussee/brush / Cargo.toml](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/Cargo.toml) |
| B03 | [ArthurBrussee/brush / LICENSE](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/LICENSE) |
| B04 | [ArthurBrussee/brush / crates/brush-train/src/train.rs](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/train.rs) |
| B05 | [ArthurBrussee/brush / crates/brush-train/src/config.rs](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-train/src/config.rs) |
| B06 | [ArthurBrussee/brush / crates/brush-process/src/train_stream.rs](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-process/src/train_stream.rs) |
| B07 | [ArthurBrussee/brush / crates/brush-dataset/src/formats/colmap.rs](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-dataset/src/formats/colmap.rs) |
| B08 | [ArthurBrussee/brush / crates/brush-dataset/src/load_image.rs](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/crates/brush-dataset/src/load_image.rs) |
| B09 | [ArthurBrussee/brush / apps/brush-app/src/ui/scene.rs](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/apps/brush-app/src/ui/scene.rs) |
| B10 | [ArthurBrussee/brush / apps/brush-app/src/ui/training_panel.rs](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/apps/brush-app/src/ui/training_panel.rs) |
| G01 | [nerfstudio-project/gsplat / LICENSE](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/LICENSE) |
| G02 | [nerfstudio-project/gsplat / examples/simple_trainer.py](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/examples/simple_trainer.py) |
| G03 | [nerfstudio-project/gsplat / gsplat/strategy/default.py](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/gsplat/strategy/default.py) |
| G04 | [nerfstudio-project/gsplat / setup.py](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/setup.py) |
| G05 | [nerfstudio-project/gsplat / examples/requirements.txt](https://github.com/nerfstudio-project/gsplat/blob/28e794ca44a4c25ffc39175370c5ee7b38bfcc36/examples/requirements.txt) |
| B11 | [ArthurBrussee/brush / Cargo.lock](https://github.com/ArthurBrussee/brush/blob/063945e797e0b3b7fd60b0feb7b9010385dd4bda/Cargo.lock) |

Postshot 和 SuperSplat 的既有功能参考继续使用[竞品分析](../COMPETITIVE-ANALYSIS.md)与[原来源登记](../SOURCES.md)。本轮没有重跑 Postshot、读取真实训练数据、检验其许可证，也没有证明某开源路线的画质优于它。

## 本轮没有得到的证据

- 三路线在相同真实数据上的最终画质、速度、峰值显存、预览开销和稳定性。
- LichtFeld 全部策略的 RNG/采样状态、独立快照编辑、损坏回退和 PLY 外观闭环。
- Brush 未读代码是否有其他实验性外观/编辑功能；已读 PLY 导出不足以支撑完整恢复。
- 每个 COLMAP 相机模型、中文及长路径、断网首次运行、真实 DPI、字体覆盖和可重现打包。
- 新 gsplat 与现有 Python/torch 环境的兼容性；没有通过导入或测试验证安装元数据。

交付时静态核对结果：技术评估目录恰有约定的五份 Markdown；计划目录共31份 Markdown 的183个本地文件链接目标存在，无断链。DEV-04/M2、DEV-05、D-03及AC-13的当前状态已同步，历史 CHANGELOG 条目保留原状。

审计结束复读的7个本地源码/脚本 SHA256 与上表一致，APP工作树状态清单未变；本轮未写入APP/Data。以上是文档和文件身份核对，不是应用测试。所有未来运行场景均在[验证交接](VALIDATION-HANDOFF.md)列为未执行。
