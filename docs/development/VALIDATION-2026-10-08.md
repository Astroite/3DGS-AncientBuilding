# GS Studio 分层验证记录（2026-10-08）

本轮在提交 `76cde52f870c448522b1ffc76c2bdfe18345be5d` 上恢复固定 GPU 环境并继续验证。历史结果保留在 [9 月验证记录](VALIDATION-2026-09-24.md)，现行阶段与验收见[项目路线图](../architecture/GS-STUDIO-PROJECT-ROADMAP.md)。合成数据验证、真实素材阶段和人工画质接受分别记结论。

## 环境与身份

| 项目 | 当前实际值 |
| --- | --- |
| Windows / Python | Windows 11 x64 / 两个环境均为 Python 3.12.13 |
| 主环境与独立训练环境 | `.venv` / `.venv-gsplat`，均重新安装当前项目入口 |
| GPU / 驱动 | RTX 4060 Ti 16 GB，compute capability 8.9 / 596.36 |
| Torch / torchvision / gsplat | 2.9.1+cu130 / 0.24.1+cu130 / 1.5.3 |
| CUDA 编译工具 | NVIDIA 官方 redistrib 13.0.2，nvcc 13.0.88；本地 `../tools/cuda-13.0.2` |
| MSVC | 已安装的 v143 14.44 工具集，经 `-MsvcVersion 14.44` 显式选择 |
| PyInstaller / FFmpeg | 6.16.0 / 8.1.1 |
| RealityScan | `D:\UnrealEngine\RealityScan_2.2\RealityScan.exe`，诊断检测通过；尚未执行本 Run 重建 |

主环境与训练环境的 `gsplat/csrc-build.json` 均绑定 Python 3.12、Torch 2.9.1+cu130 和 sm_89；扩展 SHA-256 均为 `4a31cc5d182216c8408cd2612523dd0afc1c2d328c9f20e3b6b1d5562fcc3d3c`。最小 CUDA 编译、完整 gsplat 扩展编译以及两环境 CUDA 前后向检查均通过。上游 nerfstudio 固定于 `758ea1918e082aa44776009d8e755c2f3a88d2ee`，按本项目实际辅助模块安装最小依赖并应用已有补丁。

`doctor --backend gsplat` 的必要项全部通过。可选 MediaSDK helper 未配置；已安装 Postshot 1.0.110 低于本项目要求的 1.1.69，尚无当前 Postshot 验收结论。未升级驱动、商业软件或 SDK。

## 合成训练与回归

- 最终 CPU/Qt 单元及契约检查 **105 项通过**。新增覆盖冻结 worker 中文协议、发布缓存完整性、窄窗口阶段显示、Windows 控制状态短暂访问冲突、Unicode 图像读写及 LPIPS 包内资源定位。
- `smoke-native-gsplat.py` 通过 CUDA 前后向、检查点恢复、PLY 往返和中断恢复一致性；恢复与连续训练的最大渲染差为 `1.4901161193847656e-07`。
- `check-studio.py --gpu` 通过第 2 步暂停、第 8 步停止、恢复至第 12 步、两次预览、编辑导出及项目重开；导出回读最大误差为 0。此入口验证共用 Runtime，**不是正式 Qt 全流程交互验收**。
- 完整回归曾捕获 Windows 替换控制状态文件时一次短暂 `PermissionError`。控制状态读取现最多重试 5 次、间隔 20 ms；持续权限错误仍抛出。GUI 与控制提交共用读取逻辑，失败测试线程不再阻止测试进程退出。

上述数据均标记 `synthetic_only=true`；不证明代表素材画质、百万级模型性能或独立查看器互操作。

## 真实 DJI 遮罩阶段

沿用隔离 Location `validation-dji-20260924`、Scene `flight-0004`、Run `20260928T072324Z-4dad3aa4`，配置 SHA-256 为 `3e07a16a8d64ea4e7c5e98739ee2d0c96374c9c91a81d22e7786e95a6fe75023`。Run 仅取 5 秒、5 张透视帧，原配置的 `mask_review_required=true`、`vision_qa=false` 和 QA 最少 10 样本保持不变。

真实 `mask` 在共用 GPU 锁内完成，耗时约 8 秒；5 个视图通过确定性检查，0 个丢弃。状态审计显示 `preprocess=succeeded`、`mask=succeeded`、`reconstruct/qa=pending`、Run `waiting_review`、`readable=true`、`problems=[]`。第 5 张右侧存在一个小遮罩，已提供联系表和本地审核页等待用户核对；**尚未记录人工通过，也未固定遮罩或开始 RealityScan**。

源 `DJI_20260913170900_0004_D.MP4` 在遮罩前后重新核对：449,364,334 字节，SHA-256 均为 `53ba002eb43720b0bd446f1263ae277818c9a8af8c67c1b56c383da707004d29`，最后修改时间未变化。操作只写该 Run 的派生目录。5 帧 smoke 不足以宣称正式分段 QA、训练或画质通过。

## 正式 Qt 与离线包

使用 Windows Qt 后端只读加载上述真实 Run；字体数据库有 343 个字体族，包括 Microsoft YaHei。逻辑 1600×900 和 1280×800 的截图分别为 2400×1350 和 1920×1200，实际设备比例 1.5。截图发现并修复了隐藏模型面板挤压项目树、训练预览标签页挤压阶段表和配置哈希溢出的问题；修复后四个阶段保持可读。截图只证明这些页面的渲染与状态呈现，完整尺寸/DPI矩阵、Qt 编辑交互及用户 UI 接受仍待验收。

离线构建现要求完整校验并携带 Mask R-CNN 与 LPIPS 使用的 AlexNet 权重。四个冻结入口统一从随包 `model-cache` 加载；构建不自动下载模型。权重身份为：

| 文件 | SHA-256 |
| --- | --- |
| `maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` | `73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e` |
| `alexnet-owt-7be5be79.pth` | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` |

首个实际目录包已生成并通过 58,433 个文件的哈希校验，冻结 CLI 帮助和训练器 CUDA 前后向检查通过。进一步检查发现冻结 worker 未按 UTF-8 读取 JSON、LPIPS 根据相对代码路径查找自身权重，以及 OpenCV 在中文路径下返回成功却未生成图像。这些失败均保留在本地日志；不能把初始包当作可用发布结果。

worker 现显式使用 UTF-8；LPIPS 根据模块的包内路径加载自有权重。流水线图像读写统一采用 OpenCV 编解码加 Python 文件 I/O，并沿用原子替换；写入失败不会发布成功状态。修复后源码在隔离中文 Data 路径通过真实 5 秒 DJI 准备、遮罩及门禁审计，原片身份不变；在拒绝下载代理下通过 384×288 合成输入的 12 步训练和 LPIPS 评估，全部 12 张结果图像实际存在。

最终目录包须重新构建并检查。正式离线、中文路径搬移、目标主机及断网验收不能只用文件生成和清单哈希代替。

## 证据与剩余关口

本地日志、环境清单、JUnit、训练结果、截图及工具链身份存于忽略目录 `tests/output/validation-20261008/`；真实遮罩清单、日志与联系表保存在 Data 的上述 Run 下。环境、模型权重、原片、检查点和 EXE 不进入 Git。

下一步先完成当前遮罩的人工审核，再固定当前身份并运行 RealityScan 与 QA；若 5 帧输入失败或样本不足，原样保留结论，不降低 QA 门槛。代表素材训练与画质、正式 Qt 操作、故障场景及完整离线包验收仍需各自证据。INSV、全景和室内素材没有本轮证据。
