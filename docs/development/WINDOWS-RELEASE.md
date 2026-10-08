# GS Studio Windows 离线目录包

`GSStudio.exe` 是 PyInstaller onedir 桌面入口。`worker/GSStudioWorker.exe` 执行结构化操作，`gpu-runtime/GSStudioTrainer.exe` 包含独立的固定版本 CUDA 13 训练环境；`cli/GSStudioCLI.exe` 提供命令与诊断入口。`model-cache/hub/checkpoints` 保存 Mask R-CNN 和 LPIPS 使用的 AlexNet 预训练权重，由四个入口共用。运行端无需安装 Python。不要单独复制 EXE，四个目录、`_internal` 和 `model-cache` 必须一起保留。

## 构建

构建机需准备两个完整环境：主环境装 `.[studio,packaging]`、PyTorch 2.9.1+cu130、torchvision 0.24.1 与现有流水线依赖；独立 `.venv-gsplat` 装 `.[packaging]`、PyTorch 2.9.1+cu130、gsplat 1.5.3 及已编译的 `csrc.pyd`。Qt 编辑视口直接调用 torch/gsplat，因此主环境还需安装同版本 gsplat 并复制同 ABI 的扩展；`bootstrap-gsplat-windows.ps1` 会执行这一同步。两个环境都安装固定的 PyInstaller 6.16.0。脚本用 `nvidia-smi` 检测当前 GPU 架构，编译记录 `gsplat/csrc-build.json` 绑定架构、Torch 版本和扩展哈希；发布预检核对主环境、GPU 环境及目标显卡，不能把只含 sm_120 的扩展发给 sm_89 主机。`-FfmpegDir` 可传 Gyan FFmpeg 完整包根目录（含 `bin/` 和 `LICENSE`），也可传其中的 `bin` 目录；脚本从二进制目录或其父目录查找 `LICENSE`/`COPYING`，并打包 `ffmpeg.exe`、`ffprobe.exe` 和所需 DLL。不要传只有可执行文件链接、没有许可证的 WinGet `Links` 目录。然后运行：

```powershell
& scripts/build-windows-release.ps1 -FfmpegDir 'C:\path\to\ffmpeg-8.1.1-full_build' -TorchHome 'C:\path\to\torch-cache'
```

`-TorchHome` 指向包含 `hub/checkpoints` 的模型缓存根目录；省略时使用主环境的 Torch 缓存目录。构建前应准备官方 `maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth` 和 `alexnet-owt-7be5be79.pth`。脚本核对完整 SHA-256，缺失或改变时立即失败，不在构建中下载模型。

脚本先检查 GPU 架构、两个环境、模型权重和 FFmpeg，随后生成单独的时间戳目录；不会修改原片或覆盖旧包。若构建机有未提交源码，清单中的 `source_state` 会记录为 `modified`。`release-manifest.json` 记录版本、Git 提交、目标 GPU 架构和包内每个文件的 SHA-256。发布前运行 `verify-release.ps1`，再运行 `diagnostics.ps1 -DataRoot <目录>`；诊断会占用与 GUI、CLI、训练器共用的用户级 GPU 锁。

## 使用与外部依赖

首次启动选择现有 Data 目录；路径存入当前用户的 `%LOCALAPPDATA%\GSStudio\settings.json`，可用 `GSSTUDIO_DATA_ROOT` 临时覆盖。锁位于同一用户目录的 `gpu.lock`，可用 `GSSTUDIO_GPU_LOCK_PATH` 显式覆盖。训练暂停时进程仍持有 GPU 锁与模型显存，诊断和其他 GPU 任务需等待训练结束。发布目录可以整体搬移，Data 保持独立。冻结入口将 `TORCH_HOME` 指向随包的 `model-cache`，不会依赖构建机或运行用户的已有模型缓存；源码运行仍使用原有缓存设置。

RealityScan 2.2 仍需另行安装；Postshot 和 MediaSDK helper 是可选外部程序。NVIDIA 驱动须支持随包 CUDA 13 运行时。只有 Windows x64、固定 CUDA/PyTorch/gsplat 版本是本包目标，其他 GPU 尚未验收。包内 FFmpeg 的许可证随二进制一同保留。

构建成功只证明文件生成和哈希一致。还需在目标 Windows 主机验证启动、worker、GPU 前后向、真实素材、目录搬移、断网、中文路径及 DPI，才可标记为已验收。
