# GS Studio 维护说明

日常操作只使用 [当前工作手册](../user/CURRENT-WORKFLOW.md)。本页记录入口职责、环境版本、运行期已知边界、不支持的历史形态和测试方法，不提供另一套默认流水线。

## 入口与运行环境

| 入口 | 用途和输入 | GPU / 写入行为 |
| --- | --- | --- |
| `gsstudio.ps1` | 任意目录调用 GS Studio CLI，参数原样传递 | 取决于子命令 |
| `gs-studio` | 安装后的 PySide6 工作台命令 | 取决于所选操作 |
| `scripts/session.ps1` | 初始化主解释器、Data、helper 和 SDK，支持 `-DataRoot` | 不启动 GPU，不写 Data |
| `scripts/bootstrap-windows.ps1` | 首次安装主环境 | 写 .venv，下载依赖；不自动运行 doctor |
| `scripts/bootstrap-gsplat-windows.ps1` | 安装独立训练/评估环境并编译 CUDA 扩展，支持 `-SkipPackages` | 写 .venv-gsplat 与 wheels/，下载依赖、跑 nvcc |
| `scripts/build_gsplat_csrc.py` | 把已安装 gsplat 的 CUDA 源编译成 `gsplat/csrc.pyd` | 写环境内包与 wheels/ 缓存；幂等 |
| `scripts/apply_nerfstudio_patch.py` | 安装时核验并应用投影 clamp 和嵌套路径补丁 | 修改环境内依赖代码，不处理素材 |
| `tests/manual/test-mediasdk-helper.ps1` | 指定 LocationId、SceneId、CaptureId，独立 helper 按 5 fps 验收（不代表新 Run 默认）；可用 RunId 恢复 | GPU、解码和 Run 候选缓存；不是轻量单测 |
| `tests/manual/smoke-native-gsplat.py` | 独立环境，`--output` 指定新的合成测试目录 | GPU、少量合成训练与检查点；不访问 Capture |
| `tests/manual/check-native-export.py` | 独立环境，`--experiment` 与 `--dataset` 核验 PLY 回读 | GPU，写 export-diagnostic.json；不重训 |
| `tests/manual/check-studio.py` | 独立环境，`--output` 指定新目录；`--gpu` 走 CUDA 集成，缺省只做 GUI 检查 | GPU/GUI；只写 `--output`，不访问 Capture |
| `scripts/compare-backends-v5.py` | 主环境，`--run-dir --segment --output` 对照通过的分段 | 默认实际训练；`--dry-run` 仍会准备包及记录 |
| `tools/mediasdk-helper/build.ps1` | 构建连接本机 SDK 的 helper | 写 build，不运行采集 |
| `cleanup` | LocationId、SceneId、RunId；默认预览，`--apply` 执行 | 不使用 GPU；持有 Run 锁，仅删除验证后的白名单文件 |
| `doctor` | 检查当前 Windows 重建与所选后端 | 使用 GPU 锁，产生并清理诊断临时文件；不训练 |
| `status` | 只读项目/阶段/证据/日志状态，可逐级下钻；`--json` 供脚本 | 不使用 GPU，不写任何文件；状态只来自清单与证据 |
| `scripts/gs-studio.ps1` | 启动 PySide6 工作台，可选 `-DataRoot`；创建/探测、Run、遮罩审核、QA、训练和编辑均有代码入口 | 长任务在独立 worker 进程运行；探测/重建/训练可能占用 GPU；当前未经运行验收 |
| `select-sharp` | 显式 `--source` / `--out` 的独立选帧工作台 | GPU（遮罩/投影）；只写 `--out`，不进 Data 目录 |

独立 GPU 诊断脚本应在流水线空闲时执行；不要从独立脚本绕过正在使用的生产 GPU 锁。环境、SDK、helper 构建产物和 `env` 配置不是历史垃圾，不在文档清理范围内。

GUI 与 CLI 共用 `src/gsstudio/application/` 的状态读取、操作层及错误分类，边界见 [D-13](../architecture/GS-STUDIO-DECISIONS.md)。Qt 通过 `gsstudio.interfaces.worker.main` 的 JSON 行事件调用操作层；CLI 输出仍只用于人工和脚本阅读，不作为 GUI 状态源。写入操作继续调用同一 pipeline、Run 锁与 retention 逻辑。新 GUI Run 默认启用人工遮罩审核；CLI 的默认 Run 参数仍由自身选项明确传入。`capture relink` 仅在文件数、顺序、大小和哈希完全匹配时更新外部路径。

包名与控制台入口已改为 `gsstudio`。已有 `.venv` 和 `.venv-gsplat` 的 editable 安装元数据仍可能指向旧包；恢复运行验证并准备启动工作台时，应分别重装本项目到所用解释器，生成新的 `gsstudio` 和 `gs-studio` 入口。本轮受 D-02 约束未执行安装或启动。

测试脚本单独存放在 `tests/`：`tests/unit/*.py` 是 pytest 单元测试，`tests/contract/*.py` 是接口契约测试，`tests/manual/` 是需要真机或独立环境的检查脚本，均入库。测试中间产物一律不入库：运行输出写到 `tests/output/<名称>/` 或仓库外，`.gitignore` 忽略 `tests/output/`、`tests/.tmp/`、`__pycache__/`、`.pytest_cache/`、`.coverage` 与 `htmlcov/`。

后端对照直接输入 Run 目录。保持两后端、补偿开关、步数、时长及恢复选项；失败/阻断返回非零。复用成功结果时核验包身份、预算和 PLY 哈希。`--dry-run` 是准备动作，不是只读检查。

## 环境版本（Blackwell / CUDA 13）

sm_120（RTX 5090 D v2，compute capability 12.0）主机无法执行 CUDA 11.8 构建：CUDA 11.8 工具链不支持 sm_120，PyTorch 2.1.2 的 cu118 轮子在 Blackwell 上跑不了 CUDA 内核。仓库现固定下列组合，PyTorch 轮子源为官方索引 `https://download.pytorch.org/whl/cu130`：

| 项 | 版本 |
| --- | --- |
| PyTorch / torchvision | 2.9.1+cu130 / 0.24.1+cu130 |
| CUDA 工具链 | 13.x（nvcc 13.4 实测；可用 `GSSTUDIO_CUDA_HOME` 指定） |
| gsplat | 1.5.3，源码编译，`TORCH_CUDA_ARCH_LIST=12.0` |
| nerfstudio | 固定提交 `758ea19…` + `--no-deps` + 最小依赖子集 |

nerfstudio 的 `--no-deps` 只跳过它自己固定的 CUDA 扩展（gsplat 1.4.0、nerfacc 0.5.2）。GS Studio 只导入其 COLMAP/equirect 辅助模块，未使用的重型依赖（jupyterlab、nuscenes-devkit 等）不安装。主环境与 `.venv-gsplat` 都装这套 torch，两处版本断言在 `src/gsstudio/application/doctor.py` 的 `_gsplat_runtime_check`。

gsplat 没有 cu130 轮子，因此 `bootstrap-gsplat-windows.ps1` 安装纯 Python 包后调用 `scripts/build_gsplat_csrc.py`，把随包分发的 CUDA 源编译成 `gsplat/csrc.pyd`，产物缓存在 `wheels/`（按 Python/torch 版本命名，换版本不会误复用）。CUDA 查找顺序：`GSSTUDIO_CUDA_HOME`、`CUDA_HOME`、`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.*`、`GSStudio\..\tools\cuda-13.4`；MSVC 环境由 vswhere 定位后经 vcvars64 注入。

### 上游 bug：MSVC 顶层 const 修饰名

gsplat 1.5.3 的 `RasterizeToPixels2DGSBwd.cu` 与 `RasterizeToPixelsFromWorld3DGSFwd.cu` 在 `__INS__` 显式实例化宏里把非 const 输出参数（`v_means2d`、`renders` 等）写成了 `const at::Tensor`，与 `Rasterization.h` 的声明不一致。MSVC 会把按值参数的顶层 const 编进修饰名（gcc/clang 不会），因此 Windows 链接报 LNK2019（仅这两个 kernel、共 38 个实例）。`build_gsplat_csrc.py` 编译前幂等地改写这两处宏，使实例化与头文件逐参数一致。升级 gsplat 后若 LNK2019 复现且 demangle 签名相同，优先核对这里。

### 回退阶梯

1. nvcc 拒绝当前 MSVC 时，设置 `NVCC_PREPEND_FLAGS=-allow-unsupported-compiler`。
2. 仍失败则安装 VS 2022 Build Tools（v143 工具集）供 nvcc 使用。
3. 仍失败则退回 torch 2.7.1+cu128 与 CUDA Toolkit 12.8，并同步修改 `doctor.py` 与 `native_train.py` 中的版本断言。

## 运行期已知边界

- **FFmpeg `select=` 表达式上限 100 项。** `extract_indexed_frames` 用 `eq(n,i)+…` 抽指定源帧，FFmpeg 把这条链解析进定长表达式栈，第 101 项起解析失败，报错却是打开输出文件时的 `Cannot allocate memory`（退出码 -12），看起来像磁盘或内存不足。实测边界恰为 100 项（ffmpeg 9.0.1）。现按 100 项分块，每块用 `-start_number` 接到全局序号、`-frames:v` 让每块停在自己的最后一个选中帧；分块结果已与逐帧单独抽取逐字节比对一致。INSV 走 MediaSDK helper 的 JSON 帧表，不受此限制。
- **`torch.quantile` 元素数上限 2²⁴。** 4K 透视渲染为 3840×2160×3 = 24.9M，超出上限。`native_train.exact_quantile()` 走 NumPy（插值方式与 torch 一致、无上限），`evaluate()` 与 `check-native-export.py` 共用。全景路径训练的是小投影，此前碰不到该上限。
- **PLY 往返自检比较的是"文件所编码的状态"。** 导出在 NumPy 中归一化四元数，gsplat CUDA 内核再归一化内存参数，两条 float32 路径舍入差约 1e-7，且该差异对渲染的影响随 splat 数与分辨率增长（342 万 splats 等距柱状约 0.6 个 8-bit 级，4K 针孔 173 万 splats 实测最大 0.035，但 24.9M 通道里只有 12 个超过 1/255）。自检改为用 `export_ply` 返回的归一化四元数重建参数再与 `load_ply` 回读比对：两条路位级一致，门限为每像素 1 个显示级 + RMS 1e-5，任何丢失、转置或重排仍会被位级比对拦下。四元数舍入对渲染的影响改为记录（`quaternion_rounding_render_*`），重复渲染噪声一并记录（`export_roundtrip_noise_*`）。
- **8K 人物遮罩的显存回收。** torchvision 按源分辨率粘贴每个检测的 mask，一张 8K 等距柱状图会分配若干约 118 MB 块，数量随检测到的人数变化；缓存分配器保留这些块又无法整理，二十帧内 `memory_reserved` 就超过 24 GiB 物理显存，WDDM 开始换页，单帧从约 1.5 s 掉到 100–250 s。`TorchvisionPersonSegmenter` 每帧推理后 `empty_cache()`（实测 20 帧 498 s / 41 GiB → 37 s / 0.19 GiB）。

## 不支持的历史形态

本轮只保留一条流水线与一套 Run 配置，**旧 Run 不再可读、不可恢复**：

- schema 1–5 的 Run 配置（含 schema 1 Capture 的手工拼接 MP4 路线）全部移除；`manifest.yaml` 的 `config` 只接受现行形状，读到旧文档直接报错。
- Capture 共享的 `prepared/<capture>/<hash>` 候选缓存及其指针字段移除；候选由 Run 独占，存 `inputs/primary/<preparation hash>`。
- 旧入口移除：`postshot-prepare`、`postshot-train`、`export`（Nerfstudio 发布链）、`clean`、`ingest --stitched`、`qa report --baseline-run-id`。Postshot 图片复核用 `postshot-review --dataset`，需显式路径。
- 旧 flag 移除：`--target-frames`、`--primary-frames`、`--fallback-frames`、`--fallback-per-second`、`--fallback-fov`、`--fallback-projection-size`、`--max-masked-fraction`、`--train-iterations`。`--attempt` 只有 `primary|repair`。
- 均匀抽帧与 laplacian 选帧算法、COLMAP mapper + 刚性 rig 组装链、Nerfstudio 发布/Ply 轴转换、WSL 路径换算一并移除。投影、COLMAP 模型读取与逐图遮罩挂接仍在使用。
- 训练实验身份（`dispatch.json`、原生 `config.json`）改为严格比对，不为旧记录补默认值。

## 验证

```powershell
$AppRoot = 'D:\Project\3DGS\APP'
. (Join-Path $AppRoot 'scripts\session.ps1')
Push-Location -LiteralPath $AppRoot
try {
    & $GsstudioPython -m pytest -ra
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    git diff --check
} finally {
    Pop-Location
}
```

单测使用隔离锁、临时目录和模拟外部组件，覆盖 FFmpeg 分块序号映射、全景投影重叠规则、perspective 源校验、遮罩白名单与训练身份比对。GUI 壳检查（`tests/test_gui_shell.py`）用 `QT_QPA_PLATFORM=offscreen` 离屏构造窗口，未装 PySide6 时自动跳过。文档维护不启动真实 Capture 重建、模型训练或 SDK 硬件验收。新增文档命令需核验 CLI 帮助、PowerShell 语法及模拟调用。

真实素材验收另行执行：helper 对真实 INSV 的 probe/export_frames、`tests\manual\test-mediasdk-helper.ps1` 硬件验收、`gsstudio reconstruct` 的 RealityScan 全流程，以及 gsplat 1.4.0 → 1.5.3 后的训练数值与画质，都需要在装好 SDK/RealityScan 的机器上用真实素材重新确认，不能以短训练或退出码代替。
