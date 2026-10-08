# GS Studio 分层验证记录（2026-09-24）

本记录对应工作树 `b9cdbc6` 加本轮未提交改动。它记录实施证据及缺口，不将代码存在或短测试等同于产品验收。正式 AC 定义见[验收计划](../archive/GS-Studio-Plan/quality/ACCEPTANCE.md)及[项目路线图](../architecture/GS-STUDIO-PROJECT-ROADMAP.md)。

## 输入与环境身份

- 隔离测试项目：`E:\Projects\3DGS\Data\validation-dji-20260924`；完整哈希与探测日志摘要见该目录的 `validation-report.md`。两段源视频仍在 `Data/Test`，仅作外部只读引用。9 月 28 日在短段 Scene 新建了一个仅完成预处理的 smoke Run，见下节。
- 短段 `DJI_20260913170900_0004_D.MP4`：449,364,334 字节；SHA256 `53ba002eb43720b0bd446f1263ae277818c9a8af8c67c1b56c383da707004d29`；39.422717 秒。
- 长段 `DJI_20260913170640_0003_D.MP4`：1,076,873,898 字节；SHA256 `b31e06a68b8b7242d5d351ac2138250e9eb89ba9ca57bcc337a26d12862ff1cb`；95.845750 秒。
- 两者均为 3840×2160、59.94 fps、HEVC 10-bit 的透视视频；主视频流已探测。已通过 `capture init`、`media probe`、`ingest` 和 `status --json`，Capture 状态仍为 draft。
- 主环境 Python 3.12.13。目标 GPU 为 RTX 4060 Ti（compute 8.9）；首次分层验证时 GPU 已被本项目以外的活动占用。RealityScan 2.2 位于 `D:\UnrealEngine\RealityScan_2.2\RealityScan.exe`，尚未启动。本机没有 `.venv-gsplat`、PyInstaller 或可运行的发布 EXE。最新资源复查见下节。
- 当前工作树运行 `PYTHONPATH=src`、`QT_QPA_PLATFORM=offscreen` 下的 `pytest tests/unit tests/contract -q -p no:cacheprovider`，**94 项通过**；这包括外部导入、Capture 重定位、CPU/MediaSDK 锁范围、训练等待取消与无检查点重试、日志证据保留、评估与 PLY 发布门禁、预览图像身份、编辑撤销、合成 PLY 回读、worker 和冻结路径契约。离屏测试不代替 UI 目视或 CUDA 运行。
- `scripts/build-windows-release.ps1 -FfmpegDir <目录>` 已运行预检，明确退出于缺少 `.venv-gsplat\Scripts\python.exe`，未生成发布目录。主 `.venv` 当前为 CPU 版 PyTorch，亦缺 PyInstaller/gsplat 等发布依赖；预设 CUDA 13 工具链路径未找到 `nvcc.exe`。脚本与哈希校验器的模拟目录测试通过，不能代替真实冻结。
- Qt offscreen 模式下主窗口可创建并完成一次布局绘制；该模式的字体数据库返回 0 个字体，中文均显示为方框，故截图不能作为中文字体、DPI 或视觉设计验收证据。

## 2026-09-28 Data 与资源复查

- 新建 Run 前重新读取 `Data/Test` 两段 DJI 源视频：字节数、最后修改时间与上列 SHA256 均未变化。两个隔离 Scene 当时的 `status --json` 显示 Capture `readable=true`、`problems=[]`、`runs=[]`。
- FFmpeg 与 FFprobe 的 8.1.1 可执行文件均通过只读版本查询；RealityScan 2.2.0.119430 可执行文件及 COLMAP 导出 XML 存在。RealityScan 未启动；其安装路径不在代码默认查找位置，后续须显式设置 `GSSTUDIO_REALITYSCAN_CLI=D:\UnrealEngine\RealityScan_2.2\RealityScan.exe`。
- RTX 4060 Ti 当前检查为 9,956/16,380 MiB 占用、97% 利用率。GS Studio 默认共享锁 `C:\Users\jinchao\AppData\Local\GSStudio\gpu.lock` 不存在；显卡仍繁忙，且 NVIDIA 进程列表因权限限制未给出占用者身份。不能把空闲的 GS Studio 锁等同于可用 GPU。
- `.venv-gsplat\Scripts\python.exe` 仍不存在；未执行 RealityScan、训练或真实发布 EXE 构建/启动验收。发布目录中未发现 `GSStudio.exe`。短段已完成 CPU 预处理 smoke；GPU 阶段与长段另行验证。
- 短段 `flight-0004` 新建 `20260928T072324Z-4dad3aa4` Run，使用 5 秒 smoke 配置，仅通过 FFmpeg CPU 解码和 `preprocess`。Run 审计复核 `preprocess=succeeded`、`mask/reconstruct/qa=pending`；5 个候选帧与 5 个精选帧的清单身份可读，项目浏览返回 Run `readable=true` 且 `problems=[]`。原片 SHA256 再次确认不变。首次尝试因 CPU 路径也请求用户级 GPU 锁而被沙箱阻断；该锁范围现已修正为仅在 MediaSDK 的 INSV 探测及准备时申请。授权进程产出的临时 JPEG 权限经限定在该派生 Run 内恢复继承后，普通工作区进程审计通过。未启动 GPU 阶段。
- 本轮修正训练在等待共享 GPU 锁时的状态和取消动作；取消若已有检查点则保留已有步数与证据。无检查点的失败实验可在相同输入与配置核验后显式从第 0 步重试；续跑前把原 dispatch、日志、训练摘要和失败记录归档并记录哈希。训练控制请求绑定当前进程代次，旧请求不会自动作用于重启的训练进程。外部导入若训练已成功但实验记录未写回，可核验并采纳成功证据；发布成功还需训练配置、评估记录和 PLY 哈希一致。Qt 对照改为对同一次读取的图像字节完成哈希校验和显示。上述结论仅由 CPU/离屏测试支持，真实训练仍待验收。

## AC-01～20 状态

| AC | 当前证据 | 结论与剩余条件 |
| --- | --- | --- |
| 01 正常导入 | 普通 COLMAP/共享包 CPU fixture 检查 | 实现待验收；需真实 BIN/TXT/JPG/PNG 与 Qt 导入 |
| 02 输入异常 | 缺分组、多模型、源变化等单测 | 实现待验收；坏图、中文路径及多模型 Qt 呈现待核对 |
| 03 导入预览 | Qt 相机与稀疏点代码 | 目视与真实点云待验收 |
| 04 遮罩与分组 | 显式分组和源身份单测 | 真遮罩极性、尺寸及泄漏待验收 |
| 05 外部位姿直训 | 独立不可变训练包与实验代码 | 无真实 gsplat/Postshot 训练证据 |
| 06 生命周期 | 安全步边界 pause/resume/checkpoint/stop 单测 | GPU 训练、进度、等待与结束产物待验收 |
| 07 恢复一致性 | 控制身份、外部源重定位与恢复单测 | 真实中断/重启和检查点待验收 |
| 08 资源与失败 | 错误分类及 GPU 锁代码 | GPU 忙/OOM/不可写的隔离验证待执行；当前 GPU 已被其他会话占用 |
| 09 实时查看 | 请求相机/序号匹配与 Qt 对照测试 | 真实训练渲染及交互目视待验收 |
| 10 基础编辑 | 撤销、取消、拾取权重 CPU 测试 | 真模型 GPU 拾取与手柄目视待验收 |
| 11 重开与搬移 | Capture 和外部导入源重定位的哈希单测、冻结路径布局单测 | 真实项目、EXE 目录搬移及编辑历史待验收 |
| 12 导出互操作 | 合成 SH3 PLY 编辑导出后经 CPU 回读，属性、数量、位置与源保护通过 | 真实训练 PLY 的 GPU 回读及独立查看器尚无证据 |
| 13 UI | A 设计静态选择，Qt 离屏检查 | DPI、中文路径、窗口尺寸与用户审看待验收 |
| 14 本地离线 | onedir 构建/校验脚本 | 尚无 EXE；断网与离线启动待验收 |
| 15 性能 | 无当前版本实测 | 待真实训练与大模型测量 |
| 16 画质 | 无当前版本实测 | 待代表素材及用户审看；DJI 不覆盖室内/全景 |
| 17 创建与探测 | 两段 DJI 的 CLI 登记、probe、ingest；短段 5 秒 Run 输入准备与预处理通过；原片身份未改变；合成 Capture 错哈希重定位被拒 | **部分证据**；Qt 创建、真实搬移、异常和只读全场景待验收 |
| 18 Run 门禁与恢复 | 短段 smoke Run 的预处理清单及 `audit_run` 通过；worker 契约代码 | 遮罩、RealityScan、repair、QA、重启待验收 |
| 19 分段与清理 | GUI 与应用层代码 | 无通过 QA 的分段和清理场景，待验收 |
| 20 状态与故障定位 | 服务/worker 契约与离屏测试 | 真机日志、等待、意外退出及操作恢复待验收 |

INSV、标准全景及室内画质没有本轮相应真实素材，相关结论保持**阻断**。短训只用于链路检查，不证明画质或 AC-15/16。下一层验证须等工作区 GPU 空闲并备好独立固定版本运行时，再让短段 Run 进入遮罩和重建；若重建或 QA 未通过，原样记录失败，并将下游合成故障测试与真实全流程结论分开。
