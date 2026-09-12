# APP 维护说明

日常操作只使用 [当前工作手册](CURRENT-WORKFLOW.md)。本页记录入口职责、兼容范围、测试和归档，不提供另一套默认流水线。

## 入口与运行环境

| 入口 | 用途和输入 | GPU / 写入行为 |
| --- | --- | --- |
| `gsdb.ps1` | 任意目录调用 GSDB，参数原样传给 CLI | 取决于子命令 |
| `scripts/session.ps1` | 初始化主解释器、Data、helper 和 SDK，支持 `-DataRoot` | 不启动 GPU，不写 Data |
| `scripts/bootstrap-windows.ps1` | 首次安装主环境 | 写 .venv，下载依赖；不自动运行 doctor |
| `scripts/bootstrap-gsplat-windows.ps1` | 安装独立训练/评估环境，支持 `-SkipPackages` | 写 .venv-gsplat，下载依赖 |
| `scripts/apply_nerfstudio_patch.py` | 安装时核验并应用投影 clamp 和嵌套路径补丁 | 修改环境内依赖代码，不处理素材 |
| `scripts/test-mediasdk-helper.ps1` | 指定 LocationId、SceneId、CaptureId，按 5 fps 验收；可用 RunId 恢复 | GPU、解码和 prepared 缓存；不是轻量单测 |
| `scripts/smoke-native-gsplat.py` | 独立环境，`--output` 指定新的合成测试目录 | GPU、少量合成训练与检查点；不访问 Capture |
| `scripts/check-native-export.py` | 独立环境，`--experiment` 与 `--dataset` 核验 PLY 回读 | GPU，写 export-diagnostic.json；不重训 |
| `scripts/compare-backends-v5.py` | 主环境，`--run-dir --segment --output` 对照通过的分段 | 默认实际训练；`--dry-run` 仍会准备包及记录 |
| `tools/mediasdk-helper/build.ps1` | 构建连接本机 SDK 的 helper | 写 build，不运行采集 |
| `doctor` | 检查当前 Windows 重建与所选后端 | 使用 GPU 锁，产生并清理诊断临时文件；不训练 |

独立 GPU 诊断脚本应在流水线空闲时执行；不要从独立脚本绕过正在使用的生产 GPU 锁。环境、SDK、helper 构建产物和 `env` 配置不是历史垃圾，不在文档清理范围内。

后端对照已改为直接输入 Run 目录，不再接受 `--case-file`。保持两后端、补偿开关、步数、时长及恢复选项；失败/阻断返回非零。复用成功结果时核验包身份、预算和 PLY 哈希。`--dry-run` 是准备动作，不是只读检查。

## 兼容范围

- 保留 schema 1–4 清单读取、配置哈希和恢复语义；新默认仅作用于新 schema 5 Run。
- `postshot-prepare`、`postshot-review`、`postshot-train` 为历史数据入口。schema 5 使用 `train --segment --backend`，不通过历史实验脚本跳过 QA。
- 旧 `export` 读取 Nerfstudio 训练记录，不能用于 schema 5 的分段成果。`qa approve/reject` 也不能替代未接通的分段成果发布。
- 保留现行投影、COLMAP 文件转换、PLY 处理及路径兼容所需的底层代码；移除独立历史脚本不等于删除这些能力。
- 默认 doctor 检查 RealityScan；只有 `--require colmap` 才运行旧 CUDA COLMAP 检查。Postshot 可执行文件/版本检查不验证 Studio 训练许可。
- Nerfstudio 仍用于投影和转换，因此保留安装脚本调用的补丁程序。已归档的静态 patch 文本不是运行依赖。

## 验证

```powershell
$AppRoot = 'D:\Project\3DGS\APP'
. (Join-Path $AppRoot 'scripts\session.ps1')
Push-Location -LiteralPath $AppRoot
try {
    & $GsdbPython -m pytest -ra
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    git diff --check
} finally {
    Pop-Location
}
```

单测使用隔离锁、临时目录和模拟外部组件。保持旧配置兼容测试；文档维护不启动真实 Capture 重建、模型训练或 SDK 硬件验收。新增文档命令需核验 CLI 帮助、PowerShell 语法及模拟调用；现行链接不能依赖历史归档存在。

## 本轮归档索引

归档目录：`<workspace>/archive/APP-pre-cleanup-20260912T162225/`。
基准提交：`a5d5ab54b24bbaf07657584091916d7749086f26`。
归档清单：目录内 `archive-manifest.json`，包含原相对路径、SHA-256、原因和动作。43 份原文已复制核验，29 个历史文件从 APP 移出；其余为重写前备份。归档不属于应用依赖，不会自动执行或恢复其中的队列。

| 分组 | 移出内容 |
| --- | --- |
| WSL / Conda | `bootstrap-wsl.sh`、`patch-conda-lock-manylinux.py`、三个 Conda 环境/锁/虚拟平台文件 |
| 固定实验 | `run-009-demo.ps1`、`assess-pipeline-20260912.py`、`benchmark-v5.py`、`continue-validation-v5.py` |
| 旧辅助入口 | `replay-segments-v5.py`、`prepare-postshot-experiment.py`、`hdr_to_sdr_jpg.py`、`rotate_gaussian_ply.py` |
| 专用代码与测试 | `src/gsdb/benchmarks.py`，环境锁、Postshot 例外及历史 benchmark 的三份专用测试 |
| 静态副本 | `patches` 中两份补丁文本 |
| 历史文档 | 旧 V0.2/009/V5 手册、日期固定验证与评估、研究与规范草案、原 docs/archive |

README、AGENTS、原 CURRENT-WORKFLOW 和保留入口的修改前版本也已备份。原本未被 Git 跟踪的运行环境、SDK、密钥及现有数据未纳入归档。测试结果与实际清理核验见归档目录内 `cleanup-result.json` 和 `pytest-result.txt`；历史实验记录不代表本轮重新完成了画质验收。
