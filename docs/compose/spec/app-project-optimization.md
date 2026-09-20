---
feature: app-project-optimization
status: delivered
updated: 2026-09-21
branch: main
commits: uncommitted-on-main (working tree vs dd15cf2) # fill SHA range after commit
---

# APP 工程综合体检与优化

## Report

**What was built**

在 `D:\Project\3DGS\APP` 的 main 工作树上完成 compose-next 优化切片 A–F（F 限定为 APP Studio 代码卫生）：

- **A 结构**：新增 `src/gsdb/schema_route.py`（schema 谓词）与 `src/gsdb/training_schedule.py`（densification / TrainSettings / CullSettings）；`pipeline.py` 入口 `preprocess_run` / `reconstruct_run` / `training_image_count` 改用谓词，公共符号仍从 `pipeline` 再导出，行为与 schema 1–5 兼容不变。
- **B 编码**：`session.ps1` 强制 `PYTHONUTF8`/`PYTHONIOENCODING` 与 Console UTF-8；`processes.py` GPU 采样子进程显式 UTF-8；`native_train` 的 `cmd.exe set` **优先 OEM/mbcs 解码**（避免中文 Windows 上把环境变量解成 U+FFFD）。
- **C CLI**：顶层 help 标明 schema 6 现行路径；legacy 命令 docstring 使用 `(legacy)`（避免 Rich 吞掉 `[legacy]`）；现行入口标 `Current workflow:`；**不改命令名、不删入口**。
- **D 测试卫生**：已知 `grid_sample` 警告改为 `pytest.warns`；新增 `tests/test_schema_route_and_schedule.py` 覆盖 V4/V5/V6 配置与路由谓词；UTF-8 捕获回归测试。
- **E 清理可观测性**：`retention.cleanup_run` 报告增加 `planned_files`、`status_counts`、`logical_gib`、`reclaimable_gib`、`hardlink_note`；**白名单/消费者校验/删除条件未改**。
- **F Studio 卫生**：`studio.py` / `studio_data.py` / `studio_runtime.py` 模块说明 UTF-8 I/O 边界；进度 JSONL 使用 `ensure_ascii=False` + 显式 utf-8/newline。未推进 UI/产品路线。

**Verification**

| 命令 | 结果 |
| --- | --- |
| `.venv\Scripts\python.exe -m pytest -ra` | **308 passed**（最终），约 47–55s，无 warnings summary |
| 审查后回归（误删 `math` / `__future__` 位置） | 曾 4 failed → 修复后 **308 passed** |
| `from gsdb.pipeline import densification_schedule, …` | 通过，与 `training_schedule` 同源 |
| `gsdb --help` | 显示现行路径与 `(legacy)` 标记 |
| `git diff --check` | 通过（仅有 CRLF/LF 换行警告） |
| 独立 review（general-1） | 6/7 PASS；T-F1 原判 incomplete → 已补交付后复测全绿 |

**Journey log**

1. Grill 选「综合体检后再定向」后，用户一次勾选 A–F；F 需再澄清为 APP Studio 卫生而非 Studio 产品。
2. Spec Tasks 段曾因 `old_string` 不匹配陷入编辑循环——应整段重写而非微调重试。
3. Rich/Typer 会吞掉 help 中的 `[legacy]` 标记，须用 `(legacy)`。
4. 从 `pipeline.py` 抽出 densification 后不可顺手删 `import math`：`quality_snapshot` 仍依赖。
5. `tests/` 在 `.gitignore` 中（dd15cf2 起）：测试改动仅在磁盘验收，不会进 git diff。

## [S1] Problem

用户 `/compose-next 优化现有的工程`。APP 为 Windows 原生 360→3DGS 生产应用；范围经 Grill 收敛为体检后定向，并在 main 上实施（用户拒绝 worktree）。

硬约束：兼容 schema 1–5；不改 QA/默认后端/失败语义；不启动真实重建/GPU 训练；不外发图像。

## [S2] Design

### 体检基线

- pytest：实施前 299 passed + 3 警告；实施后 308 passed、无警告 summary。
- 主要债务：`pipeline.py` 上帝模块、schema 魔数分发、GBK 子进程风险、CLI legacy 表面。

### 已交付切片

| 任务 | 交付 | 行为 |
| --- | --- | --- |
| T-B1 | session/processes/native_train 编码 | 保持；cmd.exe 优先 OEM |
| T-D1 | 警告显式化 + schema/路由测试 | 测试-only |
| T-C1 | CLI help 标注 | 不改命令名 |
| T-A1 | schema_route + training_schedule + 入口谓词 | 与 schema1–6 魔数等价 |
| T-A2 | v5/v6 模块 docstring | 语义不变 |
| T-E1 | cleanup 报告字段 | 删除语义不变 |
| T-F1 | Studio UTF-8 边界说明 + 进度 JSONL | 无 UI/产品功能 |

### 设计原则

1. 行为保持优先；2. 历史 Run 恢复路径不删；3. CPU 可观察验收；4. 最小切片。

## [S3] Out of Scope

真实重建/训练/画质验收；Data/manifest/QA 篡改；gsplat 默认提升；外部 vision QA；Studio 产品路线（DEV-05/D-03）；worktree。

## Tasks

- [x] T-B1: 统一 Windows 子进程/会话 UTF-8；cmd.exe 环境变量优先 OEM 解码 — acceptance: 显式编码；pytest 全绿 (covers: S2)
- [x] T-D1: 清理测试警告并补 schema/config 覆盖 — acceptance: 无 unhandled 警告；新测试通过 (covers: S2; depends: T-B1)
- [x] T-C1: CLI 帮助与手册对齐，legacy 标明且不改名 — acceptance: help 可见 (legacy)/Current workflow；workflow 文档测试通过 (covers: S2)
- [x] T-A1: schema 路由与 training_schedule 抽出，入口 predicates — acceptance: 公共 import 可用；308 测试通过；无 config/hash 变化 (covers: S2)
- [x] T-A2: pipeline_v5/v6 职责说明 — acceptance: 仅 docstring；v5/v6 测试通过 (covers: S2; depends: T-A1)
- [x] T-E1: cleanup 报告可观测性 — acceptance: 新字段存在；删除条件未改 (covers: S2)
- [x] T-F1: APP Studio 代码卫生 — acceptance: studio 模块 UTF-8 I/O 边界明确；studio CPU 测试通过 (covers: S2)
