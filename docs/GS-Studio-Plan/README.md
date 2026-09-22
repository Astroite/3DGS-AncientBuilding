# GS-Studio 早期规划与设计资料

本目录保存 2026-09-13 形成的下游训练/编辑产品计划、A 方向设计和技术评估，供后续实现追溯。**现行项目路线图在 [APP](../GS-STUDIO-PROJECT-ROADMAP.md)，工程路线图也在 [APP](../GS-STUDIO-ENGINEERING-ROADMAP.md)。**

**本目录基线：0.5 · 2026-09-13。** DEV-04 静态设计评审通过、当时 M2 完成；DEV-05 评估已交付，产品未验收。2026-09-22 用户将范围扩展为 APP 全流程 GUI，并选择 Python/gsdb/gsplat 主线；旧范围与旧推荐以当时记录为准，不再作为现行实施顺序。

本目录是 APP 仓库内的历史规划资料，不是第二套现行路线图。APP 与 Data 均未因路线图整合而移动；当前可运行原型仍在 APP。当前实际操作请使用 [APP 工作手册](../CURRENT-WORKFLOW.md)。工作区顶层的 `GS-Studio-Plan` 原目录保留，本目录为可随 APP 提交的参考副本。

## 当时的核心目标

当时的 P0 从 RealityScan 的 images + COLMAP 开始，涵盖检查、训练观察、基础编辑、保存项目和 PLY 导出。**现行 GS-Studio 路线已扩展到地点/场景/Capture、Run、重建与 QA 的 GUI 操作**；下列旧 PRD、需求和验收稿仍可作为下游功能基线。

**当前执行约束：不运行 CPU/GPU 测试，不启动训练、应用验证或后台监测，不干扰其他会话。测试与画质验收仅保留计划，须有用户后续明确要求才恢复执行。** GPU 空闲不构成自动恢复授权。

## 阅读入口

| 要了解的内容 | 文档 |
| --- | --- |
| 做什么、为谁做、哪些不做 | [产品需求 PRD](product/PRD.md) |
| 每项功能的编号与完成标准 | [需求清单](product/REQUIREMENTS.md) |
| Postshot 与同类工具的参考价值 | [竞品分析](research/COMPETITIVE-ANALYSIS.md) |
| 官方资料与可信程度 | [来源登记](research/SOURCES.md) |
| 相关 skill 和组织方法 | [Skill 与方法评估](research/SKILLS-AND-METHODS.md) |
| UI 三套方向、交付物与评审标准 | [UI 风格探索](design/UI-EXPLORATION.md) |
| 六张设计稿、样式说明与方向比较 | [DEV-02 交付与比较](design/deliverables/COMPARISON.md) |
| 已选 UI 方向及后续细化重点 | [D-04：采用 A 方向](planning/decisions/D-04-UI-DIRECTION.md) |
| A 完整页面、状态、组件与尺寸适配 | [DEV-04 设计交付](design/specification-a/README.md) |
| 技术路线建议、原型审计及验证交接 | [DEV-05 技术评估](research/technical-evaluation/README.md) |
| 现行项目阶段与进入条件 | [APP 项目路线图](../GS-STUDIO-PROJECT-ROADMAP.md) |
| 现行入口整合与工程顺序 | [APP 工程路线图](../GS-STUDIO-ENGINEERING-ROADMAP.md) |
| 2026-09-13 原阶段与任务记录 | [旧路线图](planning/ROADMAP.md)、[旧 Backlog](planning/BACKLOG.md) |
| 已定约束、未定事项和风险 | [决策与风险](planning/DECISIONS-AND-RISKS.md) |
| 功能、画质、性能如何验收 | [验收计划](quality/ACCEPTANCE.md) |
| 当前实现与历史验证的实际边界 | [现状基线](status/BASELINE.md) |
| 交接 Agent 的操作限制 | [AGENTS.md](AGENTS.md) |

## 文件夹结构

```text
GS-Studio-Plan/
  README.md
  AGENTS.md
  CHANGELOG.md
  research/       竞品、证据和方法评估
    technical-evaluation/ DEV-05 五份静态技术评估
  product/        PRD 与带编号的需求
  design/         UI 探索任务与交付要求
    deliverables/ 六张 SVG、三份样式说明、比较与评审入口
    specification-a/ A 方向 20 张 SVG 与 4 份设计规范
  planning/       路线图、任务、决策与风险
  quality/        验收场景与证据规则
  status/         实现及验证基线
  templates/      后续任务与决策记录模板
```

## 资料使用与维护

- 原 I（导入）、T（训练）、V（查看）、E（编辑）、O（交付）、UI、NFR 和 AC 编号在原功能范围内继续作追踪依据；上游新增编号见 APP 项目路线图。
- “已整理”“有原型”“设计评审通过”均不等于正式功能或画质验收。历史验证只支持对应版本与场景。
- 2026-09-13 的 LichtFeld 主推荐是静态评估结论。用户现已选择 Python/gsdb/gsplat 为主线；评估原文保留为来源，见[路线决策](planning/decisions/D-03-GS-STUDIO-ROUTE.md)。
- 产品方向与当前里程碑更新在 APP 两份路线图，本目录保留来源与历史设计，不维护第二套现行路线图。重要范围与默认值变更仍在 [CHANGELOG](CHANGELOG.md) 记录。

DEV-02/03 的 A 方向选择与 DEV-04 静态设计认可继续有效；正式全流程 GUI、运行验证和产品验收仍未完成。
