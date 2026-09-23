# D-03：GS-Studio 正式技术主线

- 日期：2026-09-22
- 状态：已接受；实现与质量仍待验证
- 决定者与依据：用户确认以 APP 现有 Python/gsdb/gsplat 为主线；[DEV-05](../../research/technical-evaluation/README.md)是此前的静态评估输入，不是运行对照。
- 关联需求和任务：I/T/V/E/O、UI、NFR；原 DEV-06～22 及 [APP 工程路线图](../../../../architecture/GS-STUDIO-ENGINEERING-ROADMAP.md)。
- 替代的既有决定：关闭 D-03 待决状态；未改写 DEV-05 当时的 LichtFeld 推荐结论。

## 要解决的问题

明确 APP 升级成全流程 GUI 时应基于哪条训练与应用代码主线，避免将静态推荐误当为已批准架构。

## 可行选项

| 选项 | 取舍 |
| --- | --- |
| Python/gsdb/gsplat＋正式 GUI | 直接承接 APP 现行清单、流水线、分组、训练和兼容代码；仍须解决正式 UI、恢复、预览资源协作与画质验收 |
| LichtFeld 二次开发 | DEV-05 的当时主推荐，有可复用渲染/编辑/外观机制；需要迁移现行输入与分组语义、兼容和构建验证 |
| 推迟选型 | 可等待更多实测，但阻断工程路线与任务排期 |

## 决定与理由

用户选择 Python/gsdb/gsplat 为 GS-Studio 的正式主线，并要求 APP 原位升级。此选择确定复用方向，**不代表 gsplat 画质、性能、完整恢复或正式 GUI 已通过验收**。LichtFeld 评估保留为历史研究和未来必要时的参考，不作为当前主实现任务。

## 后续影响

正式 UI 框架、项目存储、普通 COLMAP 分组与非穿透拾取仍需在相应里程碑前定案。新 GUI 实验的默认后端、现行 CLI 兼容及全流程范围见 [D-11](D-11-GUI-FIRST-SCOPE.md)。D-02 的运行验证暂停约束继续有效。
