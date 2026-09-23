# Skill 查询与文档方法

目的：为本地项目建立可交接、可追踪的开发资料，不为了目录结构安装开发环境。

## 查询结果与取舍

| 候选 | 适用性 | 本次处理 |
| --- | --- | --- |
| Spec Kit 的 specify、clarify、tasks、checklist 流程/skill | 适合组织需求、澄清问题、关联任务和检查文档完整性 | 参考流程，手工形成 Markdown 资料；不安装 CLI、不生成代码 |
| notion-spec-to-implementation | 适合把 Notion PRD 关联到计划和任务库 | 已读公开 SKILL.md；本次没有 Notion 目标，不调用其连接与写入流程 |
| 本地 skill-installer | 用户决定安装某个 skill 时适用 | 本次仅查阅安装规则，没有安装或更改全局 skill |
| 已有 visualize skill | 将来制作静态交互稿或解释视图时可能有用 | 不是建立文档目录所必需，本次不调用 |

依据：[Spec Kit 官方说明](https://github.com/github/spec-kit)、[Notion skill 原文](https://github.com/openai/plugins/blob/main/plugins/notion/skills/notion-spec-to-implementation/SKILL.md)。查到候选不意味着已安装或实际执行其流程。

## 采用的组织约定

- PRD 说明用户价值与边界；需求清单定义稳定编号和可观察完成标准。
- UI 设计独立管理，静态方案选定后再约束正式实现。
- 开发路线图描述阶段与完成条件；Backlog 描述具体交付物、依赖和状态。
- 验收按用户流程与关键失败场景组织，并关联需求；执行与结果分开记录。
- 技术决策、风险和未决事项单独登记，不在需求文档里隐式固定技术方案。
- 当前基线不混同未来目标，历史通过记录不自动继承到新代码。

这种结构参考 [Spec Kit 需求模板](https://raw.githubusercontent.com/github/spec-kit/main/templates/spec-template.md) 与 [任务模板](https://raw.githubusercontent.com/github/spec-kit/main/templates/tasks-template.md)，按本项目简化。没有采用模板中的自动测试、并行执行或发布步骤；用户当前限制优先。

本目录不声称符合某项 ISO/IEEE 认证，也不绑定某个任务平台。Markdown 便于人和 Agent 阅读，后续可按需求 ID 转成 issue，而不先建立额外系统。
