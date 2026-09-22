# 来源登记

最后整理：2026-09-13。E1=官方文档/仓库声明；E2=本地文件；E3=此前会话执行记录；E4=计划或推断。官方声明不等于本机实测。

| ID | 来源 | 等级 | 用途与限制 |
| --- | --- | --- | --- |
| S01 | [Postshot 官网](https://www.jawset.com/) | E1 | 产品能力；不证明全部档位都包含每项能力 |
| S02 | [Postshot 导入说明](https://activation.jawset.com/docs/d/Postshot%2BUser%2BGuide/Importing%2BImages) | E1 | 外部相机、图片配对、COLMAP 格式 |
| S03 | [Postshot 训练配置](https://activation.jawset.com/docs/d/Postshot%2BUser%2BGuide/Interface/Training%2BConfiguration) | E1 | 遮罩、补偿、训练参数与配置定位 |
| S04 | [Postshot 价格页](https://www.jawset.com/shop/) | E1 | 档位功能；金额为动态内容，本轮不采信占位数字 |
| S05 | [LichtFeld Studio](https://github.com/MrNeRF/LichtFeld-Studio) | E1 | 训练、编辑、恢复、导出及自动化 |
| S06 | [Brush](https://github.com/ArthurBrussee/brush) | E1 | 训练预览、输入对照、遮罩与查看 |
| S07 | [SuperSplat](https://developer.playcanvas.com/user-manual/supersplat/) | E1 | 编辑/发布的产品定位 |
| S08 | [Spec Kit](https://github.com/github/spec-kit) | E1 | 需求、规划、任务与反馈的组织方式 |
| S09 | [Spec 模板](https://raw.githubusercontent.com/github/spec-kit/main/templates/spec-template.md) | E1 | 场景、需求编号、边界与成功标准 |
| S10 | [任务模板](https://raw.githubusercontent.com/github/spec-kit/main/templates/tasks-template.md) | E1 | 按用户流程拆任务、注明依赖；不照搬其执行命令 |
| S11 | [notion-spec-to-implementation skill](https://github.com/openai/plugins/blob/main/plugins/notion/skills/notion-spec-to-implementation/SKILL.md) | E1 | 查阅后确认依赖 Notion，不用于本地目录实际操作 |
| S12 | [Windows 内容布局与间距](https://learn.microsoft.com/en-us/windows/apps/design/basics/content-basics) | E1 | UI 分组、间距与信息层级参考 |
| S13 | [Windows 字体指南](https://learn.microsoft.com/en-us/windows/apps/design/signature-experiences/typography) | E1 | 字体层级与可读性参考，不意味着选用 WinUI |
| L01 | [APP 工作规则](../../../AGENTS.md) | E2 | 保护数据、环境与原有工作流 |
| L02 | [当前工作手册](../../CURRENT-WORKFLOW.md) | E2 | RealityScan、共享包、分段 QA 和训练边界 |
| L03 | [原型说明](../../STUDIO.md) | E2 | 原型声称支持的操作及限制，不能等同产品已验收 |
| L04 | [此前 UI 计划](../../STUDIO-UI-PLAN.md) | E2 | 三方向探索与用户限制；本目录保存当时资料，现行路线图位于 APP/docs |
| L05 | 工作区外 `studio-validation/gui-02/gui-result.json`（未纳入 APP 仓库） | E2 | 合成数据窗口检查，不能证明 UI 被用户接受；远端读者不可据此复核文件 |
| L06 | 此前会话：282 passed，1 warning | E3 | 历史回归结果；之后代码仍有改动，未覆盖最新状态 |
| L07 | 此前会话：segment-007 导入结果 | E3 | 256 图、226 训练图、224,480 点、无导入错误；不代表训练画质通过 |
| L08 | 此前会话：GPU 检查等待超时 | E3 | 实际未完成 GPU 集成测试；没有通过报告 |

本轮只阅读已有材料、查询网页并写文档，没有复跑 L05～L08。若文件被后续会话更新，应重新确认其版本；此目录不是原型源码快照。
