# APP 工作入口

操作 Capture、Run、重建或训练前，先读 [当前工作手册](docs/CURRENT-WORKFLOW.md)。开发和脚本清单见 [维护说明](docs/MAINTENANCE.md)。GS-Studio 的未来方向见[项目路线图](docs/GS-STUDIO-PROJECT-ROADMAP.md)和[工程路线图](docs/GS-STUDIO-ENGINEERING-ROADMAP.md)；路线图不取代当前操作手册，不从工作区历史归档提取默认运行命令。

- 使用原生 Windows 环境和 scripts/session.ps1；gsdb.ps1 复用同一初始化。
- 保留用户未提交改动、运行环境、SDK、密钥和原始数据。
- 新 Run 使用 schema 6；旧 schema 1–5 配置读取和恢复行为保持兼容，不把新默认写入历史配置。
- Run 清单、输入哈希、分段 QA 和训练结果共同决定状态。不得将失败 Run 手改为成功或绕过完整性校验。
- 分段可训练与全路线覆盖独立。实际片段人工验收前不自动开始全量。现行 CLI 与 Run 默认不因规划自动切换；未来新 GUI 实验预选 gsplat 是单独确定的界面目标。
- 历史 QA 例外和固定场景队列不构成新任务授权。发送图像到外部服务仍需用户明确授权。
- 修改入口或文档时核验实际 CLI 和引用；仅整理文档不运行实际重建或训练。

- schema 6 默认 1 fps，局部 2 fps 补抽最多一次。清理必须使用 retention 的白名单、消费者校验和 Run 锁；不得删除原片或凭退出码清理失败阶段。
