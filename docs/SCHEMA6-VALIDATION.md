# Schema 6 实施与验证记录

本轮将新 Run 默认改为 1 fps / 每秒一张全景，薄弱区间按需 2 fps 补抽一次，并接入阶段清理。保留历史 schema 1–5 的配置与恢复分支，没有对现有 Data、Run 或实验执行清理、重建或训练。

## 已接入的行为

- 新 Run 独占候选目录，初始输入哈希与补抽身份分开记录；不更新 Capture 的历史 prepared 指针。
- 补抽使用 Capture 起点对齐的时间网格，按源帧索引去重。复用已有投影和遮罩，合并保留及剔除清单，并保留人工排除状态。
- 清理前核验下游产物，使用 Run 锁、路径白名单、哈希与持久化删除计划。Windows 文件占用或清理失败不将已成功计算阶段改为失败。
- 未选中尝试清理后标记为仅保留审计记录；选中重建的完整过滤输入、位姿、COLMAP、训练检查点和成果继续保留。
- schema 6 训练包复用图片与同极性遮罩；Postshot 反向遮罩单独保存。同盘优先硬链接，不支持时校验后复制并记录占用。
- 新增 `cleanup` 预览与 `--apply` 重试入口、`--keep-intermediates` 调试选项；QA、后端对照、遮罩确认和训练入口支持 schema 6。

## 验证

完整保留测试集使用 APP 主虚拟环境运行：

```powershell
Push-Location 'D:\Project\3DGS\APP'
try { & '.\.venv\Scripts\python.exe' -m pytest tests --tb=short }
finally { Pop-Location }
```

结果：**299 项通过**。保留一条已存在的 PyTorch `grid_sample` 默认参数警告。

新测试覆盖非整数帧率、嵌套采样网格、局部补抽去重和恢复、过滤清单合并、成功阶段清理、失败及待人工确认时保留输入、删除中断、Windows 文件占用与目录连接、跨进程 Run 锁、硬链接空间统计、复制降级、清理后 QA 重跑与两后端输入准备。外部重建和生产训练用 CPU 合成数据及模拟组件替代；测试通过不能代替 1 fps 的实际覆盖及画质验收。

从 `C:\Windows` 通过 `gsdb.ps1 cleanup --help` 调用成功；README、AGENTS、工作手册与维护说明的相对文档链接检查通过。`git diff --check` 通过。

## 清理示例与操作入口

[清理预览和完成记录示例](examples/schema6-cleanup.json) 来自独立临时目录中的 CPU 合成数据，包含真实生成的哈希、文件状态与空间统计，不代表任何生产 Run。实际记录保存在各 Run 的 `cleanup/<hash>.json`，失败待重试说明在 `cleanup-pending.json`。

日常操作见 [当前工作手册](CURRENT-WORKFLOW.md)。旧 Run 不会因恢复而自动升级；使用新采样需要新建 Run。正式 SH 3、0.5% 阈值、QA 门槛、Postshot 默认后端及人工验收条件保持不变。
