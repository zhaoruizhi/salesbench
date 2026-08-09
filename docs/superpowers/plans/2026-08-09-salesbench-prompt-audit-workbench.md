# SalesBench Prompt 与审计工作台实施计划

1. 先为运行快照加载、正式/Smoke 组织、风险检测与 HTML 渲染写失败测试。
2. 实现 `tools/audit_workbench/`，只读取声明的结果目录和 Prompt builder，不接触 `.env`。
3. 添加 `configs/pilot64_gpt4o_v6_delivery.json`，把正式、Smoke 和私有互动分析来源显式分开。
4. 运行生成器，非破坏性地产生 `outputs/formal/`、`outputs/smoke/` 和完整本地审计 HTML。
5. 生成小于 1 MB 的会话内 HTML fragment，展示汇总、全部 Prompt 和高优先级审计样本。
6. 验证交互、隐私边界、响应式布局与全量测试。
7. 提交生成器、配置、测试和审核文档；不提交 `.env`、`outputs/` 或嵌入私有样本的 HTML。
