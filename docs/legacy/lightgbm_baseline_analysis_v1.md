# LightGBM Baseline 分析报告（V1）

## 1. 结论摘要

本文分析当前 `LightGBM` 数值版 baseline 在 SalesBench V1 上的表现，并给出后续提分方向与 benchmark 改进建议。

当前 baseline 已经可以作为第一版传统机器学习基线使用：

- 样本覆盖完整：`1200 / 1200`
- OOF 交叉验证：`5-fold StratifiedKFold`
- 特征规模：`128` 个结构化特征，其中 `7` 个类别特征
- 排序能力有明显信号：`spearman_rho = 0.4641`，`auc_top30 = 0.7216`
- Pairwise 排序优于随机：`pairwise_accuracy = 0.6698`
- 分数误差比常数均值基线更低：`MAE 0.1749` vs `0.2029`

但这版模型还不是一个足够强的最终 baseline：

- 预测分数明显向中间收缩，真实分数范围是 `0.0000-1.0000`，预测范围只有 `0.1378-0.8822`
- 预测分数 IQR 只有 `0.1568`，真实分数 IQR 是 `0.3623`
- `final_grade` 的分类效果弱，`macro_f1 = 0.2253`
- A/E 两端样本误差最大，说明模型对爆款和低表现视频都不够敏感
- `publish_context` 是当前最强信号来源，尤其 `video_sales_power`，因此 benchmark 需要明确“上下文可用信息”和“疑似后验/近标签字段”的边界

一句话判断：当前 LightGBM V1 是一个合格的结构化 baseline，能提供可复现的第一条分数线；下一轮最值得做的是分数校准、Top-K/排序优化、特征来源分轨，以及对 `video_sales_power` 等字段做泄漏口径审查。

## 2. 实验设置

本次分析基于以下输出文件：

| 文件 | 作用 |
| --- | --- |
| `outputs/baselines/lightgbm/oof_predictions.jsonl` | 5-fold OOF 预测 |
| `outputs/baselines/lightgbm/evaluation.json` | benchmark 评估结果 |
| `outputs/baselines/lightgbm/feature_importance.json` | LightGBM gain/split 重要性 |
| `outputs/processed/labels_v1.jsonl` | `final_score` / `final_grade` 标签 |

当前模型输入：

| 输入来源 | 是否进入当前模型 | 说明 |
| --- | --- | --- |
| `visual_features` | 是 | 视觉结构化特征 |
| `audio_speech` | 是 | 时长、语速、文本密度等 |
| `text_language` | 是 | 标题/文案结构化语言特征，不含原始文本 |
| `publish_context` | 是 | 作者、粉丝、品类、商品热度等上下文 |
| `cross_modal_consistency` | 是 | 标题、文案、商品、品牌等一致性特征 |
| `raw_video` | 否 | 只作为资产索引，不进入数值 baseline |

当前排除字段：

| 类型 | 字段 |
| --- | --- |
| 主键/标签 | `video_id`、`final_score`、`final_grade`、`top30_label`、`top10_label` |
| 路径 | `primary_video_path` |
| 原始文本 | `title`、`video_text`、`product_title`、`small_blue_word` |
| 明显身份/名称 | `author_name`、`douyin_handle`、`shop_name`、`brand_name` |
| 日期时间原文 | `publish_date`、`publish_time` |

## 3. 总体指标

### 3.1 当前 LightGBM 表现

| 指标 | 数值 | 解读 |
| --- | ---: | --- |
| `spearman_rho` | `0.4641` | 中等排序相关，说明模型能抓到一部分强弱顺序 |
| `pearson_r` | `0.4891` | 分数线性相关中等 |
| `mae` | `0.1749` | 平均绝对误差约 0.175 |
| `nmae` | `0.4828` | 相对真实分数 IQR 的误差仍偏大 |
| `macro_f1` | `0.2253` | 五档等级预测较弱 |
| `auc_top30` | `0.7216` | Top30 区分能力可用 |
| `auc_top10` | `0.7707` | 对极高分样本有一定排序信号 |
| `pairwise_accuracy` | `0.6698` | pairwise 排序明显优于随机 |
| `ece` | `0.3184` | 当前启发式置信度未充分校准 |
| `agent_score` | `0.6530` | 当前综合分 |

### 3.2 与常数基线对比

| 方法 | Spearman | MAE | Macro-F1 | AUC Top30 | Pairwise Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| 常数均值预测 | `0.0000` | `0.2029` | `0.0667` | `0.5000` | `0.5000` |
| LightGBM V1 | `0.4641` | `0.1749` | `0.2253` | `0.7216` | `0.6698` |

LightGBM 相比常数基线有明显增益：

| 指标 | 改善 |
| --- | ---: |
| MAE 降低 | 约 `13.8%` |
| Top30 AUC 提升 | `+0.2216` |
| Pairwise Acc 提升 | `+0.1698` |

因此这不是一个“只会猜均值”的模型，它已经学到了有效排序信号。

## 4. 误差诊断

### 4.1 分数被压缩到中间区间

| 分布统计 | 真实 `final_score` | 预测 `final_pred_score` |
| --- | ---: | ---: |
| mean | `0.4813` | `0.4810` |
| min | `0.0000` | `0.1378` |
| max | `1.0000` | `0.8822` |
| IQR | `0.3623` | `0.1568` |

模型均值对得很准，但方差明显不足。这个现象会带来两个直接问题：

- 高分样本被低估，A 档样本平均预测只有 `0.5715`
- 低分样本被高估，E 档样本平均预测达到 `0.4039`

这说明模型目前更像是在做“保守回归”，而不是充分拉开强弱差距。

### 4.2 分档预测偏向 B/C/D，中间档过多

真实等级是分位数构造的，因此 A/B/C/D/E 各 `240` 条。模型预测分布如下：

| 等级 | 真实数量 | 预测数量 |
| --- | ---: | ---: |
| A | `240` | `39` |
| B | `240` | `291` |
| C | `240` | `512` |
| D | `240` | `341` |
| E | `240` | `17` |

混淆矩阵如下，行是真实等级，列是预测等级：

| True \ Pred | A | B | C | D | E |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | `29` | `102` | `86` | `23` | `0` |
| B | `6` | `70` | `122` | `42` | `0` |
| C | `4` | `60` | `105` | `71` | `0` |
| D | `0` | `37` | `116` | `82` | `5` |
| E | `0` | `22` | `83` | `123` | `12` |

这进一步证明，模型的排序信号存在，但分数刻度和等级边界没有校准好。

### 4.3 A/E 两端误差最大

| 真实等级 | 样本数 | MAE | 平均偏差 `pred - true` | 真实均值 | 预测均值 | 同档率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | `240` | `0.2634` | `-0.2622` | `0.8337` | `0.5715` | `12.1%` |
| B | `240` | `0.1342` | `-0.1180` | `0.6193` | `0.5013` | `29.2%` |
| C | `240` | `0.0824` | `+0.0014` | `0.4802` | `0.4816` | `43.8%` |
| D | `240` | `0.1297` | `+0.1123` | `0.3342` | `0.4465` | `34.2%` |
| E | `240` | `0.2651` | `+0.2647` | `0.1393` | `0.4039` | `5.0%` |

当前模型最擅长预测中间 C/D 区间，最不擅长识别真正的 A 档和 E 档。这会影响两个下游用途：

- 如果任务是“选出最值得投放/复用的视频”，模型排序还能用，但需要 Top-K 策略而不是固定分数阈值
- 如果任务是“给视频打 A-E 等级”，当前输出还需要单独做校准或训练等级模型

### 4.4 Top-K 排序优于固定阈值判断

如果用 benchmark 的真实 `top30` 分数阈值 `0.6148` 直接判断：

| 口径 | Precision | Recall | F1 | 预测正例数 | 真实正例数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定阈值 Top30 | `0.6467` | `0.2694` | `0.3804` | `150` | `360` |
| 固定阈值 Top10 | `0.5714` | `0.0333` | `0.0630` | `7` | `120` |

固定阈值召回很低，主要原因不是完全排不准，而是预测分数整体被压缩，导致很少样本超过真实阈值。

如果按预测分数排序直接取前 K：

| 口径 | K | 命中数 | Precision | Recall | Lift | 入选样本真实均分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 预测前 30% 命中真实 Top30 | `360` | `182` | `0.5056` | `0.5056` | `1.69x` | `0.6189` |
| 预测前 10% 命中真实 Top10 | `120` | `45` | `0.3750` | `0.3750` | `3.75x` | `0.7309` |
| 预测前 5% 命中真实 Top10 | `60` | `29` | `0.4833` | `0.2417` | `4.83x` | `0.7794` |

因此更推荐把当前模型先用于“候选排序/Top-K 筛选”，而不是直接把分数解释为校准后的绝对表现概率。

### 4.5 Fold 稳定性

| Fold | 样本数 | MAE | Spearman | Pearson |
| --- | ---: | ---: | ---: | ---: |
| 1 | `240` | `0.1719` | `0.4540` | `0.4834` |
| 2 | `240` | `0.1718` | `0.5234` | `0.5365` |
| 3 | `240` | `0.1690` | `0.4889` | `0.5095` |
| 4 | `240` | `0.1680` | `0.5243` | `0.5376` |
| 5 | `240` | `0.1940` | `0.3331` | `0.3743` |

第 5 折明显更弱，说明当前数据划分下存在一定样本组成差异。后续建议增加 repeated CV 或固定一份正式 dev/test split，避免单次 5-fold 过度影响结论。

## 5. 特征重要性与 Ablation

### 5.1 输入来源重要性

按 LightGBM gain 汇总，当前各输入来源贡献如下：

| 来源 | Gain 占比 | 结论 |
| --- | ---: | --- |
| `publish_context` | `30.2%` | 当前最强来源 |
| `visual_features` | `28.3%` | 有大量局部 split，但单独建模较弱 |
| `text_language` | `23.7%` | 稳定提供内容语言信号 |
| `cross_modal_consistency` | `10.5%` | 有信号，但当前加入全量模型不一定增益 |
| `audio_speech` | `7.4%` | 单独较弱，更多是辅助特征 |

Top 20 重要特征：

| 排名 | 特征 | mean_gain |
| ---: | --- | ---: |
| 1 | `publish_context__video_sales_power` | `78.42` |
| 2 | `publish_context__product_category` | `37.37` |
| 3 | `publish_context__followers_total` | `22.35` |
| 4 | `publish_context__goods_count` | `21.32` |
| 5 | `visual_features__face_area` | `18.91` |
| 6 | `publish_context__influencer_type` | `18.71` |
| 7 | `text_language__title_score` | `17.15` |
| 8 | `publish_context__product_views_30d_w` | `16.59` |
| 9 | `visual_features__face_quality` | `16.42` |
| 10 | `audio_speech__video_duration_s` | `14.91` |
| 11 | `text_language__info_value_1` | `14.63` |
| 12 | `audio_speech__speech_rate` | `14.47` |
| 13 | `cross_modal_consistency__title_text_length_ratio` | `14.47` |
| 14 | `text_language__text_first_person` | `14.25` |
| 15 | `text_language__text_score` | `13.37` |
| 16 | `text_language__info_value_3` | `13.19` |
| 17 | `visual_features__color_value` | `13.03` |
| 18 | `visual_features__smile_rate` | `12.87` |
| 19 | `cross_modal_consistency__title_product_bigram_jaccard` | `12.65` |
| 20 | `text_language__title_entropy` | `11.97` |

需要特别注意：`video_sales_power` 来自原始字段“视频带货力”，不是 `final_score` 标签本身，但它语义上非常接近商业表现，因此需要在 benchmark 中明确它到底属于“可用先验上下文”还是“后验/近标签变量”。

### 5.2 Ablation 结果

以下结果都按同样的 5-fold OOF 方式临时复跑，用于判断特征来源贡献。

| 变体 | 特征数 | Spearman | MAE | AUC Top30 | Pairwise Acc | 观察 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 去掉 `cross_modal_consistency` | `98` | `0.4728` | `0.1741` | `0.7254` | `0.6684` | 略优于全量，说明部分一致性特征可能有噪声 |
| 去掉 `visual_features` | `101` | `0.4675` | `0.1741` | `0.7263` | `0.6612` | 略优于全量，但 pairwise 略降 |
| 全量特征 | `128` | `0.4641` | `0.1749` | `0.7216` | `0.6698` | 当前正式 baseline |
| `publish_context` only | `18` | `0.4587` | `0.1761` | `0.7350` | `0.6616` | 单独就接近全量，说明上下文很强 |
| 去掉 `text_language` | `90` | `0.4519` | `0.1762` | `0.7209` | `0.6590` | 文本结构化特征有稳定贡献 |
| 去掉 `video_sales_power` | `127` | `0.4295` | `0.1795` | `0.7069` | `0.6372` | 单字段影响很大，需重点审查 |
| 去掉 `video_sales_power`、`commerce_reputation`、`product_views_30d_w`、`goods_count` | `124` | `0.3779` | `0.1850` | `0.6849` | `0.6326` | 商品/带货上下文贡献显著 |
| 去掉全部 `publish_context` | `110` | `0.3525` | `0.1872` | `0.6678` | `0.6096` | 纯内容结构化 baseline 明显弱一档 |
| `text_language` only | `38` | `0.3373` | `0.1880` | `0.6580` | `0.6098` | 文本结构化特征是最强内容单模态 |
| `cross_modal_consistency` only | `30` | `0.2515` | `0.1945` | `0.6294` | `0.5860` | 单独较弱 |
| `visual_features` only | `27` | `0.2449` | `0.1952` | `0.6284` | `0.5742` | 单独较弱 |
| `audio_speech` only | `15` | `0.2334` | `0.1949` | `0.6110` | `0.5814` | 单独较弱 |

关键结论：

- `publish_context` 是当前模型主要性能来源
- 纯内容结构化特征仍有可用信号，但与上下文特征差距明显
- `video_sales_power` 对结果影响过大，需要作为 benchmark 风险点单独标注
- 全量特征不是每项指标都最优，说明可以通过特征选择继续提分

## 6. 如何提分

### 6.1 最高优先级：做分数校准

当前最明显的问题不是排序完全失败，而是预测分数方差不足。建议先做以下校准实验：

| 方法 | 目标 | 预期改善 |
| --- | --- | --- |
| OOF quantile mapping | 把预测分数分布映射到训练标签分布 | 改善 `macro_f1`、固定阈值 Top30/Top10 recall |
| Isotonic calibration | 学习单调分数校准函数 | 保留排序，同时修正分数刻度 |
| Fold-wise threshold calibration | 每折用训练折阈值派生 grade/top 标签 | 减少全局阈值带来的分布偏差 |
| Per-segment calibration | 按 `product_bucket` 或 `fan_segment` 校准 | 缓解品类/达人规模差异 |

优先理由：当前 AUC 和 Spearman 已经有信号，但固定阈值召回很低，校准可能用较小改动带来较大指标提升。

### 6.2 增加 Top30 分类器和等级模型

当前只训练一个 `final_score` 回归器，再从分数派生 grade。建议增加多任务 baseline：

| 模型 | 目标 |
| --- | --- |
| LightGBMRegressor | 继续预测 `final_score` |
| LightGBMClassifier | 直接预测 `top30_label` / `top10_label` |
| Ordinal classifier | 直接预测 A-E 五档等级 |
| Ensemble rank score | 融合回归分数、Top30 概率、等级期望值 |

这样可以避免一个回归分数同时承担“绝对分数、Top-K、等级”三类目标。

### 6.3 用排序目标优化 pairwise / Top-K

benchmark 里已经有 `pairs_v1.jsonl`，而当前 LightGBM 只做回归。建议补一个 `LightGBMRanker` 或 pairwise baseline：

| 方案 | 适用目标 |
| --- | --- |
| `LGBMRanker(objective="lambdarank")` | 提升 `pairwise_accuracy`、NDCG、Top-K |
| 按 `product_bucket` 分组 rank | 符合当前 pairwise 样本生成逻辑 |
| 回归 + 排序 ensemble | 同时兼顾 `MAE` 和排序指标 |

如果业务重点是“从 1200 条里挑出高潜视频”，排序模型可能比单纯回归更合理。

### 6.4 做特征选择与来源分轨

Ablation 显示去掉 `cross_modal_consistency` 或 `visual_features` 后，部分指标略有提升。建议下一轮不要盲目堆全量特征，而是做以下实验：

| 实验 | 目的 |
| --- | --- |
| Top-N gain 特征 | 判断前 30/50/80 个特征是否更稳 |
| Remove noisy blocks | 检查 `cross_modal_consistency`、视觉高缺失字段是否引入噪声 |
| Grouped permutation importance | 比 gain 更可靠地判断来源贡献 |
| SHAP 分析 | 看高重要性特征是否方向合理 |

### 6.5 引入更强文本特征

当前文本只使用结构化统计和关键词特征，没有使用原始文本语义。后续可以加入：

| 特征 | 说明 |
| --- | --- |
| TF-IDF ngram | 低成本传统文本 baseline |
| sentence embedding | 标题、视频文本、商品标题的语义向量 |
| 商品词/功效词/场景词词典 | 比通用 CTA 词更贴近营销内容 |
| 文案结构特征 | 痛点、卖点、证据、价格、行动号召的顺序 |

这对“content-only”赛道尤其重要，因为当前 content-only 的 Spearman 只有 `0.3525`。

### 6.6 加强上下文交互特征

当前强信号来自 `publish_context`，但交互还比较少。可以增加：

| 交互 | 直觉 |
| --- | --- |
| `product_bucket × fan_segment` | 不同品类对达人规模敏感度不同 |
| `product_category × title/text score` | 同样的文案风格在不同品类效果不同 |
| `duration_bucket × speech_rate` | 长短视频对应不同节奏要求 |
| `followers_total × engagement style` | 头部/尾部达人互动结构不同 |

实现时要用 OOF target encoding 或 category encoding，避免编码泄漏。

### 6.7 置信度需要正式校准

当前 `confidence` 是“预测分数距训练折 Top30 阈值的归一化距离”，属于启发式，不是概率。建议改成：

| 方案 | 目标 |
| --- | --- |
| 用 OOF 预测训练 calibration model | 让 confidence 更接近“Top30 判断正确概率” |
| 分 bucket 计算 reliability curve | 检查高置信区是否真的更准 |
| 输出 calibrated confidence 和 raw confidence | 区分模型原始距离与校准概率 |

这会直接改善 `ECE` 和综合 `agent_score` 的可信度。

## 7. Benchmark 改进建议

### 7.1 明确定义多个赛道

当前 `publish_context` 里有强上下文字段，尤其 `video_sales_power`。如果所有方法都允许使用这些字段，传统 ML baseline 会很强，但 VLM / Agent 的比较可能不公平。建议把 benchmark 拆成多个 track：

| Track | 允许输入 | 目的 |
| --- | --- | --- |
| Content-only | 视频、标题、文案、商品文本、内容结构化特征 | 测内容理解能力 |
| Pre-publish context | 内容 + 达人/粉丝/品类/发布时间等先验上下文 | 测真实投放前预测 |
| Commerce context | 加入商品热度、带货口碑等商业上下文 | 测商业先验预测 |
| Sales-aware / Audit | 允许销售截图或疑似后验字段 | 用于上限分析和数据核查，不与纯内容模型直接比较 |

这样 LightGBM、VLM、multi-agent 才能在同一输入权限下公平比较。

### 7.2 建立字段泄漏等级表

建议给每个字段标注风险等级：

| 等级 | 示例 | 处理建议 |
| --- | --- | --- |
| Label | `likes`、`comments`、`shares`、`collects`、`final_score` | 永远禁止作为输入 |
| Near-label | `video_sales_power`、可能来自成交/表现统计的字段 | 默认禁用，除非进入 Commerce/Sales-aware track |
| Context | `followers_total`、`product_category`、`publish_hour` | Pre-publish track 可用 |
| Content | 文本、视觉、语音、跨模态一致性 | Content-only track 可用 |
| Asset index | 文件路径、截图路径 | 只用于定位，不作为模型特征 |

当前 `video_sales_power` 是最需要优先审查的字段，因为去掉它后 Spearman 从 `0.4641` 降到 `0.4295`，去掉一组商业强上下文字段后降到 `0.3779`。

### 7.3 调整评估指标，区分排序与校准

当前指标已经覆盖 Spearman、MAE、AUC、Pairwise、ECE，但建议增加以下指标：

| 新指标 | 为什么需要 |
| --- | --- |
| `precision@top30%` | 贴近“筛选前 30% 候选”的业务用法 |
| `recall@top30%` | 判断高潜视频覆盖率 |
| `lift@top10%` / `lift@top5%` | 衡量模型选尖子样本的价值 |
| `ndcg@k` | 更适合排序任务 |
| `quantile_mae` | 检查 A/E 两端误差 |
| `ordinal_mae` 或 weighted kappa | 比 Macro-F1 更适合有序等级 |

尤其是 Top-K 指标很重要，因为当前模型 AUC 不差，但固定分数阈值召回很低；如果只看阈值分类，会低估它的排序价值。

### 7.4 增加固定 dev/test 与 group split

当前使用 5-fold OOF 可以做 baseline，但 benchmark 最好有稳定排行榜口径：

| 划分 | 目的 |
| --- | --- |
| Random stratified dev/test | 保证基础可比性 |
| Product-bucket split | 检查跨品类泛化 |
| Author/group split | 避免同作者风格泄漏 |
| Time split | 更贴近真实上线预测 |
| Repeated CV | 降低单次 fold 偶然性 |

Fold 5 明显低于其他折，说明后续需要更关注 split 稳定性。

### 7.5 建立 baseline 套件，而不是单一 baseline

建议 benchmark 固定发布以下 baseline：

| Baseline | 作用 |
| --- | --- |
| 常数均值 / 中位数 | 最低参照线 |
| Product mean / Fan-segment mean | 上下文弱基线 |
| Content-only LightGBM | 传统 ML 内容基线 |
| Context-aware LightGBM | 当前主 baseline |
| No-near-label LightGBM | 去掉高风险字段后的安全基线 |
| LightGBMRanker | 排序基线 |
| VLM zero-shot / few-shot | 多模态模型基线 |
| Multi-agent | 复杂推理基线 |

这样每一类方法都有明确对照，后续提分也更容易定位来自哪里。

### 7.6 输出自动化误差分析报告

建议每次 baseline 运行后自动生成：

| 报告 | 内容 |
| --- | --- |
| `metrics_by_fold.json` | 每折 Spearman、MAE、AUC |
| `metrics_by_grade.json` | A-E 每档误差、偏差、召回 |
| `metrics_by_product_bucket.json` | 品类分组表现 |
| `metrics_by_fan_segment.json` | 达人规模分组表现 |
| `topk_report.json` | Precision@K、Recall@K、Lift@K |
| `calibration_report.json` | reliability curve、ECE、Brier |

这会让 baseline 从“跑出一个分数”升级为“能定位问题的 benchmark 工具”。

## 8. 下一轮推荐实验优先级

| 优先级 | 实验 | 预期收益 |
| ---: | --- | --- |
| 1 | OOF quantile mapping / isotonic calibration | 修复分数压缩，提高 grade 和固定阈值 Top-K |
| 2 | 增加 Top30/Top10 LightGBMClassifier | 直接优化筛选任务 |
| 3 | 跑 `no_video_sales_power` 与 `no-near-label` 正式安全 baseline | 明确 benchmark 公平口径 |
| 4 | LightGBMRanker + pairwise 样本 | 提升排序与 Top-K |
| 5 | Top-N 特征选择和 SHAP 审查 | 去掉噪声特征，提高稳定性 |
| 6 | 文本 TF-IDF / embedding baseline | 提升 content-only 赛道 |
| 7 | repeated CV / 固定 test split | 提升指标可信度 |

建议下一步先做两个最小闭环：

| 闭环 | 内容 |
| --- | --- |
| Baseline V1.1 | 当前 LightGBM + 分数校准 + Top-K 报告 |
| Baseline V1-safe | 去掉 `video_sales_power` 等高风险商业字段后的安全 LightGBM |

这样可以同时回答两个问题：

- 如果追求当前 benchmark 分数，传统 ML baseline 能提高到哪里？
- 如果严格避免近标签/后验字段，纯内容与安全上下文 baseline 到底有多强？
