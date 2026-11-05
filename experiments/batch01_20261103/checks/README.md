# G2 / H4 检查

G2：当前因子值头已是 Linear → LayerNorm → GELU → Linear，跳过新增训练。

H4：未发现独立 ST 预过滤；保留原有历史股票池。Qlib 对缺失/全 NaN close 报价视为停牌并拒绝不可成交订单。策略 only_tradable=False，forbid_all_trade_at_limit=True；传入 limit_threshold=None 会使用 Qlib 的区域默认值，本次初始化后的中国市场默认值为 0.095。这不等于按各股票 ST/板块实际涨跌停制度逐一建模。学习视图含 DropnaLabel/CSRankNorm，测试使用推理视图；没有按未来持有期事件筛测试股票。本批只记录现状，不修改数据或 TopkDropout 规则。
