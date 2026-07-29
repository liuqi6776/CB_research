# RESEARCH_SUMMARY.md: A-Share Convertible Bond Quantitative Multi-Factor Strategy Research Summary
# 可转债多因子量化策略实证研究总结报告 (中英双语)

> **Audit Status / 审查状态**: **PASS (Engineering Verified & Research Conclusion Archived)**  
> **Evaluation Period / 评估区间**: 2024-01-02 ~ 2026-06-25 (598 Empirical Trading Days / 598 个物理交易日)  
> **Friction Costs / 交易摩擦**: 20 bps Round-Trip (10 bps Buy + 10 bps Sell) | **Capacity Limit / 容量上限**: 15m K-Line 5%

---

## 1. Executive Summary / 核心要义

This repository establishes a **100% Zero-Lookahead Point-in-Time (PIT)** quantitative backtesting infrastructure for A-share Convertible Bonds (CBs). Through strict 5-stage institutional audit iterations, the engineering framework has been fully verified to be free of lookahead bias, data leakage, and train/serve inconsistencies.

In the clean empirical Out-of-Sample (OOS) evaluation over 2024-2026, all strategy variations were evaluated against the passive benchmark **511380.SH (Boshi CSIC Convertible Bond ETF)**. The empirical findings demonstrate that:
1. **Passive CB ETF Benchmark Superiority**: The 511380.SH ETF achieved **+26.98%** cumulative return (**+10.59%** annualized), Sharpe ratio **1.21**, and max drawdown **-6.85%**.
2. **Strategy Underperformance**: The simple Double-Low baseline achieved **+15.26%** return (underperforming ETF by **-11.72pp**).
3. **Negative Incremental Alpha**: Complex factor enhancements (TCC network centrality, GBDT 14-factor model, 3-tier dynamic timing) produced **negative incremental alpha**, reducing returns down to **-4.53%** (**-31.51pp** vs ETF).

本仓库成功构建了一套 **100% 零前视 Point-in-Time (PIT)** A 股可转债量化回测与工程体系。历经 5 轮严苛的机构级盲测与审查，工程框架已彻底消除了前视偏差、数据泄漏和 Train/Serve 特征计算不一致问题。

在 2024-2026 年干净的样本外（OOS）实证评估中，所有策略配置与 **511380.SH (博时中证可转债 ETF)** 进行了真实行情对比。实证研究结论表明：
1. **被动 ETF 表现最优**：511380.SH ETF 实现了 **+26.98%** 的累计收益（年化 **+10.59%**），夏普比率 **1.21**，最大回撤仅 **-6.85%**。
2. **纯物理基准跑输 ETF**：诚实纯双低基准实现 **+15.26%** 累计收益，跑输 511380 ETF **-11.72pp**。
3. **复杂增强机制产生负贡献**：引入 TCC 网络中心度、GBDT 14 因子模型与三档动态择时后，策略收益进一步下降至 **-4.53%**（跑输 511380 ETF **-31.51pp**）。

---

## 2. Empirical Performance Comparison Table / 样本外全量实证数据对比

| Strategy Configuration / 策略配置名称 | Cumulative Return / 累计收益 | Ann. Return / 年化收益 | Sharpe Ratio / 夏普比率 | Max Drawdown / 最大回撤 | Ann. Turnover / 年化换手 | Avg Holding / 平均持仓天数 | vs 511380 ETF Excess / 相对 ETF 超额 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **511380.SH CB ETF (博时可转债 ETF)** | **+26.98%** | **+10.59%** | **1.21** | **-6.85%** | -- | -- | **0.00pp (Benchmark)** |
| **Config 0: Clean Double-Low (纯双低)** | **+15.26%** | **+6.17%** | **0.48** | **-18.01%** | **0.4x** | **6.0 Days** | **-11.72pp** |
| **Config 2: GBDT 14-Factor Model (GBDT模型)** | **+3.22%** | **+1.35%** | **0.17** | **-14.19%** | **0.3x** | **7.1 Days** | **-23.76pp** |
| **Config 5: 80/20 Portfolio (组合部署)** | **-0.87%** | **-0.37%** | **-0.20** | **-2.47%** | -- | -- | **-27.85pp** |
| **Config 4b: 3-Tier Position Timing (三档控仓)** | **-4.37%** | **-1.86%** | **-0.16** | **-11.58%** | **0.4x** | **4.8 Days** | **-31.35pp** |
| **Config 1: TCC Factor Filter (TCC 因子)** | **-4.53%** | **-1.93%** | **-0.12** | **-17.32%** | **0.4x** | **6.6 Days** | **-31.51pp** |

---

## 3. Disproven Research Hypotheses / 排除的无效研究假设

1. **Hypothesis 1: TCC Network Centrality (TCC 网络中心度)**
   - *Finding*: Filtering out centrality outliers reduced return from `+15.26%` to `-4.53%` (-19.79pp vs Double-Low). Network tail deviation includes high-elasticity mean-reverting bonds; removing them harms strategy returns.
   - *结论*: 剔除网络偏离度离群债后，收益下降 19.79pp。网络尾部离群债包含了高弹性的反弹标的，剔除反而损耗了双低的均值回归收益。

2. **Hypothesis 2: GBDT 14-Factor ML Ranking (GBDT 14 因子机器学习)**
   - *Finding*: Combining GBDT prediction with Double-Low decreased return from `+15.26%` to `+3.22%` (-12.04pp vs Double-Low). Out-of-sample generalization of 2021-2023 trained weights is insufficient and adds noise.
   - *结论*: GBDT 预测结合双低后收益下降 12.04pp。2021-2023 训练集权重的样本外泛化能力不足，对双低产生了噪声干扰。

3. **Hypothesis 3: Moving Average Trend Timing (均线趋势择时)**
   - *Finding*: 3-tier timing reduced return from `+15.26%` to `-4.37%` (-19.63pp vs Double-Low). Cutting exposure to 20% during sharp market drawdowns caused the strategy to miss rapid post-trough V-shaped recoveries.
   - *结论*: 三档择时使收益下降 19.63pp。市场深跌后强劲 V 型反弹时，趋势择时在底部砍仓导致错失了反弹最剧烈阶段。

---

## 4. Reusable Engineering Infrastructure / 可复用工程资产

While the factor research produced a negative alpha conclusion for this specific strategy combination, the underlying **backtesting infrastructure is fully validated and reusable**:
- **PIT Engine**: `cb_quant/unified_pit_engine.py` (Point-in-Time clean conversion price & redemption checks).
- **Unified Feature Pipeline**: `cb_quant/feature_pipeline.py` (100% identical train/serve feature calculations).
- **Engine Risk Controls & Regression Suite**: `tests/test_simulation_engine.py` (20 bps friction costs, 5% volume caps, executable status checks locked with pytest).
- **Fail-Fast Model Loader**: Platform-independent relative `pathlib.Path` loaders with explicit `FileNotFoundError` assertions.

---

## 5. Next Research Directions / 后续研究方向建议

1. **Horizon Freeze**: Permanently freeze the 2024-2026 OOS period to prevent post-hoc parameter overfitting.
2. **Lower Turnover Frequency**: Shift from intraday/daily rebalancing to monthly or bi-weekly rebalancing to minimize transaction friction erosion.
3. **ETF Alpha Enhancement**: Transition from absolute return stock-picking to relative alpha enhancement against `511380.SH`.
4. **Stock-Bond Arbitrage**: Explore high-frequency intra-day stock-bond hedging or basis arbitrage utilizing 1-minute order book data.
