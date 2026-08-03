# -*- coding: utf-8 -*-
"""P2 板块相对情感反转 - 样本外验证 (2026-07-17)

步骤:
A. 样本内 sanity check: 用原 sector_sentiment_factors.csv 复算 IC (应≈-0.036)
B. 检查 super_dataset 内置 news_major_impact 的覆盖(可能更大的样本外源)
C. 样本外验证: all_news_stocks(LLM impact, 2021-05~2026-06)
   - 行业映射 stock_industry_map_cached.parquet
   - 远期收益: super_dataset 价格面板, T+1收盘买入, 持有1/2/5天
   - rel_sentiment = 个股impact - 同行业当日其他股票impact均值(leave-one-out)
   - 每日截面 Spearman IC, 均值/t值; 五分位多空
   - 窗口划分: 样本内 20241011~20250221 / 样本外前 20210531~20241010 / 样本外后 20250222~20260401
"""
import pandas as pd
import numpy as np
import duckdb
from math import erf, sqrt

root = r"C:\Users\liuqi\quant_system_v2"
con = duckdb.connect()

def pval(t):
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))

def spearman_ic_daily(df, factor="rel_sentiment", ret="ret_h2", min_n=5):
    """每日截面 Spearman IC (无scipy: 先rank再Pearson), 返回 (mean_ic, t, n_days)"""
    ics = []
    for d, g in df.groupby("date"):
        if len(g) < min_n:
            continue
        if g[factor].nunique() < 3 or g[ret].nunique() < 3:
            continue
        ics.append(g[factor].rank().corr(g[ret].rank()))  # rank后Pearson = Spearman
    ics = pd.Series(ics).dropna()
    if len(ics) < 5:
        return np.nan, np.nan, len(ics)
    t = ics.mean() / (ics.std(ddof=1) / sqrt(len(ics)))
    return ics.mean(), t, len(ics)

def quintile_spread(df, factor="rel_sentiment", ret="ret_h2"):
    """五分位组合收益 (样本内全部股票日 pooled)"""
    try:
        df = df.copy()
        df["q"] = pd.qcut(df[factor].rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"])
        piv = df.groupby("q", observed=True)[ret].mean() * 100
        return piv
    except Exception:
        return None

# ---------- A. 样本内 sanity ----------
print("=" * 60)
print("A. 样本内 sanity check (原关键词数据集)")
sf = pd.read_csv(f"{root}/sector_sentiment_factors.csv")
ic, t, n = spearman_ic_daily(sf.rename(columns={"news_date": "date"}), "rel_sentiment", "ret_2d")
print(f"   rel_sentiment vs ret_2d: IC={ic:+.4f}, t={t:+.2f}, n_days={n}  (报告值: IC=-0.036, t=-2.98)")
piv = quintile_spread(sf.rename(columns={"news_date": "date"}), "rel_sentiment", "ret_2d")
if piv is not None:
    print(f"   五分位收益(%): {dict(piv.round(3))}, 多空(Q1-Q5)={piv['Q1']-piv['Q5']:+.3f}%")

# ---------- B. super_dataset news_major_impact 覆盖 ----------
print("\n" + "=" * 60)
print("B. super_dataset 内置 news impact 覆盖检查")
cov = con.execute(f"""
    SELECT SUBSTR(CAST(trade_date AS VARCHAR),1,4) AS yr,
           COUNT(*) AS total,
           SUM(CASE WHEN news_major_impact IS NOT NULL THEN 1 ELSE 0 END) AS nn,
           SUM(CASE WHEN news_major_impact <> 0 THEN 1 ELSE 0 END) AS nz
    FROM read_parquet('{root}/daily-pro-t1/data/super_dataset.parquet')
    GROUP BY 1 ORDER BY 1
""").df()
print(cov.to_string(index=False))
print(con.execute(f"""
    SELECT news_major_impact, COUNT(*) c FROM read_parquet('{root}/daily-pro-t1/data/super_dataset.parquet')
    WHERE news_major_impact IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT 10
""").df().to_string(index=False))
