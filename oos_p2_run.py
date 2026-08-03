# -*- coding: utf-8 -*-
"""P2 样本外验证主运行: LLM impact 数据集 + 行业映射 + 价格面板远期收益

设计:
- 新闻: all_news_stocks.csv (LLM impact, 2021-05~2026-06), 同股同日多条取均值
- 行业: stock_industry_map_cached.parquet
- rel_sentiment = 个股impact - 同行业当日其他股票impact均值 (leave-one-out, 行业内≥2只)
- 收益: super_dataset 价格面板, T+1(next_trade_date)收盘买入, 持有1/2/5天 (无前视)
- 收益在1%/99%分位winsorize (原报告五分位被极端值污染)
- 窗口: pre=~20241010, in=20241011~20250221, post=20250222~
"""
import pandas as pd
import numpy as np
import duckdb
from math import erf, sqrt

root = r"C:\Users\liuqi\quant_system_v2"
con = duckdb.connect()

def pval(t):
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))

# ---------- 1. 新闻 + 行业 + rel_sentiment ----------
news = pd.read_csv(f"{root}/all_news_stocks.csv")
news = news.dropna(subset=["ts_code", "impact", "next_trade_date"])
news["date"] = news["date"].astype(int)
news["next_trade_date"] = news["next_trade_date"].astype(int)
news = news.groupby(["date", "next_trade_date", "ts_code"], as_index=False)["impact"].mean()
print(f"新闻(股-日聚合): {len(news)} 行, {news['ts_code'].nunique()} 只股票, {news['date'].nunique()} 个日期")

ind = con.execute(f"SELECT ts_code, industry FROM read_parquet('{root}/stock_industry_map_cached.parquet')").df()
news = news.merge(ind, on="ts_code", how="inner")
print(f"映射行业后: {len(news)} 行, {news['industry'].nunique()} 个行业")

# leave-one-out 行业均值
g = news.groupby(["date", "industry"])["impact"]
news["ind_sum"] = g.transform("sum")
news["ind_cnt"] = g.transform("count")
news = news[news["ind_cnt"] >= 2].copy()
news["rel_sentiment"] = news["impact"] - (news["ind_sum"] - news["impact"]) / (news["ind_cnt"] - 1)
print(f"行业日内≥2只后: {len(news)} 行")

# ---------- 2. 价格面板远期收益 (duckdb 窗口) ----------
px = con.execute(f"""
    WITH daily AS (
        SELECT ts_code, trade_date, AVG(entry_close) AS close
        FROM read_parquet('{root}/daily-pro-t1/data/super_dataset.parquet')
        WHERE entry_close > 0
        GROUP BY 1, 2
    )
    SELECT ts_code, trade_date, close,
           LEAD(close, 1) OVER w AS c_p1,
           LEAD(close, 2) OVER w AS c_p2,
           LEAD(close, 5) OVER w AS c_p5
    FROM daily
    WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
""").df()
px["trade_date"] = px["trade_date"].astype(int)

df = news.merge(px, left_on=["ts_code", "next_trade_date"], right_on=["ts_code", "trade_date"], how="inner")
for h, c in [(1, "c_p1"), (2, "c_p2"), (5, "c_p5")]:
    df[f"ret_h{h}"] = (df[c] / df["close"] - 1) * 100  # %
df = df.dropna(subset=["ret_h2"])
print(f"合并价格后: {len(df)} 行")

# winsorize
for h in [1, 2, 5]:
    lo, hi = df[f"ret_h{h}"].quantile([0.01, 0.99])
    df[f"ret_h{h}"] = df[f"ret_h{h}"].clip(lo, hi)

# ---------- 3. 分窗口 IC 与五分位 ----------
IN_START, IN_END = 20241011, 20250221
windows = {
    "样本外-前 (~20241010)": df[df["date"] < IN_START],
    "样本内 (20241011~20250221)": df[(df["date"] >= IN_START) & (df["date"] <= IN_END)],
    "样本外-后 (20250222~)": df[df["date"] > IN_END],
    "全样本": df,
}

def spearman_ic_daily(d, factor, ret, min_n=5):
    ics = []
    for _, g in d.groupby("date"):
        if len(g) < min_n or g[factor].nunique() < 3 or g[ret].nunique() < 3:
            continue
        ics.append(g[factor].rank().corr(g[ret].rank()))
    ics = pd.Series(ics).dropna()
    if len(ics) < 5:
        return np.nan, np.nan, len(ics)
    t = ics.mean() / (ics.std(ddof=1) / sqrt(len(ics)))
    return ics.mean(), t, len(ics)

def quintiles(d, factor, ret):
    try:
        x = d.copy()
        x["q"] = pd.qcut(x[factor].rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"])
        return x.groupby("q", observed=True)[ret].mean()
    except Exception:
        return None

for wname, wd in windows.items():
    print("\n" + "=" * 62)
    print(f"[{wname}] 股票日样本={len(wd)}, 天数={wd['date'].nunique()}")
    for factor, flabel in [("rel_sentiment", "rel_sentiment(相对)"), ("impact", "impact(绝对)")]:
        for h in [1, 2, 5]:
            ic, t, n = spearman_ic_daily(wd, factor, f"ret_h{h}")
            if np.isnan(ic):
                print(f"  {flabel:22s} vs ret_h{h}: 天数不足(n={n})")
            else:
                sig = "***" if pval(t) < 0.01 else ("**" if pval(t) < 0.05 else ("*" if pval(t) < 0.1 else ""))
                print(f"  {flabel:22s} vs ret_h{h}: IC={ic:+.4f}, t={t:+.2f}, p={pval(t):.3f} {sig}, n_days={n}")
    piv = quintiles(wd, "rel_sentiment", "ret_h2")
    if piv is not None:
        print(f"  rel_sentiment 五分位 ret_h2(%): " +
              ", ".join(f"{k}={v:+.3f}" for k, v in piv.items()) +
              f"  多空Q1-Q5={piv['Q1']-piv['Q5']:+.3f}%")

df.to_csv(f"{root}/oos_p2_llm_panel.csv", index=False)
print("\n已保存: oos_p2_llm_panel.csv")
