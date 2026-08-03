# -*- coding: utf-8 -*-
"""P2/P3 样本外终验 (2026-07-17)

A. P2 时间稳定性: 关键词样本内数据分前后两半, 各算 rel_sentiment IC
B. LLM 数据集 (2021-05~2026-06, 5,761股-日) 市场相对情感反转:
   rel_mkt = 个股impact - 当日其他股票impact均值 (leave-one-out)
   分窗口: pre(~20241010) / in(样本内) / post(20250222~)
C. P3 残差口径: LLM 负面(impact<=2) + 不跌(D1~D3 > -2%) → D3收盘入场 → T+5/T+10
"""
import pandas as pd
import numpy as np
import duckdb
from math import erf, sqrt

root = r"C:\Users\liuqi\quant_system_v2"
con = duckdb.connect()

def pval(t):
    return 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))

def ic_daily(d, factor, ret, min_n=5):
    ics = []
    for _, g in d.groupby("date"):
        if len(g) < min_n or g[factor].nunique() < 3 or g[ret].nunique() < 3:
            continue
        ics.append(g[factor].rank().corr(g[ret].rank()))
    ics = pd.Series(ics).dropna()
    if len(ics) < 5:
        return np.nan, np.nan, len(ics)
    return ics.mean(), ics.mean() / (ics.std(ddof=1) / sqrt(len(ics))), len(ics)

def sig(p):
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))

def quintiles(d, factor, ret):
    x = d.copy()
    x["q"] = pd.qcut(x[factor].rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"])
    return x.groupby("q", observed=True)[ret].mean()

# ========== A. P2 时间稳定性 (关键词样本内分半) ==========
print("=" * 64)
print("A. P2 时间稳定性 (原关键词数据, 样本内分前后两半)")
sf = pd.read_csv(f"{root}/sector_sentiment_factors.csv").rename(columns={"news_date": "date"})
lo, hi = sf["ret_2d"].quantile([0.01, 0.99])
sf["ret_2d_w"] = sf["ret_2d"].clip(lo, hi)
median_d = sf["date"].median()
for label, wd in [("前半", sf[sf["date"] <= median_d]), ("后半", sf[sf["date"] > median_d])]:
    ic, t, n = ic_daily(wd, "rel_sentiment", "ret_2d_w")
    p = pval(t) if not np.isnan(t) else np.nan
    print(f"  {label} ({wd['date'].min()}~{wd['date'].max()}): IC={ic:+.4f}, t={t:+.2f}, p={p:.3f} {sig(p)}, n_days={n}")

# ========== B. LLM 市场相对情感 ==========
print("\n" + "=" * 64)
print("B. LLM 数据集 市场相对情感反转 (2021-05~2026-06)")
news = pd.read_csv(f"{root}/all_news_stocks.csv").dropna(subset=["ts_code", "impact", "next_trade_date"])
news["date"] = news["date"].astype(int)
news["next_trade_date"] = news["next_trade_date"].astype(int)
news = news.groupby(["date", "next_trade_date", "ts_code"], as_index=False)["impact"].mean()
# leave-one-out 市场均值
day = news.groupby("date")["impact"]
news["mkt_sum"] = day.transform("sum")
news["mkt_cnt"] = day.transform("count")
news = news[news["mkt_cnt"] >= 2].copy()
news["rel_mkt"] = news["impact"] - (news["mkt_sum"] - news["impact"]) / (news["mkt_cnt"] - 1)
print(f"  股-日样本: {len(news)}")

px = con.execute(f"""
    WITH daily AS (
        SELECT ts_code, trade_date, AVG(entry_close) AS close
        FROM read_parquet('{root}/daily-pro-t1/data/super_dataset.parquet')
        WHERE entry_close > 0 GROUP BY 1, 2
    )
    SELECT ts_code, trade_date, close,
           LEAD(close,1) OVER w AS c1, LEAD(close,2) OVER w AS c2,
           LEAD(close,4) OVER w AS c4, LEAD(close,5) OVER w AS c5,
           LEAD(close,9) OVER w AS c9
    FROM daily WINDOW w AS (PARTITION BY ts_code ORDER BY trade_date)
""").df()
px["trade_date"] = px["trade_date"].astype(int)
df = news.merge(px, left_on=["ts_code", "next_trade_date"], right_on=["ts_code", "trade_date"], how="inner")
df["ret_h1"] = (df["c1"] / df["close"] - 1) * 100
df["ret_h2"] = (df["c2"] / df["close"] - 1) * 100
df["ret_h5"] = (df["c5"] / df["close"] - 1) * 100
df = df.dropna(subset=["ret_h2"])
for h in [1, 2, 5]:
    lo, hi = df[f"ret_h{h}"].quantile([0.01, 0.99])
    df[f"ret_h{h}"] = df[f"ret_h{h}"].clip(lo, hi)
print(f"  合并价格后: {len(df)} 行, {df['date'].nunique()} 天")

IN_S, IN_E = 20241011, 20250221
windows = {
    "pre 样本外前 (2021-05~2024-10)": df[df["date"] < IN_S],
    "in  样本内 (2024-10~2025-02)": df[(df["date"] >= IN_S) & (df["date"] <= IN_E)],
    "post 样本外后 (2025-02~2026-06)": df[df["date"] > IN_E],
    "all 全样本": df,
}
for wname, wd in windows.items():
    print(f"\n  [{wname}] n={len(wd)}, 天数={wd['date'].nunique()}")
    for factor, fl in [("rel_mkt", "rel_mkt(相对)"), ("impact", "impact(绝对)")]:
        for h in [1, 2, 5]:
            ic, t, n = ic_daily(wd, factor, f"ret_h{h}")
            if np.isnan(ic):
                print(f"    {fl:16s} vs h{h}: 天数不足(n={n})")
            else:
                p = pval(t)
                print(f"    {fl:16s} vs h{h}: IC={ic:+.4f}, t={t:+.2f}, p={p:.3f} {sig(p)}, n_days={n}")
    piv = quintiles(wd, "rel_mkt", "ret_h2")
    print(f"    rel_mkt 五分位 h2(%): " + ", ".join(f"{k}={v:+.3f}" for k, v in piv.items())
          + f"  多空Q1-Q5={piv['Q1']-piv['Q5']:+.3f}%")

# ========== C. P3 残差口径 (LLM 负面) ==========
print("\n" + "=" * 64)
print("C. P3 残差口径: LLM 负面(impact<=2) + 不跌 → D3收盘入场")
df["drop_1_3"] = (df["c2"] / df["close"] - 1) * 100       # D1收盘→D3收盘
df["r_3to5"] = (df["c4"] / df["c2"] - 1) * 100            # D3收盘→D5收盘
df["r_3to10"] = (df["c9"] / df["c2"] - 1) * 100           # D3收盘→D10收盘
neg = df[df["impact"] <= 2].dropna(subset=["drop_1_3", "r_3to5"])
print(f"  负面样本(impact<=2): {len(neg)}")
for wname, wd in [("pre", neg[neg["date"] < IN_S]), ("post", neg[neg["date"] > IN_E]), ("all", neg)]:
    g_sig = wd[wd["drop_1_3"] > -2]
    g_crash = wd[wd["drop_1_3"] < -5]
    if len(g_sig) < 10:
        print(f"  [{wname}] 样本不足 (信号组{len(g_sig)})")
        continue
    m5, m10 = g_sig["r_3to5"].mean(), g_sig["r_3to10"].mean()
    w5 = (g_sig["r_3to5"] > 0).mean() * 100
    mc5 = g_crash["r_3to5"].mean() if len(g_crash) else np.nan
    mc10 = g_crash["r_3to10"].mean() if len(g_crash) else np.nan
    # t检验: 信号组 r_3to5 vs 0
    t5 = m5 / (g_sig["r_3to5"].std(ddof=1) / sqrt(len(g_sig)))
    p5 = pval(t5)
    print(f"  [{wname}] 信号组n={len(g_sig)}: D3→D5 {m5:+.2f}% (胜率{w5:.0f}%, t={t5:.2f}, p={p5:.3f} {sig(p5)}), "
          f"D3→D10 {m10:+.2f}% | 大跌组n={len(g_crash)}: D3→D5 {mc5:+.2f}%, D3→D10 {mc10:+.2f}%")

df.to_csv(f"{root}/oos_llm_full_panel.csv", index=False)
print("\n已保存 oos_llm_full_panel.csv")
