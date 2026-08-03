# -*- coding: utf-8 -*-
"""verify_panel.py — 面板验证与统计 / Panel verification & stats for DATA_REPORT.md"""
import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
PANEL = os.path.join(BASE, "data", "panel.parquet")
FUNDA = os.path.join(BASE, "data", "funda_pit.parquet")

df = pd.read_parquet(PANEL)
print("=== 基础统计 / basic ===")
print("rows:", len(df), "| stocks:", df.ts_code.nunique(),
      "| dates:", df.trade_date.min(), "~", df.trade_date.max(),
      "| n_dates:", df.trade_date.nunique())

print("\n=== 缺失统计 / missing counts ===")
print(df.isna().sum().to_string())

print("\n=== 退市股（面板有数据但 industry=UNKNOWN）/ delisted ===")
unk = df[df.industry == "UNKNOWN"]
unk_stocks = unk.groupby("ts_code")["trade_date"].agg(["min", "max", "count"])
print("UNKNOWN stocks:", len(unk_stocks))
d24 = unk_stocks[(unk_stocks["max"] >= "20240101") & (unk_stocks["max"] <= "20241231")]
print("2024年退市 / delisted in 2024:", len(d24))
print(d24.head(10).to_string())

print("\n=== is_st / limit 统计 ===")
print("is_st rows:", int(df.is_st.sum()),
      "| limit_up rows:", int(df.limit_up.sum()),
      "| limit_down rows:", int(df.limit_down.sum()))

print("\n=== 除权除息证据 / ex-dividend evidence ===")
# 若 daily_ret 来自 pct_chg（已复权），则 close/pre_close-1 与 daily_ret 在除息日会明显偏离
# If daily_ret came from raw close/pre_close it would equal close/pre_close-1 everywhere.
raw = df.close / df.pre_close - 1
diff = (raw - df.daily_ret).abs()
ex = df[diff > 0.02].copy()
ex["raw_ret"] = raw[diff > 0.02]
print("除权息日数（raw与复权收益偏差>2%）:", len(ex))
cols = ["ts_code", "trade_date", "close", "pre_close", "daily_ret", "raw_ret"]
print(ex[cols].head(8).to_string())

print("\n=== 抽检股票 / sample stocks ===")
# 1) 正常主板股 / normal main-board stock
for code in ["600976.SH", "000001.SZ"]:
    s = df[df.ts_code == code].sort_values("trade_date")
    print(f"\n-- {code} ({s.name.iloc[0]}, {s.industry.iloc[0]}, list={s.list_date.iloc[0]}) rows={len(s)}")
    print(s[["trade_date", "close", "pre_close", "daily_ret", "circ_mv", "pe"]].head(3).to_string())

# 2) 一只2024年退市股 / one 2024-delisted stock
if len(d24):
    code = d24.index[0]
    s = df[df.ts_code == code].sort_values("trade_date")
    print(f"\n-- 退市股 {code} industry={s.industry.iloc[0]} name='{s.name.iloc[0]}' rows={len(s)}")
    print("最后5行 / last 5 rows:")
    print(s[["trade_date", "close", "pre_close", "daily_ret", "limit_down"]].tail(5).to_string())

# 3) 创业板 300 股在 2020-08-24 前后的涨跌停阈值 / ChiNext limit threshold across 2020-08-24
cyb = df[df.ts_code.str.startswith("300")]
before = cyb[(cyb.trade_date < "20200824") & (cyb.daily_ret > 0.12)]
after = cyb[(cyb.trade_date >= "20200824") & (cyb.daily_ret > 0.12)]
print("\n300xxx 涨超12%的天数 2020-08-24前(limit_up应全False):",
      len(before), "其中limit_up=True:", int(before.limit_up.sum()))
print("300xxx 涨超12%的天数 2020-08-24后(limit_up应全False除非>=19.8%):",
      len(after), "其中limit_up=True:", int(after.limit_up.sum()),
      "最小limit_up涨幅:", after[after.limit_up].daily_ret.min())

print("\n=== mkt_ret 检查 ===")
chk = df.groupby("trade_date")["daily_ret"].mean()
m = df.drop_duplicates("trade_date").set_index("trade_date")["mkt_ret"]
print("mkt_ret 与等权均值最大偏差:", float((chk - m).abs().max()))

print("\n=== funda_pit 检查 ===")
fu = pd.read_parquet(FUNDA)
print("rows:", len(fu), "| stocks:", fu.ts_code.nunique(),
      "| dates:", fu.trade_date.min(), "~", fu.trade_date.max())
print(fu.isna().sum().to_string())
# PIT 正确性抽查：一只股票的 roe 变化应只发生在公告日之后 / PIT sanity check
one = fu[fu.ts_code == fu.ts_code.iloc[0]].sort_values("trade_date")
chg = one[one.roe.ne(one.roe.shift())]
print("\n示例股票", one.ts_code.iloc[0], "roe 取值变化点:")
print(chg.head(5).to_string())
