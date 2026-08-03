# -*- coding: utf-8 -*-
"""IM 贴水年化 - 正确算法: 单合约年化基差 (2026-07-17)

对每个季月合约(IM 3/6/9/12月), 在其存续期内采样:
  年化基差 = (期货价/现货价 - 1) × 365 / 到期剩余天数
多头滚仓可捕获收益 ≈ 平均年化基差(负基差=贴水=多头收益)
到期日 = 合约月份第三个周五 (中金所规则)
"""
import akshare as ak
import pandas as pd
import numpy as np
import time, datetime as dt

ROOT = r"C:\Users\liuqi\quant_system_v2"

def third_friday(year, month):
    d = dt.date(year, month, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)

# 现货
sa = pd.read_csv(f"{ROOT}/zz1000_spot_a.csv")
sb = pd.read_csv(f"{ROOT}/zz1000_spot_b.csv")
spot = pd.concat([sa, sb], ignore_index=True)
spot["date"] = pd.to_datetime(spot["time"])
spot = spot.drop_duplicates("date").set_index("date")["close"]

# 季月合约: 202303 ~ 202612
contracts = []
for y in range(2023, 2027):
    for m in [3, 6, 9, 12]:
        if (y, m) > (2026, 12):
            continue
        contracts.append((y, m, f"IM{str(y)[2:]}{m:02d}"))

records = []
for y, m, sym in contracts:
    try:
        df = ak.futures_zh_daily_sina(symbol=sym)
        if df is None or len(df) == 0:
            print(f"{sym}: 无数据")
            continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")["close"]
        expiry = pd.Timestamp(third_friday(y, m))
        # 采样窗口: 到期前 20~120 天 (流动性好且年化稳定)
        for d, f in df.items():
            dte = (expiry - d).days
            if 20 <= dte <= 120 and d in spot.index:
                s = spot[d]
                ann_basis = (f / s - 1) * 365 / dte
                records.append({"date": d, "contract": sym, "dte": dte,
                                "fut": f, "spot": s,
                                "basis_pct": (f / s - 1) * 100,
                                "ann_basis_pct": ann_basis * 100})
        print(f"{sym}: {len(df)} 天数据, 到期 {expiry.date()}")
        time.sleep(0.3)
    except Exception as e:
        print(f"{sym}: 失败 {e}")

r = pd.DataFrame(records)
print(f"\n采样点: {len(r)}")
if len(r):
    r["year"] = r["date"].dt.year
    print("\n分年平均年化基差(负=贴水, 多头滚仓可捕获 ≈ 该值×-1... 取绝对值即为多头收益):")
    g = r.groupby("year")["ann_basis_pct"].agg(["mean", "median", "count"])
    g.columns = ["均值%", "中位数%", "采样数"]
    print(g.round(2).to_string())
    print(f"\n全区间平均年化基差: {r['ann_basis_pct'].mean():+.2f}% (中位数 {r['ann_basis_pct'].median():+.2f}%)")
    print("解读: 年化基差为负(期货<现货=贴水), 多头买入贴水合约并持有收敛, 可捕获收益 ≈ -年化基差")
    r.to_csv(f"{ROOT}/im_ann_basis_samples.csv", index=False)
    print("已保存 im_ann_basis_samples.csv")
