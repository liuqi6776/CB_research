# -*- coding: utf-8 -*-
"""主线截面因子(study_007)五道关卡验证 (2026-07-17)

对 study_007 主力因子做独立复算 + 准入检查:
  ivol(40%) + ret_1m(35%) + roe(15%) + or_yoy(5%) + netprofit_yoy(5%)

方法:
- 数据: data_day1 日频行情(2019-11~2025-12) + fina_indicator_cache(ROE等, 按ann_date PIT对齐)
- 调仓日: 每月首个交易日 (2020-01~2025-12, 72个月)
- ret_1m: 过去20交易日累计收益, 取负(反转); ivol: 过去20日(个股日收益-等权市场日收益)std, 取负
- 远期收益: 未来20交易日累计收益(pct_chg连乘, 免复权问题), winsorize 1/99
- 检查: 全期IC/t, 分半稳定性, 训练期(2020-2022)vs测试期(2023-2025), 五分位
- 简化声明: 因子未做行业/市值中性化(原研究有), 未剔ST/停牌, ivol用beta=1近似
"""
import pandas as pd
import numpy as np
import duckdb
from math import sqrt
import sys, time
sys.path.insert(0, r"C:\Users\liuqi\quant_system_v2")
from utils.factor_checks import pval_norm, sig_stars, winsorize

t0 = time.time()
con = duckdb.connect()
ROOT = r"C:\Users\liuqi\quant_system_v2"

# ---------- 1. 行情面板 ----------
import glob, os
all_files = glob.glob(r"D:/iquant_data/data_v2/data_day1/*.parquet")
good_files = [f for f in all_files if os.path.getsize(f) > 1024]  # 剔除空/损坏文件
print(f"行情文件: {len(all_files)} 个, 有效 {len(good_files)} 个")
flist = ",".join(f"'{f}'" for f in good_files)
px = con.execute(f"""
    SELECT ts_code, CAST(trade_date AS INTEGER) AS trade_date, pct_chg
    FROM read_parquet([{flist}])
    WHERE CAST(trade_date AS INTEGER) >= 20191001 AND pct_chg IS NOT NULL
""").df()
px["trade_date"] = px["trade_date"].astype(int)
print(f"行情面板: {len(px)} 行, {px['ts_code'].nunique()} 只, "
      f"{px['trade_date'].min()}~{px['trade_date'].max()}, 耗时{time.time()-t0:.0f}s")

# 等权市场日收益
mkt = px.groupby("trade_date")["pct_chg"].mean().rename("mkt_ret")
px = px.merge(mkt, on="trade_date")
px["ex_ret"] = px["pct_chg"] - px["mkt_ret"]
px["r"] = px["pct_chg"] / 100.0

# ---------- 2. 滚动因子 ----------
def build_factors(g):
    g = g.sort_values("trade_date").reset_index(drop=True)
    g["ret_1m"] = (1 + g["r"]).rolling(20).apply(np.prod, raw=True) - 1
    g["ivol"] = g["ex_ret"].rolling(20).std(ddof=1)
    cum20 = (1 + g["r"]).rolling(20).apply(np.prod, raw=True)
    g["fwd_20"] = cum20.shift(-20) - 1  # 未来20交易日累计收益
    g["n_hist"] = range(len(g))
    return g

parts = []
for code, g in px.groupby("ts_code"):
    gg = build_factors(g)
    gg["ts_code"] = code
    parts.append(gg)
px = pd.concat(parts, ignore_index=True)
print(f"滚动因子完成, 耗时{time.time()-t0:.0f}s")

# 月末/月初调仓日: 每月首个交易日
cal = sorted(px["trade_date"].unique())
cal_s = pd.Series(cal)
month_first = cal_s.groupby(cal_s // 100).min().tolist()
month_first = [d for d in month_first if 20200101 <= d <= 20251231]
print(f"调仓月数: {len(month_first)}")

panel = px[px["trade_date"].isin(month_first)].copy()
panel = panel[(panel["n_hist"] >= 60)]  # 排除上市不足60交易日
panel = panel.dropna(subset=["ret_1m", "ivol", "fwd_20"])
panel["f_rev"] = -panel["ret_1m"]     # 反转因子(取负)
panel["f_ivol"] = -panel["ivol"]      # 低波动因子(取负)
panel["fwd_20"] = panel["fwd_20"] * 100  # %
print(f"月度面板: {len(panel)} 股-月")

# ---------- 3. 财务因子 PIT ----------
fin = con.execute("""
    SELECT ts_code, ann_date, roe, or_yoy, netprofit_yoy
    FROM read_parquet('D:/iquant_data/data_v2/fundamental1/fina_indicator_cache.parquet')
    WHERE ann_date IS NOT NULL
""").df()
fin["ann_date"] = fin["ann_date"].astype(str).str.replace("-", "").astype(int)
fin = fin.dropna(subset=["ann_date"]).sort_values("ann_date")
panel = panel.sort_values("trade_date")
panel = pd.merge_asof(panel, fin, left_on="trade_date", right_on="ann_date",
                      by="ts_code", direction="backward")
print(f"PIT财务合并后: {len(panel)} 行 (roe非空 {panel['roe'].notna().sum()}), 耗时{time.time()-t0:.0f}s")

# ---------- 4. 组合因子 (winsorize+z, 等权重按研究权重) ----------
for c in ["f_rev", "f_ivol", "roe", "or_yoy", "netprofit_yoy"]:
    panel[c] = panel.groupby("trade_date")[c].transform(lambda s: winsorize(s.dropna()).reindex(s.index))
    panel[c + "_z"] = panel.groupby("trade_date")[c].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=1) + 1e-12))
comp = panel.dropna(subset=["f_rev_z", "f_ivol_z", "roe_z", "or_yoy_z", "netprofit_yoy_z"]).copy()
comp["composite"] = (0.40 * comp["f_ivol_z"] + 0.35 * comp["f_rev_z"] + 0.15 * comp["roe_z"]
                     + 0.05 * comp["or_yoy_z"] + 0.05 * comp["netprofit_yoy_z"])
print(f"组合因子面板(五因子齐全): {len(comp)} 股-月")

# ---------- 5. 准入检查 ----------
def ic_monthly(d, factor, ret="fwd_20", min_n=100):
    ics = {}
    for dt, g in d.groupby("trade_date"):
        if len(g) < min_n or g[factor].nunique() < 10:
            continue
        ics[dt] = g[factor].rank().corr(g[ret].rank())
    return pd.Series(ics).dropna()

def report(name, ics):
    if len(ics) < 5:
        print(f"  {name}: 月数不足({len(ics)})")
        return
    t = ics.mean() / (ics.std(ddof=1) / sqrt(len(ics)))
    p = pval_norm(t)
    print(f"  {name}: IC={ics.mean():+.4f}, t={t:+.2f}, p={p:.4f} {sig_stars(p)}, "
          f"IC正率={(ics > 0).mean() * 100:.0f}%, n={len(ics)}")

def gates(label, d, factor):
    print(f"\n[{label}]")
    ics = ic_monthly(d, factor)
    report(f"全期(2020-2025)", ics)
    med = d["trade_date"].median()
    report(f"前半", ic_monthly(d[d["trade_date"] <= med], factor))
    report(f"后半", ic_monthly(d[d["trade_date"] > med], factor))
    report(f"训练期2020-2022", ic_monthly(d[d["trade_date"] <= 20221231], factor))
    report(f"测试期2023-2025", ic_monthly(d[d["trade_date"] >= 20230101], factor))
    # 五分位
    x = d.dropna(subset=[factor, "fwd_20"]).copy()
    x["q"] = pd.qcut(x[factor].rank(method="first"), 5, labels=["Q1","Q2","Q3","Q4","Q5"])
    piv = x.groupby("q", observed=True)["fwd_20"].mean()
    print(f"  五分位月均fwd20(%): " + ", ".join(f"{k}={v:+.2f}" for k, v in piv.items())
          + f"  多空Q1-Q5={piv['Q1'] - piv['Q5']:+.2f}%")

# winsorize 远期收益
panel["fwd_20"] = panel.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))
comp["fwd_20"] = comp.groupby("trade_date")["fwd_20"].transform(lambda s: winsorize(s))

gates("ivol(低特质波动, 权重40%)", panel, "f_ivol")
gates("ret_1m反转(权重35%)", panel, "f_rev")
gates("roe(权重15%)", panel.dropna(subset=["roe"]), "roe")
gates("or_yoy(权重5%)", panel.dropna(subset=["or_yoy"]), "or_yoy")
gates("netprofit_yoy(权重5%)", panel.dropna(subset=["netprofit_yoy"]), "netprofit_yoy")
gates("组合因子(复算)", comp, "composite")

comp.to_csv(f"{ROOT}/mainline_factor_panel.csv", index=False)
print(f"\n已保存 mainline_factor_panel.csv, 总耗时{time.time()-t0:.0f}s")
