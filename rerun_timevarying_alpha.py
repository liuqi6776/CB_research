# -*- coding: utf-8 -*-
"""时变 α(贴水驱动) 重跑中证1000增强回测 (2026-07-17)

alpha_t = -1 × 最近60个日历日 IM 季月合约年化基差均值 (无前视, 只用 t 日及之前采样)
对比 (2023-01 ~ 2026-07, 完整策略=增强+择时+卫星):
  A 常数11% (现状) | B 时变贴水驱动 | C 常数9.3%(全样本均值, 参考) | D 0%(无增强) | 纯指数
"""
import os, sys
import pandas as pd
import numpy as np

ROOT = r"C:\Users\liuqi\quant_system_v2"
sys.path.insert(0, f"{ROOT}/etf-valuation-strategy/scripts")
from step13_backtest_premium import load_premium_data, compute_metrics

# ---------- 1. 时变 α 序列 ----------
smp = pd.read_csv(f"{ROOT}/im_ann_basis_samples.csv")
smp["date"] = pd.to_datetime(smp["date"])
daily_basis = smp.groupby("date")["ann_basis_pct"].mean()          # 当日采样均值(负=贴水)
alpha_raw = -daily_basis / 100.0                                   # 多头可捕获=正
alpha_tv = alpha_raw.rolling(60, min_periods=20).mean()            # 60日历日滚动均值
print(f"时变α序列: {alpha_tv.index.min().date()} ~ {alpha_tv.index.max().date()}, "
      f"均值 {alpha_tv.mean():.2%}, 区间 [{alpha_tv.min():.2%}, {alpha_tv.max():.2%}]")

# ---------- 2. 带回测引擎 (step13 复刻 + α序列) ----------
def run_variant_tv(df_period, alpha_series=None, alpha_const=None,
                   use_timing=True, use_satellite=True,
                   val_coeff=0.6, W_nasdaq=0.15, W_gold=0.10,
                   dev_threshold=0.10, cap=1000000.0):
    if not use_satellite:
        W_nasdaq, W_gold = 0.0, 0.0
    df = df_period.copy().reset_index(drop=True)
    df['year_week'] = df['trade_date'].dt.strftime('%Y-%U')
    rebal = set(df.groupby('year_week')['trade_date'].first())

    if alpha_series is not None:
        a = alpha_series.reindex(df['trade_date']).ffill().fillna(0.0).values
        alpha_daily = (1.0 + a) ** (1.0 / 242.0) - 1.0
    else:
        alpha_daily = np.full(len(df), (1.0 + alpha_const) ** (1.0 / 242.0) - 1.0)
    df['ret_enh'] = (1.0 + df['ret_1000']) * (1.0 + alpha_daily) - 1.0

    v_zz, v_nq, v_gd, v_bd = 0.0, 0.0, 0.0, cap
    navs = []
    for idx, row in df.iterrows():
        dt = row['trade_date']
        if idx > 0:
            v_zz *= (1.0 + row['ret_enh']); v_nq *= (1.0 + row['ret_nasdaq'])
            v_gd *= (1.0 + row['ret_gold']); v_bd *= (1.0 + row['ret_bond'])
        nav = v_zz + v_nq + v_gd + v_bd
        if dt in rebal or idx == 0:
            Q = row['val_q_1000']
            Q = 0.5 if pd.isna(Q) else Q
            if use_timing:
                W_val = val_coeff * (1.0 - Q)
                c, ma = row['close_1000'], row['ma_1000']
                if pd.isna(ma):
                    M = 1.0
                else:
                    D = (c - ma) / ma
                    M = 1.0 if Q <= 0.15 else (1.0 if D >= 0.05 else (0.8 if D >= 0 else (0.6 if D >= -0.05 else (0.4 if D >= -0.10 else 0.3))))
                W_t = W_val * M
                W_cur = v_zz / nav if nav > 0 else 0.0
                if Q <= 0.20:
                    W_zz = W_t
                elif Q <= 0.80:
                    W_zz = min(W_t, W_cur + 0.05) if W_cur < W_t else W_t
                else:
                    W_zz = W_cur if W_cur < W_t else W_t
            else:
                W_zz = 1.0 - W_nasdaq - W_gold if use_satellite else 1.0
            W_n, W_g = W_nasdaq, W_gold
            tot = W_zz + W_n + W_g
            if tot > 1.0:
                W_zz, W_n, W_g = W_zz / tot, W_n / tot, W_g / tot
                W_b = 0.0
            else:
                W_b = 1.0 - tot
            devs = [abs(v_zz / nav - W_zz) if nav > 0 else 1, abs(v_nq / nav - W_n) if nav > 0 else 1,
                    abs(v_gd / nav - W_g) if nav > 0 else 1, abs(v_bd / nav - W_b) if nav > 0 else 1]
            if any(d > dev_threshold for d in devs) or idx == 0:
                cost = (abs(nav * W_zz - v_zz) + abs(nav * W_b - v_bd)) * 0.0005 \
                     + (abs(nav * W_n - v_nq) + abs(nav * W_g - v_gd)) * 0.0010
                nav -= cost
                v_zz, v_nq, v_gd, v_bd = nav * W_zz, nav * W_n, nav * W_g, nav * W_b
        navs.append({'trade_date': dt, 'nav': nav})
    return pd.DataFrame(navs).set_index('trade_date')['nav']

# ---------- 3. 数据与回测 ----------
df_all = load_premium_data(ma_window=250, val_window=1200)
start = alpha_tv.index.min()
end = pd.to_datetime("2026-07-14")
wd = df_all[(df_all['trade_date'] >= start) & (df_all['trade_date'] <= end)].copy().reset_index(drop=True)
print(f"回测窗口: {wd['trade_date'].min().date()} ~ {wd['trade_date'].max().date()}, {len(wd)} 天")

variants = {
    "A 常数11%(现状)": dict(alpha_const=0.11),
    "B 时变贴水驱动": dict(alpha_series=alpha_tv),
    "C 常数9.3%(参考)": dict(alpha_const=0.093),
    "D 0%(无增强)": dict(alpha_const=0.0),
}
navs = {}
for name, kw in variants.items():
    navs[name] = run_variant_tv(wd, **kw)
idx_nav = (1 + wd['ret_1000']).cumprod() * 1e6
idx_nav.index = wd['trade_date']
navs["纯中证1000"] = idx_nav

print(f"\n{'变体':20s} {'总收益':>9s} {'CAGR':>8s} {'Sharpe':>7s} {'MaxDD':>9s}")
for name, nav in navs.items():
    m = compute_metrics(nav)
    print(f"{name:20s} {m['Total Return']:>8.2%} {m['CAGR']:>7.2%} {m['Sharpe']:>7.2f} {m['Max Drawdown']:>8.2%}")

# 时变α均值(实际生效值)
a_eff = alpha_tv.reindex(wd['trade_date']).ffill()
print(f"\n时变α实际生效: 均值 {a_eff.mean():.2%}, 起点 {a_eff.iloc[0]:.2%}, 终点 {a_eff.iloc[-1]:.2%}")

# ---------- 4. 保存净值与图 ----------
out = pd.DataFrame(navs)
out.to_csv(f"{ROOT}/csi1000_timevarying_alpha_nav.csv")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pd.io.common.os.path.dirname(sys.executable) + "/../.."))
try:
    from daimon_runtime import setup_plot
    setup_plot()
except Exception:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(11, 6))
for name in navs:
    lw = 2.2 if name.startswith("B") else (1.6 if name.startswith("A") else 1.0)
    ax.plot(navs[name].index, navs[name].values / 1e6, label=name, linewidth=lw)
ax.set_title("中证1000增强: 常数α vs 时变贴水驱动α (2022-11 ~ 2026-07)")
ax.set_ylabel("净值 (百万)")
ax.legend()
ax.grid(alpha=0.3)
fig.savefig(f"{ROOT}/csi1000_timevarying_alpha_nav.png", bbox_inches="tight")
print(f"已保存 csi1000_timevarying_alpha_nav.csv / .png")
