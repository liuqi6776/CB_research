# -*- coding: utf-8 -*-
"""中证1000增强收益来源反事实分解 (2026-07-17)

背景: step13 的"增强"收益 = 中证1000日收益 × (1 + alpha_annual=0.11 硬编码日化)
     择时覆盖 = val_q 估值定仓 + MA250 五档趋势 + Q<=0.15 强制入场 + 三档入场规则
     本验证已证明 val_q/MA250 在中证1000上方向不稳 → 需要分解组合超额到底来自哪里

实验 (IS 2015-01~2024-02 / OOS 2024-02~2026-03, 与 step13 一致):
  V0 纯中证1000买入持有
  V1 = V0 + 11%硬编码α (隔离α假设的贡献)
  V2 = V1 + 择时覆盖 (隔离择时的边际贡献)
  V3 = V2 + 卫星 (Nasdaq15%+Gold10%) = step13 完整策略
  V4 = 纯指数 + 择时 (无α, 检验择时本身)
  敏感度: alpha ∈ {0, 5%, 8%, 11%} 跑完整策略
"""
import os, sys
import pandas as pd
import numpy as np

SCRIPT_DIR = r"C:\Users\liuqi\quant_system_v2\etf-valuation-strategy\scripts"
sys.path.insert(0, SCRIPT_DIR)
from step13_backtest_premium import load_premium_data, compute_metrics

def run_variant(df_period, alpha_annual=0.11, use_timing=True, use_satellite=True,
                val_coeff=0.6, W_nasdaq=0.15, W_gold=0.10, dev_threshold=0.10,
                initial_capital=1000000.0):
    """step13 run_premium_backtest 的带开关复刻版"""
    if not use_satellite:
        W_nasdaq, W_gold = 0.0, 0.0
    df_period = df_period.copy().reset_index(drop=True)
    df_period['year_week'] = df_period['trade_date'].dt.strftime('%Y-%U')
    rebal_dates = set(df_period.groupby('year_week')['trade_date'].first())

    alpha_daily = (1.0 + alpha_annual) ** (1.0 / 242.0) - 1.0
    df_period['ret_enh'] = (1.0 + df_period['ret_1000']) * (1.0 + alpha_daily) - 1.0

    v_zz, v_nq, v_gd, v_bd = 0.0, 0.0, 0.0, initial_capital
    navs = []
    for idx, row in df_period.iterrows():
        dt = row['trade_date']
        if idx > 0:
            v_zz *= (1.0 + row['ret_enh'])
            v_nq *= (1.0 + row['ret_nasdaq'])
            v_gd *= (1.0 + row['ret_gold'])
            v_bd *= (1.0 + row['ret_bond'])
        nav = v_zz + v_nq + v_gd + v_bd

        if dt in rebal_dates or idx == 0:
            Q = row['val_q_1000']
            if pd.isna(Q):
                Q = 0.5
            if use_timing:
                W_val = val_coeff * (1.0 - Q)
                c, ma = row['close_1000'], row['ma_1000']
                if pd.isna(ma):
                    M = 1.0
                else:
                    D = (c - ma) / ma
                    if Q <= 0.15:
                        M = 1.0
                    elif D >= 0.05:
                        M = 1.0
                    elif D >= 0.0:
                        M = 0.8
                    elif D >= -0.05:
                        M = 0.6
                    elif D >= -0.10:
                        M = 0.4
                    else:
                        M = 0.3
                W_timed = W_val * M
                W_curr = v_zz / nav if nav > 0 else 0.0
                if Q <= 0.20:
                    W_t_zz = W_timed
                elif Q <= 0.80:
                    W_t_zz = min(W_timed, W_curr + 0.05) if W_curr < W_timed else W_timed
                else:
                    W_t_zz = W_curr if W_curr < W_timed else W_timed
            else:
                W_t_zz = 1.0 if not use_satellite else (1.0 - W_nasdaq - W_gold)

            W_t_nq, W_t_gd = W_nasdaq, W_gold
            tot = W_t_zz + W_t_nq + W_t_gd
            if tot > 1.0:
                W_t_zz, W_t_nq, W_t_gd = W_t_zz / tot, W_t_nq / tot, W_t_gd / tot
                W_t_bd = 0.0
            else:
                W_t_bd = 1.0 - tot

            devs = [abs(v_zz / nav - W_t_zz) if nav > 0 else 1,
                    abs(v_nq / nav - W_t_nq) if nav > 0 else 1,
                    abs(v_gd / nav - W_t_gd) if nav > 0 else 1,
                    abs(v_bd / nav - W_t_bd) if nav > 0 else 1]
            if any(d > dev_threshold for d in devs) or idx == 0:
                t_zz = abs(nav * W_t_zz - v_zz)
                t_nq = abs(nav * W_t_nq - v_nq)
                t_gd = abs(nav * W_t_gd - v_gd)
                t_bd = abs(nav * W_t_bd - v_bd)
                nav -= t_zz * 0.0005 + t_bd * 0.0005 + t_nq * 0.0010 + t_gd * 0.0010
                v_zz, v_nq, v_gd, v_bd = nav * W_t_zz, nav * W_t_nq, nav * W_t_gd, nav * W_t_bd
        navs.append({'trade_date': dt, 'nav': nav})
    return pd.DataFrame(navs).set_index('trade_date')['nav']

def buy_hold(df, ret_col='ret_1000', alpha_annual=0.0, cap=1000000.0):
    a = (1.0 + alpha_annual) ** (1.0 / 242.0) - 1.0
    nav = (1 + df[ret_col]).cumprod() * ((1 + a) ** np.arange(1, len(df) + 1)) * cap
    return pd.Series(nav.values, index=df['trade_date'])

# ---------- 数据 ----------
df_all = load_premium_data(ma_window=250, val_window=1200)
is_s, is_e = pd.to_datetime("2015-01-01"), pd.to_datetime("2024-02-05")
oos_s, oos_e = pd.to_datetime("2024-02-06"), pd.to_datetime("2026-03-13")

for wname, wd in [("IS 2015-2024.02", df_all[(df_all['trade_date'] >= is_s) & (df_all['trade_date'] <= is_e)]),
                  ("OOS 2024.02-2026.03", df_all[(df_all['trade_date'] >= oos_s) & (df_all['trade_date'] <= oos_e)])]:
    wd = wd.copy().reset_index(drop=True)
    print("\n" + "=" * 66)
    print(f"[{wname}] {len(wd)} 个交易日")

    res = {}
    res['V0 纯指数买入持有'] = compute_metrics(buy_hold(wd, alpha_annual=0.0))
    res['V1 = V0 + 11%硬编码α'] = compute_metrics(buy_hold(wd, alpha_annual=0.11))
    res['V2 = V1 + 择时覆盖'] = compute_metrics(run_variant(wd, alpha_annual=0.11, use_timing=True, use_satellite=False))
    res['V3 = V2 + 卫星(完整策略)'] = compute_metrics(run_variant(wd, alpha_annual=0.11, use_timing=True, use_satellite=True))
    res['V4 = 纯指数+择时(无α)'] = compute_metrics(run_variant(wd, alpha_annual=0.0, use_timing=True, use_satellite=False))

    print(f"{'变体':28s} {'CAGR':>9s} {'Sharpe':>7s} {'MaxDD':>9s}")
    for k, m in res.items():
        print(f"{k:28s} {m['CAGR']:>8.2%} {m['Sharpe']:>7.2f} {m['Max Drawdown']:>9.2%}")

    v = {k: m['CAGR'] for k, m in res.items()}
    print("\n  分解 (CAGR 边际贡献):")
    print(f"    硬编码α假设:   {v['V1 = V0 + 11%硬编码α'] - v['V0 纯指数买入持有']:+.2%}")
    print(f"    择时覆盖:      {v['V2 = V1 + 择时覆盖'] - v['V1 = V0 + 11%硬编码α']:+.2%}  (含α时)")
    print(f"    择时覆盖:      {v['V4 = 纯指数+择时(无α)'] - v['V0 纯指数买入持有']:+.2%}  (无α时)")
    print(f"    卫星分散:      {v['V3 = V2 + 卫星(完整策略)'] - v['V2 = V1 + 择时覆盖']:+.2%}")

# ---------- α 敏感度 (完整策略, OOS) ----------
print("\n" + "=" * 66)
print("[α 敏感度: 完整策略(V3), OOS 2024.02-2026.03]")
wd = df_all[(df_all['trade_date'] >= oos_s) & (df_all['trade_date'] <= oos_e)].copy().reset_index(drop=True)
for a in [0.0, 0.05, 0.08, 0.11]:
    m = compute_metrics(run_variant(wd, alpha_annual=a, use_timing=True, use_satellite=True))
    print(f"  alpha={a:.0%}: CAGR={m['CAGR']:.2%}, Sharpe={m['Sharpe']:.2f}, MaxDD={m['Max Drawdown']:.2%}")
m0 = compute_metrics(buy_hold(wd, alpha_annual=0.0))
print(f"  对照 纯指数买入持有: CAGR={m0['CAGR']:.2%}, Sharpe={m0['Sharpe']:.2f}, MaxDD={m0['Max Drawdown']:.2%}")
