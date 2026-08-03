# -*- coding: utf-8 -*-
"""真实口径回测 v2 (P0 修复, 审查意见 [3][4][5][7][10])
真实口径 vs 原口径:
  1. 日频 MaxDD: 逐交易日净值计算回撤
  2. T 日开盘成交: 调仓日只计日内段; 风控切换日隔夜按旧仓/日内按新仓
  3. 组合买入持有 + 权重漂移 (非每日 constant-mix): 调仓日按目标权重买入,
     之后份额固定, 权重随价格漂移
  4. 真实换手成本:
     - 月度调仓 = 换手率(old∩new) x COST_BPS (双边)
     - 档位切换 = |Δw| x COST_BPS/2 (单边, 单向交易)
  5. 单股权重上限敏感性 (cap)
残余简化: ETF 切换与部分成交假设未建模, 见 README。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements.direction2_hrp import _hrp_weights, WINDOW

TIER3 = {"bnd": [1.0, 0.98], "w": [1.0, 0.5]}
TIER5 = {"bnd": [1.0, 0.99, 0.98, 0.97], "w": [1.0, 0.75, 0.5, 0.25]}
COST_SINGLE = C.COST / 2.0  # 10bps


def tier_w(c, m, tier):
    try:
        c, m = float(c), float(m)
    except (TypeError, ValueError):
        return 1.0
    if not (np.isfinite(c) and np.isfinite(m)):
        return 1.0
    r = c / m
    for wgt, bnd in zip(tier["w"], tier["bnd"]):
        if r >= bnd:
            return wgt
    return 0.0


def cap_weights(w, cap):
    if cap is None:
        return w
    w = w.clip(upper=cap)
    return w / w.sum()


def main():
    env = C.Env()
    td = env.trade_dates
    stocks, _, _, _, _ = rv.load_panels(td, env.all_codes, None)
    print(f"stocks {len(stocks)}")

    # ---- 个股 open/close 宽表 (漂移模型需要绝对价格) ----
    open_df = pd.DataFrame({c: g.sort_index()["open"] for c, g in stocks.items()})
    close_df = pd.DataFrame({c: g.sort_index()["close"] for c, g in stocks.items()})

    # ETF
    etf = C.load_idx("512100.SH")
    e_open_s = etf["open"].reindex(td).fillna(np.nan)
    e_close_s = etf["close"].reindex(td).fillna(np.nan)
    e_pre_s = etf["pre_close"].reindex(td).fillna(np.nan)
    e_ovn_s = (e_open_s / e_pre_s - 1.0).fillna(0.0)
    e_intra_s = (e_close_s / e_open_s - 1.0).fillna(0.0)

    def run(use_hrp, use_ma20, tier=None, cap=None, cost_on=True):
        nav = 1.0
        navs = {}
        w_prev = 1.0
        prev_picks, prev_etf = None, False
        total_switch = 0.0
        total_turn = 0.0
        for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
            if picks is None:
                continue
            hi = td.index(rb)
            win = td[max(0, hi - WINDOW):hi]
            rets = env.pct_df.reindex(columns=picks).reindex(win)
            if use_hrp:
                w = _hrp_weights(rets)
            else:
                w = pd.Series(1.0 / len(picks), index=picks)
            w = cap_weights(w, cap)
            use_etf = not rs12_on
            # 换手
            turn = 1.0
            if prev_picks is not None and prev_etf == use_etf:
                keep = len(set(picks) & set(prev_picks))
                turn = 1.0 - keep / len(picks)
            t0 = hold[0]
            if use_etf:
                # ETF 单一资产: 份额=1, 无漂移
                pass
            else:
                # 调仓日开盘买入: 份额 a_i = w_i / open_i,t0  (V_open,t0 = Σw = 1)
                op0 = open_df.reindex(columns=picks).loc[t0].replace(0.0, np.nan)
                a = (w.reindex(picks) / op0).fillna(0.0)
                # 预取本月价格矩阵 (停牌日 ffill 延续前收, 组合市值不变)
                # 用 multiply+fillna(0)+sum 而非 dot: dot 中 NaN*0=NaN 会传播,
                # 停牌导致份额 a=0 的列在 dot 后仍污染整行
                op_m = open_df.reindex(columns=picks).reindex(hold).ffill().bfill()
                cl_m = close_df.reindex(columns=picks).reindex(hold).ffill().bfill()
                V_open = op_m.multiply(a, axis=1).fillna(0.0).sum(axis=1)
                V_close = cl_m.multiply(a, axis=1).fillna(0.0).sum(axis=1)
            month_switch = 0.0
            for j, t in enumerate(hold):
                if use_etf:
                    ovn_t, intra_t = e_ovn_s[t], e_intra_s[t]
                    if rs12_on and use_ma20:
                        w_t = tier_w(env.idx_close_1.get(t), env.ma20_1.get(t), tier)
                    else:
                        w_t = 1.0
                    if j == 0:
                        r = w_t * intra_t
                    else:
                        r = (1.0 + w_prev * ovn_t) * (1.0 + w_t * intra_t) - 1.0
                else:
                    if rs12_on and use_ma20:
                        w_t = tier_w(env.idx_close_1.get(t), env.ma20_1.get(t), tier)
                    else:
                        w_t = 1.0
                    if j == 0:
                        # 开盘买入组合, 只计日内段
                        ovn0 = 0.0
                        intra0 = V_close.iloc[0] / V_open.iloc[0] - 1.0
                        r = (1.0 + w_prev * ovn0) * (1.0 + w_t * intra0) - 1.0
                    else:
                        ovn_t = V_open.iloc[j] / V_close.iloc[j - 1] - 1.0
                        intra_t = V_close.iloc[j] / V_open.iloc[j] - 1.0
                        r = (1.0 + w_prev * ovn_t) * (1.0 + w_t * intra_t) - 1.0
                if cost_on and j == 0:
                    # 调仓日开盘买入, 扣月度换手成本 (t0 真实时点)
                    c_turn = turn * C.COST
                    nav *= (1.0 - c_turn)
                    total_turn += c_turn
                elif cost_on and j > 0 and w_t != w_prev:
                    # 月中风控切换: 单向交易, 单边成本, 在切换日扣除
                    c_sw = abs(w_t - w_prev) * COST_SINGLE
                    nav *= (1.0 - c_sw)
                    month_switch += c_sw
                nav *= (1.0 + r)
                w_prev = w_t
                navs[t] = nav
            if cost_on:
                total_switch += month_switch
            prev_picks = picks
            prev_etf = use_etf
        s = pd.Series(navs).sort_index()
        return s, total_switch, total_turn

    def stats(s, tag):
        n = len(s)
        cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
        ret = s.pct_change().dropna()
        shp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(242.0)
        dd = ((s.cummax() - s) / s.cummax()).max()
        sm = s.groupby(s.index.str[:6]).last()
        mdd_m = ((sm.cummax() - sm) / sm.cummax()).max()
        print(f"{tag:38s} 年化 {cagr:6.2%}  Sharpe {shp:5.2f}  日频MaxDD {dd:6.2%}  "
              f"月末MaxDD {mdd_m:6.2%}  卡玛(日频) {cagr / dd:.2f}")
        return dict(cagr=cagr, shp=shp, dd=dd, mdd_m=mdd_m, k=cagr / dd)

    print("=" * 104)
    print("真实口径v2 (日频MaxDD + T日开盘成交 + 买入持有漂移 + 真实换手成本)")
    print("=" * 104)
    results = {}
    variants = [
        ("BASE+VAL", False, False, None, None),
        ("+MA20三档098", False, True, TIER3, None),
        ("+MA20五档098", False, True, TIER5, None),
        ("+HRP", True, False, None, None),
        ("+HRP+MA20三档098", True, True, TIER3, None),
        ("+HRP+MA20五档098", True, True, TIER5, None),
        ("+HRP+MA20五档098+Cap5", True, True, TIER5, 0.05),
    ]
    for lb, hrp, ma20, tier, cap in variants:
        s, sw, tr = run(hrp, ma20, tier, cap)
        r = stats(s, lb)
        r["switch"] = sw
        r["turn"] = tr
        results[lb] = r
        print(f"    (档位切换摩擦 {sw:.2%} NAV, 调仓换手 {tr:.2%} NAV)")
        print("-" * 104)
    print()
    print("原口径参考 (月末MaxDD / 固定20bps / constant-mix):")
    print("  BASE+VAL          年化14.78%  Sharpe0.84  月末MaxDD 20.66%  卡玛 0.72")
    print("  +MA20三档098       年化15.72%  Sharpe0.93  月末MaxDD 18.06%  卡玛 0.87")
    print("  +HRP               年化15.00%  Sharpe0.82  月末MaxDD 18.37%  卡玛 0.82")
    print("  +HRP+MA20三档098   年化15.58%  Sharpe0.95  月末MaxDD 17.34%  卡玛 0.90")
    print("  +HRP+MA20五档098   年化16.90%  Sharpe0.99  月末MaxDD 17.16%  卡玛 0.98")
    print()
    print("五档 vs 三档 (真实口径, 年化增量):")
    print(f"  HRP+五档 - HRP+三档 = {results['+HRP+MA20五档098']['cagr'] - results['+HRP+MA20三档098']['cagr']:+.2%}")
    print(f"  等权+五档 - 等权+三档 = {results['+MA20五档098']['cagr'] - results['+MA20三档098']['cagr']:+.2%}")

    out = "\n".join([f"{k}: {v}" for k, v in results.items()])
    with open(os.path.join(C.OUT_DIR, "risk_control_real.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"\n[saved] {os.path.join(C.OUT_DIR, 'risk_control_real.txt')}")


if __name__ == "__main__":
    main()
