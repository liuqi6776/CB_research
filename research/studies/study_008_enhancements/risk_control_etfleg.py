# -*- coding: utf-8 -*-
"""第二步: ETF 腿专属风控 (RS12 弱段持 512100 时叠加 MA5/MA10/TREND 减仓)
与股票腿 MA20 解耦:
  - 股票腿: 000852 close vs MA20 五档 (原 TIER5)
  - ETF 腿: 512100 close vs MA5 / MA10 / MA120(regime TREND 开关), 均为 T-1 信号 T 日生效
底座 = +HRP+MA20五档098 (最优变体), 真实口径 (日频MaxDD / T日开盘 / 漂移 / 真实成本)
验证目标: 2026-07 大跌段 (RS12 弱段 + 512100 -19.7%) 是 ETF 腿回撤下限来源,
          专属减仓是否能在保弱段收益的同时降回撤。
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
TIER3_95 = {"bnd": [1.0, 0.95], "w": [1.0, 0.5]}
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

    open_df = pd.DataFrame({c: g.sort_index()["open"] for c, g in stocks.items()})
    close_df = pd.DataFrame({c: g.sort_index()["close"] for c, g in stocks.items()})

    etf = C.load_idx("512100.SH")
    e_open_s = etf["open"].reindex(td).fillna(np.nan)
    e_close_s = etf["close"].reindex(td).fillna(np.nan)
    e_pre_s = etf["pre_close"].reindex(td).fillna(np.nan)
    e_ovn_s = (e_open_s / e_pre_s - 1.0).fillna(0.0)
    e_intra_s = (e_close_s / e_open_s - 1.0).fillna(0.0)

    def run(use_hrp=True, use_ma20=True, tier=TIER5, cap=None, cost_on=True, etf_cfg=None):
        """etf_cfg: None=ETF腿无风控; {'ma':5|10,'tier':tier} MA档位; {'trend':True,'trend_w':0.25} TREND开关"""
        nav = 1.0
        navs = {}
        w_prev = 1.0
        prev_picks, prev_etf = None, False
        total_switch = 0.0
        total_turn = 0.0
        etf_cut_days = 0   # ETF 腿降仓天数累计
        etf_cut_months = {}  # 每月降仓天数
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
            turn = 1.0
            if prev_picks is not None and prev_etf == use_etf:
                keep = len(set(picks) & set(prev_picks))
                turn = 1.0 - keep / len(picks)
            t0 = hold[0]
            if use_etf:
                pass
            else:
                op0 = open_df.reindex(columns=picks).loc[t0].replace(0.0, np.nan)
                a = (w.reindex(picks) / op0).fillna(0.0)
                op_m = open_df.reindex(columns=picks).reindex(hold).ffill().bfill()
                cl_m = close_df.reindex(columns=picks).reindex(hold).ffill().bfill()
                V_open = op_m.multiply(a, axis=1).fillna(0.0).sum(axis=1)
                V_close = cl_m.multiply(a, axis=1).fillna(0.0).sum(axis=1)

            def etf_w(t):
                if etf_cfg is None:
                    return 1.0
                if etf_cfg.get("trend"):
                    c1 = env.etf_close_1.get(t)
                    m120 = env.ma120_1.get(t)
                    if c1 is None or m120 is None or not (np.isfinite(c1) and np.isfinite(m120)):
                        return 1.0
                    return 1.0 if c1 > m120 else etf_cfg.get("trend_w", 0.25)
                ma = env.ma5_1 if etf_cfg.get("ma") == 5 else env.ma10_1
                return tier_w(env.etf_close_1.get(t), ma.get(t), etf_cfg.get("tier"))

            month_cut = 0
            month_switch = 0.0
            for j, t in enumerate(hold):
                if use_etf:
                    ovn_t, intra_t = e_ovn_s[t], e_intra_s[t]
                    if rs12_on and use_ma20:
                        w_t = tier_w(env.idx_close_1.get(t), env.ma20_1.get(t), tier)
                    else:
                        w_t = etf_w(t)  # ETF 腿专属风控 (弱段内)
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
                        ovn0 = 0.0
                        intra0 = V_close.iloc[0] / V_open.iloc[0] - 1.0
                        r = (1.0 + w_prev * ovn0) * (1.0 + w_t * intra0) - 1.0
                    else:
                        ovn_t = V_open.iloc[j] / V_close.iloc[j - 1] - 1.0
                        intra_t = V_close.iloc[j] / V_open.iloc[j] - 1.0
                        r = (1.0 + w_prev * ovn_t) * (1.0 + w_t * intra_t) - 1.0
                if cost_on and j == 0:
                    c_turn = turn * C.COST
                    nav *= (1.0 - c_turn)
                    total_turn += c_turn
                elif cost_on and j > 0 and w_t != w_prev:
                    c_sw = abs(w_t - w_prev) * COST_SINGLE
                    nav *= (1.0 - c_sw)
                    month_switch += c_sw
                nav *= (1.0 + r)
                w_prev = w_t
                navs[t] = nav
                if use_etf and etf_cfg is not None and w_t < 1.0:
                    etf_cut_days += 1
                    month_cut += 1
            if cost_on:
                total_switch += month_switch
            if month_cut:
                etf_cut_months[rb[:6]] = month_cut
            prev_picks = picks
            prev_etf = use_etf
        s = pd.Series(navs).sort_index()
        return s, total_switch, total_turn, etf_cut_days, etf_cut_months

    def stats(s, tag, cut_days=0):
        n = len(s)
        cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
        ret = s.pct_change().dropna()
        shp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(242.0)
        dd = ((s.cummax() - s) / s.cummax()).max()
        sm = s.groupby(s.index.str[:6]).last()
        mdd_m = ((sm.cummax() - sm) / sm.cummax()).max()
        cut_s = f"  ETF降仓{cut_days}天" if cut_days else ""
        print(f"{tag:42s} 年化 {cagr:6.2%}  Sharpe {shp:5.2f}  日频MaxDD {dd:6.2%}  "
              f"月末MaxDD {mdd_m:6.2%}  卡玛 {cagr / dd:.2f}{cut_s}")
        return dict(cagr=cagr, shp=shp, dd=dd, mdd_m=mdd_m, k=cagr / dd)

    print("=" * 116)
    print("第二步: ETF 腿专属风控 (底座 = +HRP+MA20五档098, 真实口径, 2020-01~2026-07 数据补全)")
    print("=" * 116)
    results = {}
    variants = [
        ("基准: ETF腿无风控", None),
        ("+ETF腿 MA5三档098 (降0.5)", {"ma": 5, "tier": TIER3}),
        ("+ETF腿 MA5三档095 (降0.5)", {"ma": 5, "tier": TIER3_95}),
        ("+ETF腿 MA10三档098 (降0.5)", {"ma": 10, "tier": TIER3}),
        ("+ETF腿 MA10五档098 (0.25~1)", {"ma": 10, "tier": TIER5}),
        ("+ETF腿 TREND MA120 (弱0.25)", {"trend": True, "trend_w": 0.25}),
        ("+ETF腿 TREND MA120 (弱0.50)", {"trend": True, "trend_w": 0.50}),
    ]
    for lb, etf_cfg in variants:
        s, sw, tr, cut_days, cut_months = run(etf_cfg=etf_cfg)
        r = stats(s, lb, cut_days)
        r["switch"] = sw
        r["turn"] = tr
        r["cut_days"] = cut_days
        r["cut_months"] = cut_months
        results[lb] = r
        print(f"    (档位切换摩擦 {sw:.2%} NAV, 调仓换手 {tr:.2%} NAV)")
        if etf_cfg is not None:
            print(f"    ETF降仓月分布: {cut_months}")
        print("-" * 116)

    # 7 月大跌段核查: 输出 ETF 腿降仓最多的月份, 确认 2026-07 在列
    print()
    all_cut = {}
    for lb, _ in variants:
        cm = results[lb].get("cut_months", {})
        if cm:
            all_cut[lb] = sorted(cm.items(), key=lambda x: -x[1])[:3]
    for lb, lst in all_cut.items():
        print(f"  {lb}: ETF降仓最活跃月 {lst}")
    print()
    out = "\n".join([f"{k}: { {kk: vv for kk, vv in v.items() if kk != 'cut_months'} }" for k, v in results.items()])
    with open(os.path.join(C.OUT_DIR, "risk_control_etfleg.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"\n[saved] {os.path.join(C.OUT_DIR, 'risk_control_etfleg.txt')}")


if __name__ == "__main__":
    main()
