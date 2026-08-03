# -*- coding: utf-8 -*-
"""研报止盈止损方法适配测试 (真实口径v2, 等权+MA20五档098 底座)
研报四大类 -> 本策略(月度调仓组合)适配:
  1. 波动率型-ATR止损: 净值回撤 > N1*ATR -> 降仓0.5; > N2*ATR -> 降仓0.25;
     谷底回升5%恢复满仓 (ATR=近14日|Δnav|均值, 自适应止损距离)
  2. 指标型-净值移动平均止损: 净值 < 自身MA20 -> 0.5; < MA50 -> 0.25
  3. 时间型-月度时间止损: 上月组合收益<0 -> 本月仓位减半 (与RS12动量重叠)
  4. 对照: 固定DD(10,15)+谷底回升5% (前轮最优), 无止损基准
价格型(入场价/保本)与布林带/RSI/SAR/MACD 不适用月度调仓组合, 见汇报。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements.risk_control_real import TIER5, COST_SINGLE, tier_w


def main():
    env = C.Env()
    td = env.trade_dates
    stocks, _, _, _, _ = rv.load_panels(td, env.all_codes, None)
    print(f"stocks {len(stocks)}")

    open_df = pd.DataFrame({c: g.sort_index()["open"] for c, g in stocks.items()})
    close_df = pd.DataFrame({c: g.sort_index()["close"] for c, g in stocks.items()})

    etf = C.load_idx("512100.SH")
    e_ovn_s = (etf["open"].reindex(td) / etf["pre_close"].reindex(td) - 1.0).fillna(0.0)
    e_intra_s = (etf["close"].reindex(td) / etf["open"].reindex(td) - 1.0).fillna(0.0)

    def run(method=None, atr_n1=None, atr_n2=None, dd_stop=None, dd_floor=None, recov=0.05):
        nav = 1.0
        peak = 1.0
        navs = {}
        nav_hist = [1.0]
        w_prev = 1.0
        prev_picks, prev_etf = None, False
        total_switch = 0.0
        total_turn = 0.0
        prev_month_ret = None
        for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
            if picks is None:
                continue
            w = pd.Series(1.0 / len(picks), index=picks)  # 等权底座
            use_etf = not rs12_on
            turn = 1.0
            if prev_picks is not None and prev_etf == use_etf:
                keep = len(set(picks) & set(prev_picks))
                turn = 1.0 - keep / len(picks)
            t0 = hold[0]
            if not use_etf:
                op0 = open_df.reindex(columns=picks).loc[t0].replace(0.0, np.nan)
                a = (w.reindex(picks) / op0).fillna(0.0)
                op_m = open_df.reindex(columns=picks).reindex(hold).ffill().bfill()
                cl_m = close_df.reindex(columns=picks).reindex(hold).ffill().bfill()
                V_open = op_m.multiply(a, axis=1).fillna(0.0).sum(axis=1)
                V_close = cl_m.multiply(a, axis=1).fillna(0.0).sum(axis=1)
            month_first_nav = nav
            month_switch = 0.0
            in_dd = False
            nav_low = nav
            for j, t in enumerate(hold):
                # MA20五档 (指数信号)
                w_t = tier_w(env.idx_close_1.get(t), env.ma20_1.get(t), TIER5) if rs12_on else 1.0
                # ---- 止损层 ----
                if method == "dd":
                    dd = (peak - nav) / peak if peak > 0 else 0.0
                    if in_dd:
                        nav_low = min(nav_low, nav)
                        if (nav / nav_low - 1.0) >= recov:
                            in_dd = False
                            w_dd = 1.0
                        elif dd >= dd_floor:
                            w_dd = 0.25
                        else:
                            w_dd = 0.5
                    else:
                        if dd >= dd_stop:
                            in_dd = True
                            nav_low = nav
                            w_dd = 0.5
                        else:
                            w_dd = 1.0
                    w_t = min(w_t, w_dd)
                elif method == "atr":
                    atr = None
                    if len(nav_hist) >= 15:
                        x = np.array(nav_hist[-15:])
                        atr = np.abs(np.diff(x) / x[:-1]).mean()
                    dd = (peak - nav) / peak if peak > 0 else 0.0
                    if atr is not None and atr > 0:
                        if in_dd:
                            nav_low = min(nav_low, nav)
                            if (nav / nav_low - 1.0) >= recov:
                                in_dd = False
                                w_dd = 1.0
                            elif dd >= atr_n2 * atr:
                                w_dd = 0.25
                            else:
                                w_dd = 0.5
                        else:
                            if dd >= atr_n1 * atr:
                                in_dd = True
                                nav_low = nav
                                w_dd = 0.5
                            else:
                                w_dd = 1.0
                        w_t = min(w_t, w_dd)
                elif method == "navma":
                    if len(nav_hist) >= 50:
                        arr = np.array(nav_hist)
                        ma20 = arr[-20:].mean()
                        ma50 = arr[-50:].mean()
                        if nav < ma50:
                            w_dd = 0.25
                        elif nav < ma20:
                            w_dd = 0.5
                        else:
                            w_dd = 1.0
                        w_t = min(w_t, w_dd)
                elif method == "timestop":
                    if prev_month_ret is not None and prev_month_ret < 0:
                        w_t = min(w_t, 0.5)
                # ---- 当日收益 (隔夜按旧仓/日内按新仓) ----
                if use_etf:
                    ovn_t, intra_t = e_ovn_s[t], e_intra_s[t]
                    r = w_t * intra_t if j == 0 else (1.0 + w_prev * ovn_t) * (1.0 + w_t * intra_t) - 1.0
                else:
                    if j == 0:
                        r = w_t * (V_close.iloc[0] / V_open.iloc[0] - 1.0)
                    else:
                        ovn_t = V_open.iloc[j] / V_close.iloc[j - 1] - 1.0
                        intra_t = V_close.iloc[j] / V_open.iloc[j] - 1.0
                        r = (1.0 + w_prev * ovn_t) * (1.0 + w_t * intra_t) - 1.0
                # ---- 成本 ----
                if j == 0:
                    c_turn = turn * C.COST
                    nav *= (1.0 - c_turn)
                    total_turn += c_turn
                elif w_t != w_prev:
                    c_sw = abs(w_t - w_prev) * COST_SINGLE
                    nav *= (1.0 - c_sw)
                    month_switch += c_sw
                nav *= (1.0 + r)
                w_prev = w_t
                peak = max(peak, nav)
                nav_hist.append(nav)
                navs[t] = nav
            total_switch += month_switch
            prev_month_ret = nav / month_first_nav - 1.0 if month_first_nav > 0 else 0.0
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
        print(f"{tag:36s} 年化 {cagr:6.2%}  Sharpe {shp:5.2f}  日频MaxDD {dd:6.2%}  卡玛 {cagr / dd:.2f}")
        return dict(cagr=cagr, shp=shp, dd=dd, k=cagr / dd)

    print("=" * 104)
    print("研报止盈止损方法适配 (真实口径, 等权+MA20五档098 底座)")
    print("=" * 104)
    results = {}
    variants = [
        ("基准(无止损)", None, None, None, None, None),
        ("固定DD(10,15)+回升5%", "dd", None, None, 0.10, 0.15),
        ("ATR(2,3)+回升5%", "atr", 2.0, 3.0, None, None),
        ("ATR(2.5,3.5)+回升5%", "atr", 2.5, 3.5, None, None),
        ("ATR(3,4)+回升5%", "atr", 3.0, 4.0, None, None),
        ("净值MA20/50", "navma", None, None, None, None),
        ("时间止损(月负减半)", "timestop", None, None, None, None),
    ]
    for lb, method, a1, a2, ds, df_ in variants:
        s, sw, tr = run(method=method, atr_n1=a1, atr_n2=a2, dd_stop=ds, dd_floor=df_)
        r = stats(s, lb)
        r.update(switch=sw, turn=tr)
        results[lb] = r
        print(f"    (切换摩擦 {sw:.2%} NAV, 调仓换手 {tr:.2%} NAV)")
        print("-" * 104)

    print("\nvs 基准 (日频MaxDD / 年化 / 卡玛):")
    b = results["基准(无止损)"]
    for lb in [v[0] for v in variants[1:]]:
        r = results[lb]
        print(f"  {lb:24s} MaxDD {b['dd'] - r['dd']:+.2%}  年化 {r['cagr'] - b['cagr']:+.2%}  "
              f"Sharpe {r['shp'] - b['shp']:+.2f}  卡玛 {r['k'] - b['k']:+.2f}")

    out = "\n".join([f"{k}: {v}" for k, v in results.items()])
    with open(os.path.join(C.OUT_DIR, "risk_control_methods.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"\n[saved] {os.path.join(C.OUT_DIR, 'risk_control_methods.txt')}")


if __name__ == "__main__":
    main()
