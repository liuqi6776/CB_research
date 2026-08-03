# -*- coding: utf-8 -*-
"""真实口径 v2 + 组合净值回撤止损 (止盈止损层)
在日频 MaxDD 真实度量之上, 叠加 DD-based de-risking (移动止盈):
  - 净值从历史高点回撤 >= dd_stop  -> 仓位降到 stop_w (0.5)
  - 回撤 >= dd_floor               -> 仓位降到 floor_w (0.0, 空仓锁利)
  - 净值创新高(peak)                -> 仓位恢复 1.0
  - 与 MA20 档位取更保守值: w_t = min(ma20_tier, w_dd)
时序 (PIT): t 日开盘用 t-1 收盘后净值判断回撤, 决定今日暴露; 切换成本单边10bps。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements.direction2_hrp import _hrp_weights, WINDOW
from research.studies.study_008_enhancements.risk_control_real import TIER3, TIER5, COST_SINGLE, tier_w, cap_weights


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

    def run(use_hrp, use_ma20, tier=None, dd_stop=None, dd_floor=None,
            stop_w=0.5, floor_w=0.0, recov=None, cap=None):
        # recov: 恢复条件 'half'=回撤收窄到 dd_stop/2; float>0 = 净值从谷底回升该比例
        nav = 1.0
        peak = 1.0
        navs = {}
        w_prev = 1.0
        prev_picks, prev_etf = None, False
        total_switch = 0.0
        total_turn = 0.0
        n_stop = n_floor = n_cash = n_recov = 0
        in_dd = False
        nav_low = 1.0
        for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
            if picks is None:
                continue
            hi = td.index(rb)
            win = td[max(0, hi - WINDOW):hi]
            rets = env.pct_df.reindex(columns=picks).reindex(win)
            w = _hrp_weights(rets) if use_hrp else pd.Series(1.0 / len(picks), index=picks)
            w = cap_weights(w, cap)
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
            month_switch = 0.0
            for j, t in enumerate(hold):
                # ---- MA20 档位 (指数信号) ----
                if use_ma20 and rs12_on:
                    w_t = tier_w(env.idx_close_1.get(t), env.ma20_1.get(t), tier)
                else:
                    w_t = 1.0
                # ---- 组合净值回撤止损层 (基于昨日收盘净值, PIT) ----
                if dd_stop is not None:
                    dd = (peak - nav) / peak if peak > 0 else 0.0
                    if in_dd:
                        nav_low = min(nav_low, nav)
                        if recov == "half":
                            recover = dd <= dd_stop / 2.0
                        elif isinstance(recov, (int, float)):
                            recover = (nav / nav_low - 1.0) >= recov
                        else:
                            recover = False
                        if recover:
                            in_dd = False
                            w_dd = 1.0
                            n_recov += 1
                        elif dd >= dd_floor:
                            w_dd = floor_w
                            n_floor += 1
                            if floor_w == 0.0:
                                n_cash += 1
                        else:
                            w_dd = stop_w
                            n_stop += 1
                    else:
                        if dd >= dd_stop:
                            in_dd = True
                            nav_low = nav
                            w_dd = stop_w
                            n_stop += 1
                        else:
                            w_dd = 1.0
                    w_t = min(w_t, w_dd)
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
                # ---- 成本 (调仓日换手 / 切换单边) ----
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
                navs[t] = nav
            total_switch += month_switch
            prev_picks = picks
            prev_etf = use_etf
        s = pd.Series(navs).sort_index()
        return s, total_switch, total_turn, n_stop, n_floor, n_cash, n_recov

    def stats(s, tag):
        n = len(s)
        cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
        ret = s.pct_change().dropna()
        shp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(242.0)
        dd = ((s.cummax() - s) / s.cummax()).max()
        print(f"{tag:38s} 年化 {cagr:6.2%}  Sharpe {shp:5.2f}  日频MaxDD {dd:6.2%}  卡玛 {cagr / dd:.2f}")
        return dict(cagr=cagr, shp=shp, dd=dd, k=cagr / dd)

    print("=" * 104)
    print("真实口径 + 组合净值回撤止损 (止盈止损层; 日频MaxDD度量)")
    print("=" * 104)
    results = {}
    grid = {}
    # ---- 敏感性网格: DD阈值 x 谷底回升 ----
    dd_pairs = [(0.08, 0.12), (0.09, 0.14), (0.10, 0.15), (0.12, 0.18)]
    recovs = [0.02, 0.03, 0.05]
    for ds, df_ in dd_pairs:
        grid[(ds, df_)] = {}
        for rec in recovs:
            lb = f"+DD({int(ds*100)},{int(df_*100)}) 回升{int(rec*100)}%"
            s, sw, tr, ns, nf, nc, nrec = run(True, True, TIER5, dd_stop=ds, dd_floor=df_,
                                              stop_w=0.5, floor_w=0.5, recov=rec)
            r = stats(s, lb)
            r.update(switch=sw, turn=tr, n_stop=ns, n_floor=nf, n_cash=nc, n_recov=nrec)
            results[lb] = r
            grid[(ds, df_)][rec] = r
            print(f"    (切换摩擦 {sw:.2%}, 降仓 {ns} 次, 清仓 {nf} 次, 恢复 {nrec} 次)")
            print("-" * 104)
    # ---- 基准 ----
    s, sw, tr, ns, nf, nc, nrec = run(True, True, TIER5, dd_stop=None)
    r0 = stats(s, "基准+MA20五档098 (无止损)")
    r0.update(switch=sw, turn=tr)
    results["基准+MA20五档098 (无止损)"] = r0
    print("-" * 104)

    print("\n敏感性网格 (卡玛 = 年化/日频MaxDD):")
    print(f"{'DD阈值':>8s} | " + " | ".join([f"回升{int(rec*100)}%" for rec in recovs]))
    for ds, df_ in dd_pairs:
        row = [grid[(ds, df_)][rec] for rec in recovs]
        print(f"{int(ds*100)}/{int(df_*100)}  | " + " | ".join(
            [f"年化{r['cagr']:5.1%} MaxDD{r['dd']:5.1%} 卡玛{r['k']:4.2f}" for r in row]))
    print(f"基准       | 年化{r0['cagr']:5.1%} MaxDD{r0['dd']:5.1%} 卡玛{r0['k']:4.2f}")
    print("\n相邻参数平滑性检验 (MaxDD / 年化 对参数变化的敏感度):")
    for i in range(len(dd_pairs) - 1):
        for j in range(len(recovs)):
            r_a = grid[dd_pairs[i]][recovs[j]]
            r_b = grid[dd_pairs[i + 1]][recovs[j]]
            print(f"  {int(dd_pairs[i][0]*100)}/{int(dd_pairs[i][1]*100)}->{int(dd_pairs[i+1][0]*100)}/{int(dd_pairs[i+1][1]*100)} 回升{int(recovs[j]*100)}%: "
                  f"年化 Δ{r_b['cagr']-r_a['cagr']:+.2%}  MaxDD Δ{r_b['dd']-r_a['dd']:+.2%}")
        r_a = grid[dd_pairs[i]][recovs[0]]
        r_b = grid[dd_pairs[i]][recovs[1]]
        r_c = grid[dd_pairs[i]][recovs[2]]
        print(f"  回升2%->3%->5% (阈值{int(dd_pairs[i][0]*100)}/{int(dd_pairs[i][1]*100)}): 卡玛 {r_a['k']:.2f} -> {r_b['k']:.2f} -> {r_c['k']:.2f}")

    out = "\n".join([f"{k}: {v}" for k, v in results.items()])
    with open(os.path.join(C.OUT_DIR, "risk_control_ddstop.txt"), "w", encoding="utf-8") as f:
        f.write(out + "\n")
    print(f"\n[saved] {os.path.join(C.OUT_DIR, 'risk_control_ddstop.txt')}")


if __name__ == "__main__":
    main()
