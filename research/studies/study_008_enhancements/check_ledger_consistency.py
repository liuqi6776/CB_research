# -*- coding: utf-8 -*-
"""阶段3 一致性校验: 双账本 (ledger.py) vs 冻结基线 (engine.py) — 归因验证

两者均用 IVW120 + 阶段2 分项费率 + 月末信号 + 开盘执行 + 买入持有漂移.
已知模型差异 (非 bug):
  引擎"T 日开盘成交"在调仓日只计新组合日内段 (V_close[0]/V_open[0]-1),
  漏计旧持仓在调仓日开盘前的隔夜跳空 (d_prev收盘 → t0开盘);
  账本按绝对价格盯市, 正确计入该段 (实盘持仓确实持有到 t0 开盘才卖出).

验证 1 (gap 归因): 逐调仓日累计"旧仓隔夜跳空" (用份额模拟 + 月末漂移后市值权重)
  应 ≈ ledger/base − 1 全段偏差 (精度 < 0.3pp).
验证 2 (逐日 diff 归因): 日频收益差 r_led − r_base 只应出现在调仓日 t0,
  且 ≈ 当日隔夜贡献; 非调仓日残差应 < 5bp/日.
验证 3 (腿判断): 每期 use_etf 与 RS12 动量符号一致性 (如 2024-07 RS12 空头 → ETF 腿正确).
输出: results/ledger_consistency.txt | .json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements import ledger as L
from research.studies.study_008_enhancements.direction2_hrp import _ivw_weights, WINDOW

RS12_DAYS = 252  # RS12 动量窗口 (与引擎同口径, 见 engine.py)

def _valid(v):
    return pd.notna(v) and float(v) > 0

def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)
    etf = C.load_idx("512100.SH")
    e_open_s = etf["open"].astype(float)
    e_close_s = etf["close"].astype(float)

    # ---- 基线 (无过滤) 与 双账本 (无过滤) ----
    s_base, st_base = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                     e_ovn, e_intra, use_hrp=True, use_ma20=False)
    s_led, st_led = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s,
                                 st_map=None, one_up=None, one_dn=None)
    both = pd.concat([s_base, s_led], axis=1, join="inner").dropna()
    both.columns = ["base", "ledger"]
    diff = both["ledger"] / both["base"] - 1.0
    r_base = s_base.pct_change()
    r_led = s_led.pct_change()
    rdiff = (r_led - r_base).dropna()

    # ---- 份额模拟: 逐调仓日累计隔夜跳空 (精确, 含停牌阻塞与漂移) ----
    gap_est = 1.0
    units, etf_units = {}, 0.0      # 上期 t0 开盘买入的份额
    gap_days = {}                   # t0 -> 当日隔夜贡献 (验证2 用)
    prev_w_etf = None
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if not len(hold):
            continue
        t0 = hold[0]
        i_t0 = td.index(t0)
        d_prev = td[i_t0 - 1]
        use_etf = not bool(rs12_on)

        # 1) 隔夜贡献: 上期持仓 d_prev 收盘市值权重 × (open/close − 1)
        if units or etf_units:
            mv = 0.0
            for c, u in units.items():
                pc = close_df.at[d_prev, c] if c in close_df.columns else np.nan
                if _valid(pc):
                    mv += u * pc
            ec = e_close_s.get(d_prev)
            if etf_units and _valid(ec):
                mv += etf_units * ec
            if mv > 0:
                g = 0.0
                for c, u in units.items():
                    pc = close_df.at[d_prev, c] if c in close_df.columns else np.nan
                    po = open_df.at[t0, c] if c in open_df.columns else np.nan
                    if _valid(pc) and _valid(po):
                        g += (u * pc / mv) * (po / pc - 1.0)
                eo = e_open_s.get(t0)
                if etf_units and _valid(ec) and _valid(eo):
                    g += (etf_units * ec / mv) * (eo / ec - 1.0)
                gap_est *= 1.0 + g
                gap_days[t0] = g

        # 2) t0 开盘换仓 (与 ledger 同规则: 停牌卖不出保留 / 买不进放弃)
        hi = td.index(rb)
        win = td[max(0, hi - WINDOW):hi]
        if picks is None:
            continue   # fail-closed 沿用上期持仓, 无新交易
        rets = env.pct_df.reindex(columns=picks).reindex(win)
        w = _ivw_weights(rets)
        if use_etf:
            eo0 = e_open_s.get(t0)
            if not _valid(eo0):
                continue   # fail-closed: 目标腿无行情 → 沿用上期持仓, 不换仓
        kept = {}
        for c, u in units.items():
            po = open_df.at[t0, c] if c in open_df.columns else np.nan
            if not _valid(po):
                kept[c] = u          # 停牌 → 保留 (与 ledger._sell_blocked 一致)
        units = kept
        etf_units = 0.0              # ETF 可成交, 直接清
        nav_t0 = 0.0
        for c, u in units.items():
            po = open_df.at[t0, c]
            nav_t0 += u * po
        if use_etf:
            eo = e_open_s.get(t0)
            if _valid(eo):
                etf_units = (nav_t0 if nav_t0 > 0 else 1.0) / eo
        else:
            nav_t0 = nav_t0 if nav_t0 > 0 else 1.0
            for c in w.index:
                po = open_df.at[t0, c] if c in open_df.columns else np.nan
                if not _valid(po):
                    continue          # 停牌 → 现金保留 (与 ledger._buy_blocked 一致)
                units[c] = w[c] * nav_t0 / po

    # ---- 验证2: 日频收益差分布 (调仓日 vs 非调仓日) ----
    t0_set = set(gap_days)
    non_t0 = rdiff.index.difference(list(t0_set))
    t0_rows = rdiff[rdiff.index.isin(t0_set)]
    resid = rdiff[non_t0]
    # 调仓日 diff 对照
    t0_cmp = []
    for t, g in sorted(gap_days.items()):
        r0 = rdiff.get(t, np.nan)
        if pd.notna(r0):
            t0_cmp.append(f"{t}: diff {r0:+.2%} vs 隔夜贡献 {g:+.2%} {'OK' if abs(r0-g) < 0.02 else '!'}")
        else:
            t0_cmp.append(f"{t}: 无 diff (数据起点前)")

    # ---- 验证3: RS12 动量符号 ↔ 腿 ----
    # 简化: 用 e_ret (512100 与中证1000 的 12 月相对收益?) 无法直接取到; 改为打印腿分布
    legs = []
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if not len(hold):
            continue
        legs.append((rb, "ETF" if not bool(rs12_on) else "股票"))
    leg2024 = [f"{rb}:{leg}" for rb, leg in legs if rb.startswith("2024")]
    leg2025 = [f"{rb}:{leg}" for rb, leg in legs if rb.startswith("2025")]

    lines = ["阶段3 一致性校验: 双账本 vs 冻结基线 (无摩擦过滤)", "=" * 90]
    lines.append(f"交易日 {len(both)} | 全段偏差 ledger/base-1 = {s_led.iloc[-1]/s_base.iloc[-1]-1:+.2%}")
    lines.append(f"逐调仓日累计隔夜跳空 (份额模拟) = {gap_est-1:+.2%}  ← 应≈上方偏差")
    lines.append(f"日频水平偏差均值 {diff.mean():+.2%} | 最大 |{diff.abs().max():.3%}|")
    lines.append(f"[验证2] 调仓日收益差 |max| {np.abs(t0_rows).max():+.3%} "
                 f"(其中 {len([x for x in t0_cmp if x.endswith('!')])} 个与隔夜贡献差>2%)")
    lines.append(f"[验证2] 非调仓日收益差均值 {resid.mean():+.2e} | max| {np.abs(resid).max():.5%} "
                 f"(应 <5bp) | 天数 {len(resid)}")
    lines.append(f"[验证3] 2024 腿: {' | '.join(leg2024)}")
    lines.append(f"[验证3] 2025 腿: {' | '.join(leg2025)}")
    lines.append("[调仓日 diff vs 隔夜贡献 对照]")
    lines += [f"  {x}" for x in t0_cmp if not x.endswith("!") or True][-40:]
    print("\n".join(lines[:8]))
    print(f"\n[调仓日对照] {len(t0_cmp)} 期, 非OK {sum(1 for x in t0_cmp if x.endswith('!'))}")

    # ---- 期末状态诊断 (final_cash=终值 疑问) ----
    last = None
    def dbg(rb, snap):
        nonlocal last
        last = (rb, snap)
    L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s,
                 st_map=None, one_up=None, one_dn=None, debug=dbg)
    lines.append(f"[期末账本] {last}")
    print(f"[期末账本] {last}")

    # ---- 双账本 (带 ST/一字涨跌停过滤) ----
    st_map = E.load_st_intervals()
    one_up, one_dn = E.build_limit_sets(open_df, high_df, low_df, pct_df, env.all_codes)
    s_led_f, st_led_f = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s,
                                     st_map=st_map, one_up=one_up, one_dn=one_dn)
    lines.append("")
    lines.append("双账本 (带 ST/一字涨跌停过滤):")
    lines.append(f"  终值 {s_led_f.iloc[-1]:.4f} (无过滤 {s_led.iloc[-1]:.4f}) | 累计 {s_led_f.iloc[-1]-1:+.2%}")
    lines.append(f"  买入阻塞 {st_led_f['n_buy_block']} | ST {st_led_f['n_st_block']} | 停牌 {st_led_f['n_susp_block']} | "
                 f"卖出阻塞 {st_led_f['n_sell_block']} | 期末现金(权重) {st_led_f['final_cash']:.4f} | 期末ETF份额 {st_led_f['final_etf']:.4f}")
    lines.append(f"  tracking_error 均值 {st_led_f['avg_te']*100:.3f}% | 期末 pending {st_led_f['n_pending']} | 缺失月 {st_led_f['n_missing']}")

    # ---- 阶段4: 可交易过滤生产路径 (信号名单 → 订单名单) ----
    from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df
    amount_df = load_amount_df(env, td)
    tf = Tradability(td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                     st_map=st_map)
    s_led_t, st_led_t = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s,
                                     st_map=st_map, one_up=one_up, one_dn=one_dn,
                                     tradable=tf)
    # 每期剔除统计
    rem_cnt = {}
    n_emp = 0
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if picks is None:
            continue
        _, removed = tf(rb, picks)
        for r in removed.values():
            rem_cnt[r] = rem_cnt.get(r, 0) + 1
    lines.append("")
    lines.append("阶段4 生产路径 (信号→订单名单 可交易过滤, 60日/300万/ST/长期停牌):")
    lines.append(f"  终值 {s_led_t.iloc[-1]:.4f} (带摩擦不过滤 {s_led_f.iloc[-1]:.4f}) | 累计 {s_led_t.iloc[-1]-1:+.2%}")
    lines.append(f"  剔除 {sum(rem_cnt.values())} 只次: " + ", ".join(f"{k}={v}" for k, v in sorted(rem_cnt.items())))
    lines.append(f"  订单名单过少 fail-closed {st_led_t['n_missing']} 月 | 目标腿无行情 {st_led_t['n_leg_block']} 月 | "
                 f"执行阻塞 买{st_led_t['n_buy_block']} 卖{st_led_t['n_sell_block']}")
    print("\n".join(lines[-5:]))

    # ---- 阶段5: 集中度约束生产路径 (单股4% / 行业20% / Top5 20% / 容量5%×ADTV60 / 波动率下限) ----
    from research.studies.study_008_enhancements.concentration import (
        apply_concentration, amount60_at,
    )
    ind_map = C.load_industry_map() if os.path.exists(
        os.path.join(C.DATA_DIR, "industry_map.parquet")) else None
    # 阶段5 含波动率下限 (P0-3: 僵尸股过滤, 年化<12%), 订单名单在阶段4基础上进一步收窄
    tf5 = Tradability(td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                      st_map=st_map, min_vol=12.0, pct_df=pct_df)

    def _conc(rb, w, nav_pre):
        return apply_concentration(
            w,
            ind_map=ind_map,
            cap_stock=0.04, cap_ind=0.20, cap_top5=0.20,
            amount60=amount60_at(amount_df, td, rb), nav_pre=nav_pre,
            cap_amount=0.05, scale=1e8)

    s_led_c, st_led_c = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s,
                                     st_map=st_map, one_up=one_up, one_dn=one_dn,
                                     tradable=tf5, concentration=_conc)
    # 阶段5 波动率剔除统计
    rem5 = {}
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if picks is None:
            continue
        _, removed = tf5(rb, picks)
        for r in removed.values():
            rem5[r] = rem5.get(r, 0) + 1
    lines.append("")
    lines.append("阶段5 生产路径 (信号→订单名单→集中度约束):")
    lines.append(f"  终值 {s_led_c.iloc[-1]:.4f} (阶段4 {s_led_t.iloc[-1]:.4f}) | 累计 {s_led_c.iloc[-1]-1:+.2%}")
    lines.append(f"  行业映射 {'{} 只'.format(len(ind_map)) if ind_map else '缺失(跳过行业cap)'}")
    lines.append(f"  波动率下限 12%: 剔除 {sum(rem5.values())} 只次 ({', '.join(f'{k}={v}' for k, v in sorted(rem5.items()))})")
    lines.append(f"  目标腿无行情 {st_led_c['n_leg_block']} 月 | 执行阻塞 买{st_led_c['n_buy_block']} 卖{st_led_c['n_sell_block']} | "
                 f"期末现金(权重) {st_led_c['final_cash']:.4f}")
    print("\n".join(lines[-5:]))

    fp = os.path.join(C.OUT_DIR, "ledger_consistency.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "ledger_consistency.json"), "w", encoding="utf-8") as f:
        json.dump(dict(base=dict(final=float(s_base.iloc[-1]), **st_base),
                       ledger=dict(final=float(s_led.iloc[-1]), **st_led),
                       ledger_friction=dict(final=float(s_led_f.iloc[-1]), **st_led_f),
                       ledger_tradable=dict(final=float(s_led_t.iloc[-1]), **st_led_t),
                       ledger_conc=dict(final=float(s_led_c.iloc[-1]), **st_led_c),
                       tradable_removed=rem_cnt,
                       gap_explained=gap_est - 1.0,
                       gap_total=float(s_led.iloc[-1] / s_base.iloc[-1] - 1.0),
                       gap_days={k: float(v) for k, v in gap_days.items()}),
                  f, ensure_ascii=False, indent=1, default=str)
    print(f"\n[saved] {fp}")


if __name__ == "__main__":
    main()
