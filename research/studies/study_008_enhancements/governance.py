# -*- coding: utf-8 -*-
"""阶段7 (用户验收 P0-4 治理阈值): 账户治理层

峰值回撤分级状态机 + 实盘日报字段生成器.
治理阈值 (生产运维动作建议, 基线冻结不改回测路径, 治理层为叠加观测):
  L0 正常  : 无动作
  L1 -15%  : 黄灯 — 风控复核, 禁止对亏损标的加仓
  L2 -20%  : 橙灯 — 暂停新开仓 (只减不加), 审查 fail-closed 计数
  L3 -25%  : 红灯 — 强制降至半仓/暂停交易, 人工接管
"""
import json
import os

import numpy as np
import pandas as pd


class Governance:
    """账户治理状态机: 按峰值回撤 (1 - nav/peak) 分级触发"""

    LEVELS = (0.15, 0.20, 0.25)          # L1/L2/L3 阈值
    ACTIONS = {0: "正常", 1: "黄灯-风控复核/禁止加仓",
               2: "橙灯-暂停新开仓(只减不加)", 3: "红灯-强制半仓/人工接管"}

    def __init__(self):
        self.peak = None
        self.state = 0
        self.events = []                  # [(date, level, dd, peak)]
        self.daily = []                   # [(date, nav, dd, level)]

    def level_of(self, dd):
        if dd >= self.LEVELS[2]:
            return 3
        if dd >= self.LEVELS[1]:
            return 2
        if dd >= self.LEVELS[0]:
            return 1
        return 0

    def on_nav(self, t, nav):
        """每日 NAV 输入, 返回 (level, dd)"""
        if self.peak is None or nav > self.peak:
            self.peak = nav
        dd = self.peak / nav - 1.0 if nav > 0 else 1.0
        lv = self.level_of(dd)
        if lv != self.state:
            self.events.append((str(t), lv, float(dd), float(self.peak)))
            self.state = lv
        self.daily.append((str(t), float(nav), float(dd), lv))
        return lv, dd

    def summary(self):
        d = pd.DataFrame(self.daily, columns=["date", "nav", "dd", "level"])
        out = {"n_days": len(d),
               "peak_nav": float(self.peak) if self.peak is not None else None}
        for lv in (1, 2, 3):
            sub = d[d.level == lv]
            out[f"L{lv}_days"] = int(len(sub))
            out[f"L{lv}_ratio"] = float(len(sub) / len(d)) if len(d) else 0.0
            if len(self.events) and any(e[1] == lv for e in self.events):
                first = next(e for e in self.events if e[1] == lv)
                out[f"L{lv}_first"] = dict(date=first[0], dd=first[2])
        return out


def daily_report(navs, stats, gov, date, holdings=None):
    """实盘日报字段 (阶段7): 由账本 NAV + stats + 治理状态机拼装"""
    dd = gov.daily[-1][2] if gov.daily else 0.0
    lv = gov.daily[-1][3] if gov.daily else 0
    rep = {
        "date": str(date),
        "nav": float(navs.loc[date]) if date in navs.index else float(navs.iloc[-1]),
        "peak_nav": float(gov.peak) if gov.peak is not None else None,
        "drawdown_pct": round(dd * 100, 2),
        "gov_level": lv,
        "gov_action": Governance.ACTIONS[lv],
        # 账户结构
        "cash": round(float(stats.get("final_cash", 0.0)), 4),
        "etf_units": round(float(stats.get("final_etf", 0.0)), 3),
        "n_stocks": int(stats.get("n_units", 0)),
        "n_pending_orders": int(stats.get("n_pending", 0)),
        # 风控计数 (P0-4 阻断告警承载)
        "fail_closed_days": int(stats.get("n_missing", 0)) + int(stats.get("n_leg_block", 0)),
        "blocked_buy": int(stats.get("n_buy_block", 0)),
        "blocked_sell": int(stats.get("n_sell_block", 0)),
        "blocked_st": int(stats.get("n_st_block", 0)),
        "blocked_susp": int(stats.get("n_susp_block", 0)),
        "fees_total": round(float(stats.get("fees", 0.0)), 2),
        "avg_te_pct": round(float(stats.get("avg_te", 0.0)) * 100, 2),
        # 告警: 并发阻断 (有未成交订单或当日治理级>0) → 1
        "alert": int(int(stats.get("n_pending", 0)) > 0 or lv > 0),
    }
    if holdings is not None:
        rep["holdings"] = holdings
    return rep


def main():
    """示例: 对阶段5 生产路径 NAV 运行治理状态机 + 输出日报"""
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    from research.studies.study_008_enhancements import common as C
    from research.studies.study_008_enhancements import engine as E
    from research.studies.study_008_enhancements import ledger as L
    from research.studies.study_008_enhancements.concentration import (
        apply_concentration, amount60_at,
    )
    from research.studies.study_008_enhancements.tradability import (
        Tradability, load_amount_df,
    )

    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)
    etf = C.load_idx("512100.SH")
    e_open_s = etf["open"].astype(float)
    e_close_s = etf["close"].astype(float)
    st_map = E.load_st_intervals()
    one_up, one_dn = E.build_limit_sets(open_df, high_df, low_df, pct_df, env.all_codes)
    amount_df = load_amount_df(env, td)
    tf5 = Tradability(td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                      st_map=st_map, min_vol=12.0, pct_df=pct_df)
    ind_map = C.load_industry_map()

    def _conc(rb, w, nav_pre):
        return apply_concentration(w, ind_map=ind_map, cap_stock=0.04, cap_ind=0.20,
                                   cap_top5=0.20,
                                   amount60=amount60_at(amount_df, td, rb),
                                   nav_pre=nav_pre, cap_amount=0.05, scale=1e8)

    s, st = L.run_ledger(env, td, open_df, close_df, e_open_s, e_close_s,
                         st_map=st_map, one_up=one_up, one_dn=one_dn,
                         tradable=tf5, concentration=_conc)
    gov = Governance()
    for t, nav in s.items():
        gov.on_nav(t, nav)
    sum_ = gov.summary()
    print("阶段7 账户治理: 阶段5 生产路径 (终值 %.4f)" % s.iloc[-1])
    print(json.dumps(sum_, ensure_ascii=False, indent=1))
    print("\n治理触发事件 (日期, 级别, 回撤, 峰值):")
    for e in gov.events:
        print("  %s -> L%d  dd=%.1f%%  peak=%.4f" % (e[0], e[1], e[2] * 100, e[3]))
    rep = daily_report(s, st, gov, s.index[-1])
    print("\n实盘日报示例 (最后一日 %s):" % rep["date"])
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    fp = os.path.join(C.OUT_DIR, "governance_daily_report.json")
    with open(fp, "w", encoding="utf-8") as f:
        json.dump({"summary": sum_, "events": gov.events,
                   "last_daily": rep}, f, ensure_ascii=False, indent=1)
    print("\n[saved] %s" % fp)


if __name__ == "__main__":
    main()
