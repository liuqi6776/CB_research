# -*- coding: utf-8 -*-
"""阶段3 (P0-2): 双账本持仓簿 + 订单状态机 — 可成交执行层

与 engine.run_backtest (冻结基线 v1.0.0 口径) 平行运行, 生产改造路径:
  - target_weight : 月末信号目标权重 (IVW120, 经 ST/涨跌停/停牌过滤)
  - actual_weight : 实际持仓权重 (T日开盘成交, 份额持有, 期间随价格漂移)
  - cash_weight   : 现金权重 (买不进 → 现金保留, 不重新分配给其他股票)
  - pending_orders: 未成交买单跨期顺延 (blocked_reason 记录; 目标退出后 DROPPED 作废)
  - tracking_error: 目标 vs 实际 ½Σ|Δw| (含现金腿), 每次调仓后计算
  - 订单状态机    : NEW → FILLED / BLOCKED(顺延) / DROPPED(目标退出)
核心规则 (用户验收清单 P0-2):
  "买不进: 现金保留; 卖不出: 旧仓继续承担收益风险, 不现金化"
成本沿用阶段2 (P0-1) 分项费率, 仅对"实际执行"的换手计费 (未成交部分零成本)。

账本恒等: NAV = cash + Σ(份额×价格) [+ etf份额×etf价格]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements.direction2_hrp import _ivw_weights, WINDOW
from research.studies.study_008_enhancements.engine import is_st
from research.studies.study_008_enhancements.risk_control_real import (
    ETF_FEE, stock_buy_fee, stock_sell_fee,
)

REASON_ST = "ST"
REASON_UP = "LIMIT_UP"      # 一字涨停不可买
REASON_DN = "LIMIT_DOWN"    # 一字跌停不可卖
REASON_SUSP = "SUSPEND"     # 停牌


class Order:
    """订单状态机: NEW → FILLED / BLOCKED(顺延) / DROPPED(目标退出)"""
    __slots__ = ("code", "side", "target_w", "reason", "month", "status")

    def __init__(self, code, side, target_w, reason, month):
        self.code = code
        self.side = side
        self.target_w = target_w
        self.reason = reason
        self.month = month
        self.status = "NEW"


class PortfolioBook:
    """双账本持仓簿 (权重单位; NAV = cash + Σ份额×价格 + etf份额×etf价格)

    份额口径: units[code] = 建仓目标权重 / 建仓开盘价 → 价值贡献 = 份额×现价 = 权重漂移
    """

    def __init__(self):
        self.units = {}        # code -> 份额
        self.etf_units = 0.0   # 512100 份额
        self.cash = 0.0        # 现金权重 (残余)
        self.pending = {}      # code -> Order (未成交买单, 顺延)
        self.fees = 0.0
        # 事件计数
        self.n_buy_block = 0
        self.n_sell_block = 0
        self.n_st_block = 0
        self.n_susp_block = 0

    # ---------- 定价 ----------
    def _open(self, open_df, t0, code):
        try:
            v = open_df.at[t0, code]
        except KeyError:
            return np.nan
        return float(v) if pd.notna(v) and v > 0 else np.nan

    def _close(self, close_df, t0, code):
        try:
            v = close_df.at[t0, code]
        except KeyError:
            return np.nan
        return float(v) if pd.notna(v) else np.nan

    def _buy_blocked(self, code, rb, t0, st_map, one_up, p_open):
        if st_map and is_st(st_map, code, rb):
            return REASON_ST
        if one_up and (code, t0) in one_up:
            return REASON_UP
        if pd.isna(p_open):
            return REASON_SUSP
        return None

    def _sell_blocked(self, code, t0, one_dn, p_open):
        if pd.isna(p_open):
            return REASON_SUSP
        if one_dn and (code, t0) in one_dn:
            return REASON_DN
        return None

    # ---------- 调仓前市值 (供阶段5 容量约束等) ----------
    def market_value(self, t, open_df, etf_open_s):
        """t 开盘时点持仓市值 (调仓前, NAV_pre)"""
        v = self.cash
        for c, u in self.units.items():
            p = self._open(open_df, t, c)
            if pd.notna(p):
                v += u * p
        if self.etf_units:
            eo = float(etf_open_s.get(t, np.nan)) if etf_open_s is not None else np.nan
            if pd.notna(eo):
                v += self.etf_units * eo
        return v

    # ---------- 调仓执行 (T 日开盘) ----------
    def rebalance(self, rb, t0, w, use_etf, open_df, close_df, etf_open_s,
                  st_map=None, one_up=None, one_dn=None, cost_on=True, leg="etf"):
        """按目标权重 w 在 t0 开盘调仓. 返回 (tracking_error, cost, n_block_new)
        买不进 → 现金保留; 卖不出 → 旧仓继续持有 (不现金化)."""
        eo = float(etf_open_s.get(t0, np.nan)) if (etf_open_s is not None
                                                    and (use_etf or self.etf_units)) else np.nan
        p_ref = {}
        for c in list(self.units):
            p = self._open(open_df, t0, c)
            p_ref[c] = p if pd.notna(p) else self._close(close_df, t0, c)
        # NAV 前值 (空账户首期 = 1.0)
        nav_pre = self.cash + sum(u * (p_ref[c] if pd.notna(p_ref[c]) else 0.0)
                                  for c, u in self.units.items())
        if self.etf_units and pd.notna(eo):
            nav_pre += self.etf_units * eo
        if nav_pre <= 0:
            nav_pre = 1.0
        targets = set(w.index) if not use_etf else set()
        buy_vol = sell_vol = buy_etf_vol = sell_etf_vol = 0.0
        n_new_block = 0
        p_now = dict(p_ref)  # 持仓计价: 既有持仓用参考价, 新买入用开盘价

        # ---- 1) 卖出: 目标外清仓 / 目标内超配削减 (仅可执行部分) ----
        for c in list(self.units.keys()):
            p = p_ref[c]
            cur_w = (self.units[c] * p / nav_pre) if pd.notna(p) else 0.0
            w_c = w.get(c, 0.0) if not use_etf else 0.0
            if w_c >= cur_w - 1e-12 or cur_w <= 1e-12:
                continue
            po = self._open(open_df, t0, c)
            reason = self._sell_blocked(c, t0, one_dn, po)
            if reason is not None:
                # 旧仓继续承担收益风险, 不现金化
                self.n_sell_block += 1
                continue
            delta = cur_w - w_c
            self.units[c] = w_c * nav_pre / po if w_c > 1e-12 else 0.0
            if self.units[c] <= 1e-12:
                del self.units[c]
            sell_vol += delta

        # ---- 2) 买入 / 回补 (含 pending 顺延自然重试) ----
        if not use_etf:
            for c in w.index:
                po = self._open(open_df, t0, c)
                cur_w = 0.0
                if c in self.units:
                    pc = p_ref.get(c, np.nan)
                    cur_w = self.units[c] * pc / nav_pre if pd.notna(pc) else 0.0
                w_c = float(w[c])
                if w_c <= cur_w + 1e-12:
                    continue
                reason = self._buy_blocked(c, rb, t0, st_map, one_up, po)
                if reason is not None:
                    # 买不进 → 现金保留, 不重新分配; 订单顺延
                    if reason == REASON_ST:
                        self.n_st_block += 1
                    elif reason == REASON_SUSP:
                        self.n_susp_block += 1
                    else:
                        self.n_buy_block += 1
                    n_new_block += 1
                    if c in self.pending:
                        self.pending[c].status = "BLOCKED"
                    else:
                        self.pending[c] = Order(c, "BUY", w_c, reason, rb)
                        self.pending[c].status = "BLOCKED"
                    continue
                delta = w_c - cur_w
                self.units[c] = w_c * nav_pre / po
                p_now[c] = po
                buy_vol += delta
                if c in self.pending:
                    self.pending.pop(c).status = "FILLED"
        else:
            # 股票→防御腿: 用当前剩余现金买 ETF (现金腿 leg='cash' 不买, 卖不出的旧仓仍保留)
            if self.etf_units == 0 and pd.notna(eo) and leg != "cash":
                avail = nav_pre - sum(u * (p_ref[c] if pd.notna(p_ref[c]) else 0.0)
                                      for c, u in self.units.items())
                if avail > 0:
                    self.etf_units = avail / eo
                    buy_etf_vol = avail / nav_pre
        # ETF→股票 / ETF→ETF
        if not use_etf and self.etf_units > 0 and pd.notna(eo):
            sell_etf_vol = self.etf_units * eo / nav_pre
            self.etf_units = 0.0

        # ---- 3) 现金残余 + 成本 (仅计实际执行) ----
        val = 0.0
        for c, u in self.units.items():
            pc = p_now.get(c, np.nan)
            val += u * pc if pd.notna(pc) else 0.0
        if self.etf_units:
            val += self.etf_units * eo if pd.notna(eo) else 0.0
        self.cash = nav_pre - val
        cost = 0.0
        if cost_on:
            cost = (buy_vol * stock_buy_fee(rb) + sell_vol * stock_sell_fee(rb)
                    + (buy_etf_vol + sell_etf_vol) * ETF_FEE)
            self.cash -= cost * nav_pre
            self.fees += cost * nav_pre

        # ---- 4) 订单顺延清理: 目标已退出 → DROPPED ----
        for code in list(self.pending):
            if code not in targets:
                self.pending.pop(code).status = "DROPPED"

        # ---- 5) tracking_error: 目标 vs 实际 ½Σ|Δw| (含现金腿) ----
        nav_post = self.cash + val
        actual = {}
        for c, u in self.units.items():
            pc = p_now.get(c, np.nan)
            if pd.notna(pc) and nav_post > 0:
                actual[c] = u * pc / nav_post
        if self.etf_units and nav_post > 0:
            actual["ETF"] = self.etf_units * eo / nav_post
        if nav_post > 0:
            actual["CASH"] = self.cash / nav_post
        target_t = dict(w) if not use_etf else {"ETF": 1.0}
        target_t["CASH"] = 0.0
        te = 0.5 * sum(abs(actual.get(c, 0.0) - target_t.get(c, 0.0))
                       for c in set(actual) | set(target_t))
        return te, cost, n_new_block

    # ---------- 持有期日频盯市 (份额不变, 权重随价格漂移) ----------
    def mark_days(self, hold, close_df, etf_close_s, navs):
        codes = [c for c in self.units if c in close_df.columns]
        cl = close_df[codes].reindex(hold).ffill().bfill() if codes else None
        for t in hold:
            v = self.cash
            if self.etf_units:
                e = etf_close_s.get(t)
                if pd.notna(e):
                    v += self.etf_units * e
            if cl is not None and t in cl.index:
                row = cl.loc[t]
                v += float((pd.Series(self.units, index=codes) * row).sum())
            navs[t] = v


def run_ledger(env, td, open_df, close_df, etf_open_s, etf_close_s,
               st_map=None, one_up=None, one_dn=None,
               start=None, end=None, cost_on=True, debug=None,
               tradable=None, concentration=None,
               leg="etf", bond_open_s=None, bond_close_s=None):
    """双账本执行层回测. 返回 (nav Series, stats dict)
    与 engine.run_backtest 平行: 冻结基线口径不动, 生产改造路径走本函数.
    tradable: 可选 callable(rb, picks) -> (order_picks, removed) 阶段4 可交易过滤
      (信号名单 → 订单名单: 剔除 rb 时已知 ST/退市长期停牌/极低流动性).
    concentration: 可选 callable(rb, w, nav_pre) -> w' 阶段5 集中度约束
      (单股/行业/Top5/容量 上限, 对 IVW120 目标权重应用).
    leg: RS12 空头防御腿 (阶段6 对照) — 'etf'(默认, 512100) / 'cash'(持币) /
      'bond'(短债 ETF, 需 bond_open_s/bond_close_s).
    fail-closed (用户验收 P0-4): 信号缺失 / 订单名单为空或过少 / 目标腿无行情
      → 沿用上期可执行持仓, 不主动产生新交易, 零成本, 计数 (阻断告警阶段7日报承载).
    debug: 可选回调 debug(rb, snapshot dict) 每次调仓后调用 (阶段7 日报复用)"""
    book = PortfolioBook()
    navs = {}
    te_list = []
    n_missing = 0
    n_leg_block = 0
    n_trad_removed = 0
    # 盯市价格前向填充: 仅处理数据边界缺失 (如 512100 截至 20260731, 调仓日 20260803 无价)
    etf_close_mark = etf_close_s.ffill()
    bond_close_mark = bond_close_s.ffill() if bond_close_s is not None else None
    def_close_mark = etf_close_mark   # 当前防御腿收盘 (fail-closed 沿用上期腿盯市)
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if start is not None and rb < start:
            continue
        if end is not None and rb > end:
            continue
        if picks is None:
            # fail-closed: 信号缺失 → 沿用上期可执行持仓, 不主动交易 (零成本)
            n_missing += 1
            book.mark_days(hold, close_df, def_close_mark, navs)
            continue
        t0 = hold[0]
        if tradable is not None:
            # 阶段4: 信号名单 → 订单名单 (rb 时可观测过滤)
            picks, removed = tradable(rb, picks)
            n_trad_removed += len(removed)
            if len(picks) < 10:
                # fail-closed: 订单名单过少 → 沿用上期持仓 (不静默退化)
                n_missing += 1
                book.mark_days(hold, close_df, def_close_mark, navs)
                continue
        hi = td.index(rb)
        win = td[max(0, hi - WINDOW):hi]
        rets = env.pct_df.reindex(columns=picks).reindex(win)
        w = _ivw_weights(rets)
        use_etf = not bool(rs12_on)
        # 防御腿价格序列: 全局固定 (阶段6 leg 配置), 不随 RS12 切换 —
        # 卖出持仓防御份额必须按该腿自身价格 (bond 份额不能按 512100 计价)
        if leg == "bond" and bond_open_s is not None:
            def_open_s, def_close_s = bond_open_s, bond_close_s
        elif leg == "cash":
            def_open_s, def_close_s = None, None
        else:
            def_open_s, def_close_s = etf_open_s, etf_close_s
        if use_etf and def_open_s is not None:
            eo = float(def_open_s.get(t0, np.nan))
            if pd.isna(eo):
                # fail-closed: 目标腿无行情 → 沿用上期持仓, 不调仓
                # (旧代码会卖出股票后买不进 ETF → 全现金化, 违反 P0-4)
                n_leg_block += 1
                book.mark_days(hold, close_df, def_close_mark, navs)
                continue
        if concentration is not None:
            # 阶段5: 集中度约束 (单股/行业/Top5/容量), 需调仓前 NAV (防御腿按当期价格)
            nav_pre = book.market_value(t0, open_df, def_open_s)
            if nav_pre > 0:
                w = concentration(rb, w, nav_pre)
        te, cost, n_block = book.rebalance(rb, t0, w, use_etf, open_df, close_df,
                                           def_open_s, st_map, one_up, one_dn,
                                           cost_on, leg=leg)
        te_list.append(te)
        # 更新防御腿盯市序列 (下期 fail-closed 沿用本腿)
        def_close_mark = def_close_s.ffill() if (use_etf and def_close_s is not None) \
            else etf_close_mark
        book.mark_days(hold, close_df, def_close_mark, navs)
        if debug is not None:
            debug(rb, dict(t0=t0, use_etf=use_etf, n_units=len(book.units),
                           cash=book.cash, etf=book.etf_units, fees=book.fees,
                           n_pending=len(book.pending), te=te))
    s = pd.Series(navs).sort_index()
    stats = dict(fees=book.fees, avg_te=float(np.mean(te_list)) if te_list else 0.0,
                 n_buy_block=book.n_buy_block, n_sell_block=book.n_sell_block,
                 n_st_block=book.n_st_block, n_susp_block=book.n_susp_block,
                 n_pending=len(book.pending), n_missing=n_missing,
                 n_leg_block=n_leg_block, n_trad_removed=n_trad_removed,
                 final_cash=book.cash, final_etf=book.etf_units, n_units=len(book.units))
    return s, stats
