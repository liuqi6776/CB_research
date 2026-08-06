# -*- coding: utf-8 -*-
"""P1 统一真实口径引擎 (study_008_enhancements)

在 risk_control_real / risk_control_ddstop 基础上参数化扩展, 供 P1 三个子任务共用:
  - 子区间回测 (start/end, 前段定参 / 后段 OOS 独立样本验证)
  - 单股 cap / 行业簇 cap (审查意见 #10)
  - ST 过滤 / 一字涨停不可买 / 停牌现金化 (残余简化建模)

口径与 risk_control_real v2 一致, 成本按阶段2 (P0-1) 升级:
  日频 MaxDD / T 日开盘成交 / 买入持有漂移 / 资金权重换手成本
    Turnover = ½Σ|w_new − w_old| (资产集合 = 旧股∪新股∪512100∪现金)
    Cost = BuyTurnover×c_buy + SellTurnover×c_sell (股票卖出含印花税, ETF 独立费率)
    + MA20 档位切换 |Δw|×单边10bps (仅风控路径)
  DD 止损层与 risk_control_ddstop 一致 (降仓至 0.5 不清仓 / 谷底回升恢复)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements.direction2_hrp import (
    _hrp_weights, _ivw_weights, WINDOW,
)
from research.studies.study_008_enhancements.risk_control_real import (
    TIER3, TIER5, COST_SINGLE, ETF_FEE, stamp_sell, stock_buy_fee,
    stock_sell_fee, tier_w, cap_weights,
)


def asset_turnover_cost(prev_w, w, prev_etf, use_etf, date):
    """阶段2 (P0-1): 资金权重换手与买卖分项成本.

    资产集合 = 旧股 ∪ 新股 ∪ {512100} ∪ {现金}(全仓模型恒 0, 预留):
      - 股票腿: Turnover = ½Σ|w_new − w_old|, 买卖分侧加总
      - 腿切换 (股票↔ETF): 整腿按 1.0 搬移, 换手 100%
      - Cost = Buy×c_buy + Sell×c_sell, 股票卖出含印花税(按日期分档), ETF 用独立费率
    返回 (buy_stock, sell_stock, buy_etf, sell_etf, cost_total)
    """
    buy_stock = sell_stock = buy_etf = sell_etf = 0.0
    if prev_w is None:
        # 首次建仓: 全仓买入目标资产腿
        if use_etf:
            buy_etf = 1.0
        else:
            buy_stock = float(w.sum())
    elif not prev_etf and not use_etf:
        # 股票→股票: 目标权重差 (新旧股并集上对齐)
        codes = sorted(set(prev_w.index) | set(w.index))
        w_old = prev_w.reindex(codes, fill_value=0.0)
        w_new = w.reindex(codes, fill_value=0.0)
        d = w_new - w_old
        buy_stock = float(d.clip(lower=0.0).sum())
        sell_stock = float((-d).clip(lower=0.0).sum())
    elif prev_etf and use_etf:
        pass  # ETF→ETF: 无换手
    elif prev_etf:  # ETF→股票
        sell_etf = 1.0
        buy_stock = float(w.sum())
    else:  # 股票→ETF
        sell_stock = float(prev_w.sum())
        buy_etf = 1.0
    cost = (buy_stock * stock_buy_fee(date) + sell_stock * stock_sell_fee(date)
            + (buy_etf + sell_etf) * ETF_FEE)
    return buy_stock, sell_stock, buy_etf, sell_etf, cost


def load_prices(env, td):
    """宽表: 个股 open/close/high/low/pct_chg + ETF 隔夜/日内段"""
    stocks, _, _, _, _ = rv.load_panels(td, env.all_codes, None)
    open_df = pd.DataFrame({c: g.sort_index()["open"] for c, g in stocks.items()})
    close_df = pd.DataFrame({c: g.sort_index()["close"] for c, g in stocks.items()})
    high_df = pd.DataFrame({c: g.sort_index()["high"] for c, g in stocks.items()})
    low_df = pd.DataFrame({c: g.sort_index()["low"] for c, g in stocks.items()})
    pct_df = pd.DataFrame({c: g.sort_index()["pct_chg"] for c, g in stocks.items()})
    etf = C.load_idx("512100.SH")
    e_ovn_s = (etf["open"].reindex(td) / etf["pre_close"].reindex(td) - 1.0).fillna(0.0)
    e_intra_s = (etf["close"].reindex(td) / etf["open"].reindex(td) - 1.0).fillna(0.0)
    return open_df, close_df, high_df, low_df, pct_df, e_ovn_s, e_intra_s


def limit_pct(code, date):
    """涨跌停幅度: 创业板注册制 2020-08-24 起 20%, 科创板 20%, 北交所 30%, 主板 10%"""
    if code.startswith(("300", "301")):
        return 0.20 if date >= "20200824" else 0.10
    if code.startswith("688"):
        return 0.20
    if code.startswith(("8", "4")):
        return 0.30
    return 0.10


def build_limit_sets(open_df, high_df, low_df, pct_df, codes):
    """返回 (one_up, one_dn): set of (code, date) — 一字涨停/一字跌停日
    一字板 = 当日 open==high==low (全天无波动封死), 结合涨跌幅确认方向
    """
    codes = [c for c in codes if c in open_df.columns]
    sub_open = open_df[codes]
    sub_high = high_df[codes]
    sub_low = low_df[codes]
    sub_pct = pct_df[codes]
    one = (sub_open == sub_high) & (sub_high == sub_low) & sub_pct.notna()
    one_up, one_dn = set(), set()
    for code in codes:
        lim = sub_pct[code].index.to_series().map(lambda d: limit_pct(code, d))
        up_mask = one[code] & (sub_pct[code] >= lim * 0.98)
        dn_mask = one[code] & (sub_pct[code] <= -lim * 0.98)
        one_up |= {(code, d) for d in sub_pct.index[up_mask.values]}
        one_dn |= {(code, d) for d in sub_pct.index[dn_mask.values]}
    return one_up, one_dn


def load_st_intervals():
    """data/st_history.parquet → {ts_code: [(start, end), ...]} ST 区间"""
    fp = os.path.join(C.DATA_DIR, "st_history.parquet")
    if not os.path.exists(fp):
        return {}
    df = pd.read_parquet(fp)
    df["start_date"] = df["start_date"].astype(str)
    df["end_date"] = df["end_date"].astype(str).replace("None", "99999999")
    out = {}
    for code, g in df[df["name"].str.contains("ST", na=False)].groupby("ts_code"):
        out[str(code)] = sorted(zip(g["start_date"], g["end_date"]))
    return out


def is_st(st_map, code, date):
    for s, e in st_map.get(code, []):
        if s <= date <= e:
            return True
        if s > date:
            break
    return False


def cap_industry(w, ind_map, cap, iters=2):
    """行业簇权重上限: 超限行业内个股等比例压缩 -> 整体归一化 (迭代收敛)"""
    if cap is None or ind_map is None:
        return w
    w = w.copy()
    ind = pd.Series({c: ind_map.get(c, "NA") for c in w.index}, index=w.index)
    for _ in range(iters):
        s = w.groupby(ind).sum()
        over = s[s > cap]
        if over.empty:
            break
        for ind_name in over.index:
            mask = (ind == ind_name).values
            w[mask] *= cap / over[ind_name]
        w = w / w.sum()
    return w


def build_fdf(env, rb, ext_panels=None):
    """调仓日 rb 的原始因子宽表 (行=成分股, 列=因子名); 有效股数 < TOP_N 返回 None.

    score_at / score_layer 共用: 保持与历史 score_at 完全一致的成员筛选口径
    (fwd 有值 + 至少 3 个因子字段, VAL 快照缺失月自动降级)。
    """
    members = rv.load_index_weight(rb)
    if members is None:
        return None
    fvals = {}
    for code in members:
        f1, f2, ft = env.ret_1m.get(code), env.ivol.get(code), env.turn.get(code)
        fr = env.fwd.get(code)
        if fr is None or rb not in fr.index:
            continue
        row = {}
        if f1 is not None and rb in f1.index:
            row["ret_1m"] = f1.loc[rb]
        if f2 is not None and rb in f2.index:
            row["ivol"] = f2.loc[rb]
        if ft is not None and rb in ft.index:
            row["turn"] = ft.loc[rb]
        for name in env.panels:
            p = env.panels[name].get(rb)
            if p is not None and code in p.index:
                v = p.loc[code]
                if np.isfinite(v):
                    row[name] = v
        if ext_panels:
            for name in ext_panels:
                p = ext_panels[name].get(rb)
                if p is not None and code in p.index:
                    v = p.loc[code]
                    if np.isfinite(v):
                        row[name] = v
        if len(row) >= 3:
            fvals[code] = row
    if len(fvals) < rv.TOP_N:
        return None
    return pd.DataFrame(fvals).T


def score_at(env, rb, ext_panels=None):
    """调仓日 rb 的截面打分 (与 common.Env._build_picks 同口径)

    返回全成分股 scored Series (zscore 均值: ret_1m+ivol+turnover_vol_20+VAL
    +ext_panels 可选增强因子, 缺 VAL 月份降级 BASE 3 因子); 数据不足返回 None
    (与 _build_picks 的 None 语义一致, 调用方走 fail-closed)。

    ext_panels: 可选 dict name -> {rb: Series(全成分股截面值)}, 增强因子面板
      (因子方向统一为"高值=好"; 与 VAL 面板同结构)。None 时行为与历史完全一致。
    """
    fdf = build_fdf(env, rb, ext_panels)
    if fdf is None:
        return None
    zdf = fdf.apply(sf.winsorize_series).apply(
        lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
    ext_cols = list(ext_panels.keys()) if ext_panels else []
    cols = [c for c in sf.BASE_COLS + ["VAL"] + ext_cols if c in zdf.columns]
    has = zdf[cols].dropna()
    if len(has) < rv.TOP_N:
        return None
    return has.mean(axis=1)


def run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                 e_ovn_s, e_intra_s,
                 use_hrp=True, use_ma20=True, tier=TIER5,
                 dd_stop=None, dd_floor=None, stop_w=0.5, floor_w=0.0, recov=None,
                 cap=None, cap_ind=None, ind_map=None,
                 st_map=None, limit_sets=None,
                 start=None, end=None, cost_on=True,
                 use_std_hrp=False,
                 tradable=None, concentration=None,
                 leg="etf", bond_open_s=None, bond_close_s=None,
                 score_thr=None, ext_panels=None, score_fn=None):
    """统一真实口径回测. 返回 (nav Series, stats dict)

    use_hrp=True: 基线权重 IVW120 (逆波动, _ivw_weights)
    use_std_hrp=True: 实验分支标准 HRP (层次聚类+递归二分, _hrp_weights), 不覆盖基线
    tradable: 可选 callable(rb, picks) -> (order_picks, removed) 阶段4 可交易过滤
      (信号名单 → 订单名单; 剔除后 <10 只 → fail-closed 沿用上期持仓)
    concentration: 可选 callable(rb, w, nav_pre) -> w' 阶段5 集中度约束
      (单股/行业/Top5/容量 上限, 对 IVW120 目标权重应用; nav_pre=调仓前净值)
    leg: RS12 空头防御腿 (阶段6 对照) — 'etf'(默认, 512100) / 'bond'(短债 ETF,
      需 bond_open_s/bond_close_s; 空头期持 511260, 收益=隔夜×日内)
    """
    one_up, one_dn = (limit_sets if limit_sets is not None else (set(), set()))
    # 阶段6 防御腿: bond 腿收益序列 (隔夜=今开/昨收-1, 日内=今收/今开-1)
    if leg == "bond" and bond_open_s is not None:
        b_ovn = (bond_open_s / bond_close_s.shift(1) - 1.0).fillna(0.0)
        b_intra = (bond_close_s / bond_open_s - 1.0).fillna(0.0)
    else:
        b_ovn = b_intra = None
    nav = 1.0
    peak = 1.0
    navs = {}
    w_prev = 1.0
    prev_picks, prev_etf = None, False
    prev_w = None
    total_switch = 0.0
    total_turn = 0.0
    total_buy = 0.0
    total_sell = 0.0
    n_stop = n_floor = n_recov = 0
    in_dd = False
    nav_low = 1.0
    n_buy_block = n_st_block = n_suspend = n_missing = n_trad_removed = 0   # 摩擦计数
    n_selected = []      # 分数阈值模式: 每期信号选股数量 (fail-closed 沿用前)
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if start is not None and rb < start:
            continue
        if end is not None and rb > end:
            continue
        # ---- 分数阈值选股 (实验分支): 等权买入全部 score >= score_thr 的成分股 ----
        if score_thr is not None:
            scored = (score_fn(env, rb) if score_fn is not None
                      else score_at(env, rb, ext_panels))
            if scored is None:
                picks = None                    # 数据不足 → 走下方 fail-closed
                n_selected.append(0)
            else:
                picks = scored[scored >= score_thr].index.tolist()
                n_selected.append(len(picks))
        # ---- fail-closed: 数据缺失(无有效新名单)或订单名单过少时沿用上一期可执行持仓 ----
        carried = False
        if picks is None:
            if prev_picks is None:
                continue  # 从未建仓, 缺失月保持现金
            picks = prev_picks
            comb = env.pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
            e_ret = env.etf_ret.reindex(hold).fillna(0.0)
            carried = True
            n_missing += 1
            print(f"[warn] {rb} 因子/名单缺失, fail-closed 沿用上期持仓 {len(prev_picks)} 只 (累计 {n_missing} 月)", flush=True)
        elif tradable is not None:
            # 阶段4: 信号名单 → 订单名单 (rb 时可观测过滤: ST/退市长期停牌/低流动性)
            picks, removed = tradable(rb, picks)
            n_trad_removed += len(removed)
            if len(picks) < 10:
                # fail-closed: 订单名单过少 → 沿用上期持仓 (不静默退化)
                if prev_picks is None:
                    continue
                picks = prev_picks
                comb = env.pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
                e_ret = env.etf_ret.reindex(hold).fillna(0.0)
                carried = True
                n_missing += 1
                print(f"[warn] {rb} 订单名单 {len(prev_picks)} 只过少, fail-closed 沿用 (累计 {n_missing} 月)", flush=True)
        hi = td.index(rb)
        win = td[max(0, hi - WINDOW):hi]
        if carried and prev_w is not None:
            # fail-closed 沿用月: 保持上期权重不动 (不重新平衡)
            w = prev_w.copy()
        else:
            rets = env.pct_df.reindex(columns=picks).reindex(win)
            if use_std_hrp:
                w = _hrp_weights(rets)
            elif use_hrp:
                w = _ivw_weights(rets)
            else:
                w = pd.Series(1.0 / len(picks), index=picks)
            w = cap_weights(w, cap)
            w = cap_industry(w, ind_map, cap_ind)
            if concentration is not None:
                # 阶段5: 集中度约束 (单股/行业/Top5/容量), nav=调仓前净值
                w = concentration(rb, w, nav)
        # ---- 买入池过滤 (ST / 一字涨停不可买 / 停牌) ----
        t0 = hold[0]
        blocked = set()
        if st_map:
            for c in w.index:
                if is_st(st_map, c, rb):
                    blocked.add(c)
                    n_st_block += 1
        if one_up:
            for c in w.index:
                if (c, t0) in one_up:
                    blocked.add(c)
                    n_buy_block += 1
        if blocked:
            w = w.drop(index=[c for c in blocked if c in w.index])
            if w.sum() > 0:
                w = w / w.sum()
            else:
                continue  # 全部不可买, 该月空仓 (极端)
        use_etf = not rs12_on
        if carried:
            # fail-closed 沿用月: 资产腿与上期一致, 不产生换手
            use_etf = prev_etf
        # ---- 阶段2: 资金权重换手成本 (调仓日一次性计) ----
        buy_stock, sell_stock, buy_etf, sell_etf, cost_total = asset_turnover_cost(
            prev_w, w, prev_etf, use_etf, rb)
        total_buy += buy_stock * stock_buy_fee(rb) + buy_etf * ETF_FEE
        total_sell += sell_stock * stock_sell_fee(rb) + sell_etf * ETF_FEE
        if not use_etf:
            op0 = open_df.reindex(columns=w.index).loc[t0].replace(0.0, np.nan)
            a = (w.reindex(w.index) / op0).fillna(0.0)
            n_suspend += int((a <= 0).sum())
            op_m = open_df.reindex(columns=w.index).reindex(hold).ffill().bfill()
            cl_m = close_df.reindex(columns=w.index).reindex(hold).ffill().bfill()
            V_open = op_m.multiply(a, axis=1).fillna(0.0).sum(axis=1)
            V_close = cl_m.multiply(a, axis=1).fillna(0.0).sum(axis=1)
        month_switch = 0.0
        for j, t in enumerate(hold):
            if use_ma20 and rs12_on:
                w_t = tier_w(env.idx_close_1.get(t), env.ma20_1.get(t), tier)
            else:
                w_t = 1.0
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
            if use_etf:
                if b_ovn is not None:
                    ovn_t, intra_t = b_ovn.get(t, 0.0), b_intra.get(t, 0.0)
                else:
                    ovn_t, intra_t = e_ovn_s[t], e_intra_s[t]
                if j == 0:
                    r = w_t * intra_t
                else:
                    r = (1.0 + w_prev * ovn_t) * (1.0 + w_t * intra_t) - 1.0
            else:
                if j == 0:
                    r = w_t * (V_close.iloc[0] / V_open.iloc[0] - 1.0)
                else:
                    ovn_t = V_open.iloc[j] / V_close.iloc[j - 1] - 1.0
                    intra_t = V_close.iloc[j] / V_open.iloc[j] - 1.0
                    r = (1.0 + w_prev * ovn_t) * (1.0 + w_t * intra_t) - 1.0
            if cost_on and j == 0:
                nav *= (1.0 - cost_total)
                total_turn += cost_total
            elif cost_on and j > 0 and w_t != w_prev:
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
        prev_w = w
    s = pd.Series(navs).sort_index()
    stats = dict(switch=total_switch, turn=total_turn, cost_buy=total_buy,
                 cost_sell=total_sell, n_stop=n_stop, n_floor=n_floor,
                 n_recov=n_recov, n_buy_block=n_buy_block, n_st_block=n_st_block,
                 n_suspend=n_suspend, n_missing=n_missing, n_trad_removed=n_trad_removed,
                 n_selected=n_selected)
    return s, stats


def daily_stats(s):
    """日频指标: 年化/Sharpe/日频MaxDD/卡玛"""
    n = len(s)
    if n < 2:
        return dict(cagr=0.0, shp=0.0, dd=0.0, k=0.0)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
    ret = s.pct_change().dropna()
    shp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(242.0)
    dd = ((s.cummax() - s) / s.cummax()).max()
    return dict(cagr=cagr, shp=shp, dd=dd, k=cagr / dd)
