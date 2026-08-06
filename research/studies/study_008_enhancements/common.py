# -*- coding: utf-8 -*-
"""study_008_enhancements 共享数据模块
复用 risk_control_bt.py 的数据加载与月度选股逻辑 (BASE+VAL Top50, 2020-2026):
  - trade_dates / rebal / hold 分段
  - picks_map: 每月 Top50
  - 月度 comb (个股日收益面板) / e_ret (512100) / 风控信号序列
模块级一次性加载并缓存, 三个方向脚本直接 import。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic import style_factors as sf

TOP_N = rv.TOP_N
COST = rv.COST_BPS / 10000.0
SQRT_242 = np.sqrt(242.0)
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STUDY_DIR, "data")
OUT_DIR = os.path.join(STUDY_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


class Env:
    """一次性加载全部基础数据"""

    def __init__(self):
        self.trade_dates = rv.load_trade_dates()
        months = {d[:6]: d for d in self.trade_dates if d[:4] >= str(rv.START_YEAR)}
        # 保留最后一个月 (数据可到 2026-07-31, 末尾月作为最新 OOS 样本纳入回测)
        self.rebal = sorted(months.values())
        all_codes = set()
        for rb in self.rebal:
            m = rv.load_index_weight(rb)
            if m:
                all_codes |= m
        self.all_codes = sorted(all_codes)
        stocks, pct_df, _, _, _ = rv.load_panels(self.trade_dates, self.all_codes, None)
        self.pct_df = pct_df
        ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, self.all_codes)
        self.ret_1m, self.ivol, self.turn, self.fwd = ret_1m, ivol, turn, fwd
        val_map = sf.load_valuation(self.rebal, self.all_codes)
        funda_map = sf.build_funda_pit(self.rebal, self.all_codes)
        self.panels = sf.build_factors(val_map, funda_map, self.rebal)
        self._load_market()
        self.picks_map = self._build_picks()

    def _load_market(self):
        sml = load_idx("000852.SH")
        big = load_idx("000300.SH")
        etf = load_idx("512100.SH")
        ratio = sml["close"] / big["close"].reindex(sml.index)
        self.sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(self.rebal)
        idx_ret = sml["pct_chg"] / 100.0
        idx_close = sml["close"]
        ma20 = idx_close.rolling(20).mean()
        idx_vol20 = idx_ret.rolling(20).std() * SQRT_242
        self.idx_close_1 = idx_close.shift(1)
        self.ma20_1 = ma20.shift(1)
        self.idx_vol20_1 = idx_vol20.shift(1)
        self.etf_ret = etf["pct_chg"] / 100.0
        self.idx_ret = idx_ret
        # ETF 腿专属风控信号 (512100, T-1 已知 T 日生效; 与股票腿 MA20 解耦)
        etf_close = etf["close"]
        self.etf_close_1 = etf_close.shift(1)
        self.ma5_1 = etf_close.rolling(5).mean().shift(1)
        self.ma10_1 = etf_close.rolling(10).mean().shift(1)
        self.ma120_1 = etf_close.rolling(120).mean().shift(1)

    def _build_picks(self):
        picks_map = {}
        for rb in self.rebal:
            members = rv.load_index_weight(rb)
            if members is None:
                continue
            fvals = {}
            for code in members:
                f1, f2, ft = self.ret_1m.get(code), self.ivol.get(code), self.turn.get(code)
                fr = self.fwd.get(code)
                if fr is None or rb not in fr.index:
                    continue
                row = {}
                if f1 is not None and rb in f1.index:
                    row["ret_1m"] = f1.loc[rb]
                if f2 is not None and rb in f2.index:
                    row["ivol"] = f2.loc[rb]
                if ft is not None and rb in ft.index:
                    row["turn"] = ft.loc[rb]
                for name in self.panels:
                    p = self.panels[name].get(rb)
                    if p is not None and code in p.index:
                        v = p.loc[code]
                        if np.isfinite(v):
                            row[name] = v
                if len(row) >= 3:
                    fvals[code] = row
            if len(fvals) < TOP_N:
                continue
            fdf = pd.DataFrame(fvals).T
            zdf = fdf.apply(sf.winsorize_series).apply(
                lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
            # 可用因子列 (缺 VAL 快照的月份降级为 BASE 3 因子, 保证最新月仍纳入回测)
            cols = [c for c in sf.BASE_COLS + ["VAL"] if c in zdf.columns]
            has = zdf[cols].dropna()
            if len(has) < TOP_N:
                continue
            picks_map[rb] = has.mean(axis=1).nlargest(TOP_N).index.tolist()
        return picks_map

    def month_segments(self):
        """逐月 yield (rb, rb_next, hold, picks, comb, e_ret, rs12_on)"""
        for i, rb in enumerate(self.rebal):
            if i + 1 >= len(self.rebal):
                rb_next = self.trade_dates[-1]  # 最后一个调仓月: hold 延伸到数据末尾
            else:
                rb_next = self.rebal[i + 1]
            hi, hn = self.trade_dates.index(rb), self.trade_dates.index(rb_next)
            hold = self.trade_dates[hi + 1:hn + 1]
            if len(hold) == 0:
                continue
            if rb not in self.picks_map:
                yield (rb, rb_next, hold, None, None, None, None)
                continue
            picks = self.picks_map[rb]
            comb = self.pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
            e_ret = self.etf_ret.reindex(hold).fillna(0.0)
            rs12_on = bool(self.sig_rs12.loc[rb]) if rb in self.sig_rs12.index else True
            yield (rb, rb_next, hold, picks, comb, e_ret, rs12_on)


# 两融全市场融资余额序列 (SSE+SZSE+BSE 合并, 单位: 元)
def load_margin_series(env):
    if getattr(env, "_margin", None) is not None:
        return env._margin
    frames = []
    mdir = "D:/iquant_data/data_v2/margin1"
    for d in env.trade_dates:
        fp = os.path.join(mdir, f"{d}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        if "rzye" not in df.columns or df.empty:
            continue
        frames.append((d, float(df["rzye"].sum())))
    s = pd.Series(dict(frames)).sort_index()
    env._margin = s
    return s


# 北向净流入序列 (2020-01~2024-08)
def load_north_series():
    df = pd.read_parquet(os.path.join(DATA_DIR, "north_flow.parquet"))
    s = pd.Series(df["north_net"].values, index=df["trade_date"].astype(str)).sort_index()
    return s.astype(float)


# 东财行业映射: ts_code -> industry
def load_industry_map():
    df = pd.read_parquet(os.path.join(DATA_DIR, "industry_map.parquet"))
    return dict(zip(df["ts_code"].astype(str), df["industry"].astype(str)))


# 通用月度指标
def monthly_metrics(nav, ann=242.0):
    """nav: pd.Series(index=月序). 年化/Sharpe/MaxDD/卡玛/月胜率"""
    nav = nav.dropna()
    rets = nav / nav.shift(1) - 1
    rets = rets.dropna()
    years = len(rets) / 12.0
    total = nav.iloc[-1] / nav.iloc[0] - 1 if len(nav) else 0.0
    ann_ret = (1 + total) ** (1 / years) - 1 if years > 0 else 0.0
    sharpe = rets.mean() / (rets.std() + 1e-9) * np.sqrt(12.0)
    hwm = nav.cummax()
    dd = nav / hwm - 1.0
    mdd = dd.min() if len(dd) else 0.0
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan
    win = (rets > 0).mean()
    return dict(ann=ann_ret, sharpe=sharpe, mdd=mdd, calmar=calmar, win=win, final=nav.iloc[-1] if len(nav) else 1.0)


def metrics_table(rows):
    """rows: list of (label, nav_series, extra dict)"""
    lines = []
    lines.append(f"{'策略':<28}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'卡玛':>7}{'月胜率':>8}{'终值':>8}")
    out = []
    for label, nav, extra in rows:
        m = monthly_metrics(nav)
        extra_s = ""
        if extra:
            extra_s = "  " + " ".join(f"{k}={v}" for k, v in extra.items())
        lines.append(f"{label:<28}{m['ann']*100:>7.2f}%{m['sharpe']:>8.2f}{m['mdd']*100:>8.2f}%{m['calmar']:>7.2f}{m['win']*100:>7.1f}%{m['final']:>8.3f}{extra_s}")
        out.append(dict(label=label, **m, **extra))
    return "\n".join(lines), out
