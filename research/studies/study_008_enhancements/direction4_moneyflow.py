# -*- coding: utf-8 -*-
"""方向A: 主力资金流因子 (smart money) 加入选股
- 因子: mf20 = 过去20日超大单净流入累计 / 过去20日(超大单+大单)成交额累计; 截至调仓日 rb (T-1 无前视)
- 数据: D:/iquant_data/data_v2/moneyflow1 (日频全市场, 2020-2026 全覆盖)
- 对比: BASE+VAL vs BASE+VAL+MF (有无 MA20), 同框架 20bps / RS12 / 月度调仓
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.factor_dic import run_validation as rv
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements import common as C

MF_DIR = "D:/iquant_data/data_v2/moneyflow1"
MF_WINDOW = 20
MA20_DEEP = 0.98
_mf_cache = {}


def read_mf(d):
    """读单日资金流 -> DataFrame(ts_code 索引, elg_net, elg_turn)"""
    if d in _mf_cache:
        return _mf_cache[d]
    fp = os.path.join(MF_DIR, f"{d}.parquet")
    if not os.path.exists(fp):
        _mf_cache[d] = None
        return None
    df = pd.read_parquet(fp)
    if df is None or df.empty or "ts_code" not in df.columns:
        _mf_cache[d] = None
        return None
    out = pd.DataFrame({
        "elg_net": (df["buy_elg_amount"].fillna(0.0) - df["sell_elg_amount"].fillna(0.0)).values,
        "elg_turn": (df["buy_elg_amount"].fillna(0.0) + df["sell_elg_amount"].fillna(0.0)
                     + df["buy_lg_amount"].fillna(0.0) + df["sell_lg_amount"].fillna(0.0)).values,
    }, index=df["ts_code"].astype(str))
    _mf_cache[d] = out
    return out


def mf_factor_series(env, rb):
    """调仓日 rb 的 mf20 因子 (截至 rb 含, 过去20交易日)"""
    hi = env.trade_dates.index(rb)
    win = env.trade_dates[max(0, hi - MF_WINDOW + 1):hi + 1]
    acc_net, acc_turn = None, None
    for d in win:
        df = read_mf(d)
        if df is None:
            continue
        acc_net = df["elg_net"] if acc_net is None else acc_net.add(df["elg_net"], fill_value=0.0)
        acc_turn = df["elg_turn"] if acc_turn is None else acc_turn.add(df["elg_turn"], fill_value=0.0)
    if acc_net is None or acc_turn is None:
        return None
    mf = acc_net / (acc_turn + 1e-9)
    mf = mf.replace([np.inf, -np.inf], np.nan)
    return mf


def build_picks(env, mf_map, use_mf):
    """复用 Env._build_picks 逻辑; use_mf=True 时加入资金流因子"""
    picks_map = {}
    for rb in env.rebal:
        members = rv.load_index_weight(rb)
        if members is None:
            continue
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
            if use_mf:
                mfs = mf_map.get(rb)
                if mfs is not None and code in mfs.index:
                    v = mfs.loc[code]
                    if np.isfinite(v):
                        row["mf"] = v
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < C.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(sf.winsorize_series).apply(
            lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        cols = sf.BASE_COLS + ["VAL"] + (["mf"] if use_mf else [])
        cols = [c for c in cols if c in zdf.columns]
        has = zdf[cols].dropna()
        if len(has) < C.TOP_N:
            continue
        picks_map[rb] = has.mean(axis=1).nlargest(C.TOP_N).index.tolist()
    return picks_map


def backtest(env, picks_map, use_ma20=True):
    """与 common/方向2 相同回测框架, 返回月度 nav"""
    navs = {}
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        nav = navs.get(rb, 1.0)
        if picks_map.get(rb) is None:
            navs[rb_next] = nav
            continue
        picks = picks_map[rb]
        comb = env.pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
        w = pd.Series(1.0 / len(picks), index=picks)
        cr = (comb * w.reindex(comb.columns)).sum(axis=1, min_count=1)
        for t in hold:
            r_t = e_ret.loc[t]
            if rs12_on:
                ww = 1.0
                if use_ma20:
                    c = env.idx_close_1.get(t, np.nan)
                    m = env.ma20_1.get(t, np.nan)
                    if np.isfinite(c) and np.isfinite(m):
                        ww = 1.0 if c >= m else (0.5 if c >= MA20_DEEP * m else 0.0)
                r_t = ww * cr.loc[t]
            nav *= (1.0 + r_t)
        navs[rb_next] = nav * (1.0 - C.COST)
    return pd.Series(navs).sort_index()


def main():
    env = C.Env()
    print("构建资金流因子面板...")
    mf_map = {}
    for i, rb in enumerate(env.rebal):
        mf_map[rb] = mf_factor_series(env, rb)
    # 覆盖率报告
    ok = [rb for rb, s in mf_map.items() if s is not None and len(s)]
    print(f"资金流因子可用调仓月: {len(ok)}/{len(env.rebal)}")

    print("重新选股 (BASE+VAL vs +MF)...")
    picks_base = build_picks(env, mf_map, use_mf=False)
    picks_mf = build_picks(env, mf_map, use_mf=True)

    # 池重叠度
    overlap = []
    for rb in picks_base:
        if rb in picks_mf:
            a, b = set(picks_base[rb]), set(picks_mf[rb])
            overlap.append(len(a & b) / C.TOP_N)
    if overlap:
        print(f"两个组合月度成分重叠率均值: {np.mean(overlap):.1%}")

    print("回测...")
    navs = {
        "BASE+VAL": backtest(env, picks_base, use_ma20=False),
        "BASE+VAL+MF": backtest(env, picks_mf, use_ma20=False),
        "BASE+VAL+MA20": backtest(env, picks_base, use_ma20=True),
        "BASE+VAL+MF+MA20": backtest(env, picks_mf, use_ma20=True),
    }
    rows = []
    for lb in ["BASE+VAL", "BASE+VAL+MF", "BASE+VAL+MA20", "BASE+VAL+MF+MA20"]:
        rows.append((lb, navs[lb], {}))
    txt, _ = C.metrics_table(rows)
    report = []
    report.append("=" * 84)
    report.append("方向A: 主力资金流因子 (smart money) 加入选股")
    report.append(f"因子: mf20 = 过去{MF_WINDOW}日超大单净流入累计 / 同期(超大单+大单)成交额累计, 截至调仓日 T-1")
    report.append("对比: BASE+VAL 四因子 vs 加入 mf 后五因子; 同 20bps / RS12 / MA20三档(0.98)")
    report.append("")
    report.append(txt)
    report.append("=" * 84)
    out = "\n".join(report)
    with open(os.path.join(C.OUT_DIR, "direction4_moneyflow.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    print(out)

    fig, ax = plt.subplots(figsize=(13, 6))
    for lb, nav in navs.items():
        ax.plot(np.arange(len(nav)), nav.values, label=lb, lw=1.6)
    ax.set_title("主力资金流因子增量 (BASE+VAL, 2020-2026)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "direction4_moneyflow.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
