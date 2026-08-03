# -*- coding: utf-8 -*-
"""
林奇因子(PEG)验证: 彼得·林奇 GARP 框架
  PEG = PE(TTM) / 净利润同比增速(%)
  逻辑: 低 PEG(成长+估值合理) = 好, 方向 neg(取负)

统一框架(与 21 因子验证/合并回测可比):
  - 样本: 2020.01~2026.06, 中证1000成分股, 月末调仓
  - PIT: pe_ttm 取调仓日当天(other_day1), netprofit_yoy 取 ann_date<=调仓日 的最新公告
  - 未来收益: T+1~T+20 累计, Top50 月度调仓, 20bps 双边成本

输出:
  1. PEG 单独: IC/ICIR/NW t/分组/Top50 回测
  2. 与现有因子 IC 时序相关(正交性) + 与小盘超额相关(Q1 口径)
  3. 合并: POOL_NI = ret_1m+ivol+turn+roe+peg 行业中性, 对比 ENH_F_NI
  4. RS12 择时(小盘弱->持基准)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings
from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb

OUT_DIR = rv.OUT_DIR
MV_DIR = os.path.join(settings.DATA_PATH, "other_day1")
FUNDA_PATH = os.path.join(settings.DATA_PATH, "fundamental1", "fina_indicator_cache.parquet")
PE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pe_ttm")
FINA_YOY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "fina_yoy.parquet")
COST = rv.COST_BPS / 10000.0
MIN_YOY = 5.0       # 净利润同比 >= 5% 才计算 PEG(负增长/微增 PEG 无意义)
MAX_YOY = 300.0     # 增速极端值截断
MAX_PE = 300.0      # PE 截断


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def load_pe_ttm(rebal_dates, all_codes):
    """调仓日 pe_ttm(滚动市盈率, tushare daily_basic): {rb: {code: pe_ttm}}"""
    out = {}
    if not os.path.isdir(PE_DIR):
        return out
    for rb in rebal_dates:
        fp = os.path.join(PE_DIR, f"{rb}.parquet")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_parquet(fp, columns=["ts_code", "pe_ttm"])
        except Exception:
            try:
                df = pd.read_parquet(fp)[["ts_code", "pe_ttm"]]
            except Exception:
                continue
        df = df[df["ts_code"].astype(str).isin(all_codes)].dropna(subset=["pe_ttm"])
        if not df.empty:
            out[rb] = dict(zip(df["ts_code"].astype(str), df["pe_ttm"]))
    return out


def build_yoy_pit(rebal_dates, all_codes):
    """netprofit_yoy PIT 面板: 每月取 ann_date<=调仓日 的最新公告值(无前视)
    数据源: data/fina_yoy.parquet (tushare fina_indicator, 2020-2026 逐股全历史)"""
    if not os.path.exists(FINA_YOY):
        return {}
    funda = pd.read_parquet(FINA_YOY)[["ts_code", "ann_date", "netprofit_yoy"]]
    funda["ann_date"] = funda["ann_date"].astype(str).str[:8]
    funda = funda.dropna(subset=["netprofit_yoy"]).sort_values("ann_date")
    out = {}
    for rb in rebal_dates:
        latest = funda[funda["ann_date"] <= rb].drop_duplicates("ts_code", keep="last")
        out[rb] = latest.set_index("ts_code")["netprofit_yoy"]
    return out


def build_peg(pe_map, yoy_map, rebal_dates, all_codes):
    """PEG 面板: {rb: {code: peg}}, 高值=差(低 PEG 好), 方向在回测时取负"""
    out = {}
    for rb in rebal_dates:
        pe = pe_map.get(rb, {})
        yoy = yoy_map.get(rb, {})
        if not pe or len(yoy) == 0:
            continue
        common = set(pe) & set(yoy.index)
        rows = {}
        for c in common:
            p, g = pe[c], yoy[c]
            if p <= 0 or not MIN_YOY <= g <= MAX_YOY:
                continue
            rows[c] = p / g          # PEG
        if len(rows) >= 50:
            out[rb] = rows
    return out


def main():
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]
    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"[load] 调仓日 {len(rebal)} 个, 成分股 {len(all_codes)} 只")

    pe_map = load_pe_ttm(rebal, all_codes)
    yoy_map = build_yoy_pit(rebal, all_codes)
    peg = build_peg(pe_map, yoy_map, rebal, all_codes)
    print(f"[load] PEG 面板 {len(peg)} 个月有效")

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    roe_pit = cb.build_roe_pit(rebal)
    ind_map = cb.load_industry_map()

    # ---------- 1. PEG 单独验证(月度截面 IC) ----------
    ic_list, group_stats = [], {q: [] for q in range(5)}
    fwd_series = {c: fwd[c] for c in fwd if c in all_codes}
    for rb in rebal:
        pegd = peg.get(rb)
        if not pegd:
            continue
        f = pd.Series(pegd)          # PEG 高=差
        f = -f                        # 取负: 高值=好(低PEG好), 方向统一
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        fr = {}
        for c in f.index:
            if c in members and c in fwd_series and rb in fwd_series[c].index:
                fr[c] = fwd_series[c].loc[rb]
        r = pd.Series(fr)
        f = f.reindex(r.index).dropna()
        r = r.reindex(f.index).dropna()
        if len(f) < 50:
            continue
        fw = rv.winsorize(f)
        ic = fw.rank().corr(r.rank())
        if np.isfinite(ic):
            ic_list.append((rb, ic))
        try:
            q = pd.qcut(fw.rank(method="first"), 5, labels=False)
            for qq in range(5):
                sel = r[q == qq]
                if len(sel):
                    group_stats[qq].append(sel.mean())
        except Exception:
            pass

    ics = pd.Series([x[1] for x in ic_list], index=[x[0] for x in ic_list])
    t_val, _ = rv.newey_west_t(ics.values)
    print("\n" + "=" * 90)
    print("PEG 因子单独验证 (低PEG=好, 方向已取负)")
    print("=" * 90)
    print(f"IC均值={ics.mean():.4f}  ICIR={ics.mean()/ics.std(ddof=1):.4f}  NW t(lag=4)={t_val:.2f}  正IC占比={(ics>0).mean():.1%}  n={len(ics)}")
    print("分组(未来20日均收益):")
    for q in range(5):
        m = np.nanmean(group_stats[q]) if group_stats[q] else np.nan
        print(f"  Q{q+1}(低PEG): {m:.4f}")
    ics.to_csv(os.path.join(OUT_DIR, "ic_peg.csv"))

    # ---------- 2. 正交性 + 小盘超额 ----------
    sml = load_idx("000852.SH")
    big = load_idx("000300.SH")

    def fwd_idx(idx_daily, rb, n=rv.FORWARD_DAYS):
        dates = sorted(idx_daily.index)
        if rb not in dates:
            return np.nan
        i = dates.index(rb)
        seg = dates[i + 1:i + 1 + n]
        if len(seg) < n * 0.5:
            return np.nan
        r = idx_daily["pct_chg"].reindex(seg).fillna(0.0) / 100.0
        return (1 + r).prod() - 1

    prem = pd.Series({rb: fwd_idx(sml, rb) - fwd_idx(big, rb) for rb in rebal})
    print("\n[正交性] PEG 与现有因子 IC 时序相关:")
    # 现有因子 IC 面板(从 results/ic_*.csv)
    def load_ic(f):
        fp = os.path.join(OUT_DIR, f"ic_{f}.csv")
        if not os.path.exists(fp):
            return None
        df = pd.read_csv(fp)
        return pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].values.astype(str))
    for f, fname in [("ret_1m", "ret_1m"), ("ivol", "ivol"), ("turn", "turnover_vol_20")]:
        s = load_ic(fname)
        if s is None:
            continue
        rho = s.reindex(ics.index).corr(ics)
        print(f"  PEG vs {f:<20} rho={rho:.3f}")
    rho_prem = ics.corr(prem.reindex(ics.index))
    print(f"  PEG IC vs 小盘超额(000852-000300) rho={rho_prem:.3f}  (|rho|>=0.4 小盘驱动, <0.2 全市场通用)")

    # ---------- 3. 合并回测: POOL_NI vs ENH_F_NI + PEG 单独 ----------
    print("\n" + "=" * 90)
    print("合并回测 (Top50, 20bps, 基准 000852 / ETF512100)")
    print("=" * 90)
    etf = load_idx("512100.SH")
    bench_m, bench_etf = {}, {}
    port = {"LYNCH": [], "ENH": [], "ENH_F_NI": [], "POOL_NI": []}
    for i, rb in enumerate(rebal):
        if i + 1 >= len(rebal):
            continue
        members = rv.load_index_weight(rb)
        if members is None:
            continue
        rb_next = rebal[i + 1]
        hi, hn = trade_dates.index(rb), trade_dates.index(rb_next)
        hold = trade_dates[hi + 1:hn + 1]
        fvals = {}
        for code in members:
            f1, f2, ft = ret_1m.get(code), ivol.get(code), turn.get(code)
            fr = fwd.get(code)
            if fr is None or rb not in fr.index:
                continue
            row = {}
            if f1 is not None and rb in f1.index:
                row["ret_1m"] = f1.loc[rb]
            if f2 is not None and rb in f2.index:
                row["ivol"] = f2.loc[rb]
            if ft is not None and rb in ft.index:
                row["turn"] = ft.loc[rb]
            if rb in roe_pit and code in roe_pit[rb].index:
                row["roe"] = roe_pit[rb].loc[code]
            if peg.get(rb) and code in peg[rb]:
                pv = -peg[rb][code]           # 低PEG好 -> 高值=好
                if np.isfinite(pv):
                    row["peg"] = pv
            if len(row) >= 1:
                fvals[code] = row
        if len(fvals) < rv.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(rv.winsorize).apply(lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        ind = pd.Series({c: ind_map.get(c, "NA") for c in fdf.index}, index=fdf.index)

        def _run(name, cols, neut):
            if not all(c in zdf.columns for c in cols):
                return
            has = zdf[cols].dropna()
            if len(has) < rv.TOP_N:
                return
            sc = has.mean(axis=1)
            if neut:
                sc = sc.groupby(ind).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
            picks = sc.nlargest(rv.TOP_N).index
            sub = pct_df.reindex(columns=picks).reindex(hold).fillna(0.0) / 100.0
            port[name].append(dict(rb=rb, net=(1 + sub.mean(axis=1)).prod() - 1 - COST))

        _run("LYNCH", ["peg"], False)                       # PEG 单独(全期, 有PEG数据才有效)
        _run("ENH", ["ret_1m", "ivol", "turn"], False)      # 现有 3 因子全期
        _run("ENH_F_NI", ["ret_1m", "ivol", "turn", "roe"], True)
        _run("POOL_NI", ["ret_1m", "ivol", "turn", "roe", "peg"], True)   # 加 PEG 增量
        b = sml["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_m[rb] = (1 + b).prod() - 1
        be = etf["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_etf[rb] = (1 + be).prod() - 1

    def stats(pr, bm):
        pr, bm = pd.Series(pr), pd.Series(bm)
        nav = (1 + pr).prod()
        navs = (1 + pr).cumprod()
        years = len(pr) / 12.0
        return dict(cagr=nav ** (1 / years) - 1, sharpe=pr.mean() / pr.std(ddof=1) * np.sqrt(12),
                    mdd=(navs.cummax() - navs).max(), win=(pr > 0).mean(),
                    excess=(1 + pr).prod() / (1 + bm).prod() - 1, n=len(pr))

    bm_i, bm_e = pd.Series(bench_m), pd.Series(bench_etf)
    # RS12 信号
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)

    print(f"{'策略':<26}{'n':>4}{'年化':>8}{'Sharpe':>8}{'MaxDD':>9}{'月胜率':>8}{'超额v000852':>12}{'超额vETF':>10}")
    for name, neut_lbl in [("LYNCH", "PEG单独(全期)"), ("ENH", "3因子(现有,全期)"),
                           ("ENH_F_NI", "4因子+行业中性"), ("POOL_NI", "5因子(含PEG)+行业中性")]:
        pr = pd.Series({r["rb"]: r["net"] for r in port[name]})
        if pr.empty:
            continue
        s0 = stats(pr, bm_i.reindex(pr.index))
        s0e = stats(pr, bm_e.reindex(pr.index))
        print(f"{name+' '+neut_lbl:<24}{s0['n']:>4}{s0['cagr']:>8.2%}{s0['sharpe']:>8.2f}{s0['mdd']:>9.2%}{s0['win']:>8.1%}{s0['excess']:>12.2%}{s0e['excess']:>10.2%}")
        # RS12 择时
        pr_t = pr.where(sig_rs12.reindex(pr.index), bm_e.reindex(pr.index))
        st = stats(pr_t, bm_e.reindex(pr.index))
        print(f"{'  +RS12择时(vs ETF)':<24}{st['n']:>4}{st['cagr']:>8.2%}{st['sharpe']:>8.2f}{st['mdd']:>9.2%}{st['win']:>8.1%}{'-':>12}{st['excess']:>10.2%}")

    # PEG 覆盖率/描述统计
    cover = []
    for rb in rebal:
        pegd = peg.get(rb)
        if pegd:
            cover.append(len(pegd))
    if cover:
        peg_means = [np.mean(list(v.values())) for v in peg.values() if v]
        print(f"\n[PEG 数据] 月均覆盖 {np.mean(cover):.0f} 只/中证1000; PEG 均值(截面月均) "
              f"{np.nanmean(peg_means):.1f}")

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "lynch_factor.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"PEG: IC={ics.mean():.4f} ICIR={ics.mean()/ics.std(ddof=1):.4f} NWt={t_val:.2f} pos={ (ics>0).mean():.3f}\n")
        fh.write(f"PEG vs 小盘超额 rho={rho_prem:.3f}\n")
    print(f"[保存] {os.path.join(OUT_DIR, 'lynch_factor.txt')}")


if __name__ == "__main__":
    main()
