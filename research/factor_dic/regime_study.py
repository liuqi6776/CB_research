# -*- coding: utf-8 -*-
"""
市场状态研究: 基准ETF区间(2020~2026) 因子有效性 vs 小盘/全市场 + 时期归因 + 择时检测

研究问题:
  Q1 哪些有效因子是小盘暴露驱动, 哪些是全市场通用?
  Q2 2020~2026 各时期哪些因子有效/失效, 为什么?
  Q3 用什么信号检测市场状态(小盘占优/趋势/波动), 检测是否有效?
  Q4 用因子组合 + 状态择时, 回测效果如何?

数据: 本地日频面板(中证1000成分) + 指数日线(000852/000300/000905/000016/932000/512100/510300)
因子IC: results/ic_*.csv(21因子, 月度, 与验证框架同口径: 未来20日收益)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic.factor_lib import FACTOR_REGISTRY

OUT_DIR = rv.OUT_DIR
IDX_DIR = rv.IDX_DIR
FACTORS = [e[0] for e in FACTOR_REGISTRY]
SMALL_IDX, BIG_IDX, MICRO_IDX = "000852.SH", "000300.SH", "932000.CSI"
COST = rv.COST_BPS / 10000.0


def fmt_tbl(df, pct=None, num=None):
    """表格格式化: pct列显示百分数, num列显示3位小数, 其余原样"""
    d = df.copy()
    for c in (pct or []):
        d[c] = d[c].map(lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else str(x))
    for c in (num or []):
        d[c] = d[c].map(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else str(x))
    return d


def load_idx(code):
    df = pd.read_parquet(os.path.join(IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def load_ic_panel():
    """21因子月度IC: DataFrame(index=调仓日, columns=因子)"""
    out = {}
    for f in FACTORS:
        fp = os.path.join(OUT_DIR, f"ic_{f}.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        s = pd.Series(df.iloc[:, 1].values, index=df.iloc[:, 0].values.astype(str))
        out[f] = s
    return pd.DataFrame(out).sort_index()


def fwd_idx_ret(idx_daily, rb, n=rv.FORWARD_DAYS):
    """从 rb 次日起 n 个交易日的指数累计收益"""
    dates = sorted(idx_daily.index)
    if rb not in dates:
        return np.nan
    i = dates.index(rb)
    seg = dates[i + 1:i + 1 + n]
    if len(seg) < n * 0.5:
        return np.nan
    r = idx_daily["pct_chg"].reindex(seg).fillna(0.0) / 100.0
    return (1 + r).prod() - 1


def main():
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]

    ic = load_ic_panel()
    print(f"[load] 21因子IC面板 {ic.shape[0]}月 x {ic.shape[1]}因子")

    sml, big, mic = load_idx(SMALL_IDX), load_idx(BIG_IDX), load_idx(MICRO_IDX)
    # 月度/未来20日: 因子IC对应的未来收益窗口 = 调仓日后20日
    prem = pd.Series({rb: fwd_idx_ret(sml, rb) - fwd_idx_ret(big, rb) for rb in rebal})
    prem_micro = pd.Series({rb: fwd_idx_ret(mic, rb) - fwd_idx_ret(big, rb) for rb in rebal})
    sml_ret = pd.Series({rb: fwd_idx_ret(sml, rb) for rb in rebal})
    big_ret = pd.Series({rb: fwd_idx_ret(big, rb) for rb in rebal})

    # ---------- Q1: 因子IC vs 小盘超额 ----------
    print("\n" + "=" * 100)
    print("Q1 因子 vs 小盘超额(000852-000300, 未来20日): IC时序相关 + 方向")
    print("=" * 100)
    rows = []
    for f in ic.columns:
        s = ic[f].dropna()
        rho = s.corr(prem.reindex(s.index)) if len(s) > 10 else np.nan
        rho_mic = s.corr(prem_micro.reindex(s.index)) if len(s) > 10 else np.nan
        ics = s.mean()
        rows.append((f, ics, rho, rho_mic))
    q1 = pd.DataFrame(rows, columns=["因子", "IC均值", "rho_小盘超额", "rho_微盘超额"]).sort_values("IC均值", ascending=False)
    print(fmt_tbl(q1, num=["IC均值", "rho_小盘超额", "rho_微盘超额"]).to_string(index=False))
    print("  判定: |rho|>=0.4 小盘驱动; |rho|<0.2 全市场通用; 0.2~0.4 部分小盘")

    # ---------- Q2: 时期归因(按半年段) ----------
    print("\n" + "=" * 100)
    print("Q2 时期归因: 每半年段的市场收益/波动 + 各因子IC")
    print("=" * 100)
    # 按半年段(6/12月边界)划分, 无重复计数: 段1=[0,b0]含06边界, 段2=[b0+1,b1]...
    bpos = [i for i, rb in enumerate(rebal) if rb[4:6] in ("06", "12")]
    starts = [0] + [b + 1 for b in bpos]
    ends = bpos + [len(rebal) - 1]
    segs = [(rebal[starts[k]], rebal[ends[k]]) for k in range(len(starts)) if starts[k] <= ends[k]]
    # 波动率: 000852 60日滚动, 年化(sqrt252)
    sml_vol = sml["pct_chg"].rolling(60).std() / 100.0 * np.sqrt(252)
    period_rows = []
    for k, (s0, s1) in enumerate(segs):
        rb_in = rebal[starts[k]:ends[k] + 1]
        if len(rb_in) < 3:
            continue
        sr = sml_ret.reindex(rb_in).add(1).prod() - 1
        br = big_ret.reindex(rb_in).add(1).prod() - 1
        p = (1 + sml_ret.reindex(rb_in)).prod() / (1 + big_ret.reindex(rb_in)).prod() - 1
        vol = sml_vol.reindex(rb_in).mean()
        icm = ic.reindex(rb_in).mean().sort_values(ascending=False)
        best = ",".join(f"{k}({v:.2f})" for k, v in icm.head(3).items())
        worst = ",".join(f"{k}({v:.2f})" for k, v in icm.tail(3).items())
        period_rows.append((f"{s0[:6]}-{s1[:6]}", len(rb_in), sr, br, p, vol, best, worst))
    q2 = pd.DataFrame(period_rows, columns=["区间", "月数", "中证1000", "沪深300", "小盘超额", "年化波动", "最强3因子(IC)", "最弱3因子(IC)"])
    print(fmt_tbl(q2, pct=["中证1000", "沪深300", "小盘超额", "年化波动"]).to_string(index=False))

    # ---------- Q3: 状态检测信号评估 ----------
    print("\n" + "=" * 100)
    print("Q3 状态检测信号: 在调仓日(T)取值, 预测未来20日 000852 收益 / 小盘超额")
    print("=" * 100)
    # 信号构造(T日可得, 纯历史)
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig = pd.DataFrame(index=rebal)
    sig["RS12"] = (ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0          # 过去12月小盘相对强度>0
    sig["RS3"] = (ratio / ratio.shift(60)).rolling(5).mean() - 1.0 > 0            # 过去3月小盘相对强度>0
    sig["TREND"] = sml["close"] > sml["close"].rolling(120).mean()                 # 000852 在MA120上方
    sig["LOWVOL"] = sml_vol < sml_vol.rolling(500, min_periods=120).median()       # 60日波动低于历史中位
    sig["RS12_TREND"] = sig["RS12"] & sig["TREND"]
    sig["RS3_LOWVOL"] = sig["RS3"] & sig["LOWVOL"]
    # 目标
    tgt = pd.DataFrame({f"000852_20d": sml_ret, "prem_20d": prem}, index=rebal)
    ev = []
    for c in sig.columns:
        on, off = tgt[sig[c] == True].dropna(), tgt[sig[c] == False].dropna()
        if len(on) < 5 or len(off) < 5:
            continue
        hit = (on["000852_20d"] > 0).mean()
        prem_on = on["prem_20d"].mean()
        prem_off = off["prem_20d"].mean()
        ret_on, ret_off = on["000852_20d"].mean(), off["000852_20d"].mean()
        med_on, med_off = on["000852_20d"].median(), off["000852_20d"].median()
        ev.append((c, len(on), hit, ret_on, ret_off, med_on, med_off, prem_on, prem_off))
    q3 = pd.DataFrame(ev, columns=["信号", "ON月数", "ON胜率", "ON均收", "OFF均收", "ON中收", "OFF中收", "ON小盘超额", "OFF小盘超额"])
    print(fmt_tbl(q3, pct=["ON胜率", "ON均收", "OFF均收", "ON中收", "OFF中收", "ON小盘超额", "OFF小盘超额"]).to_string(index=False))

    # 状态时间线(按信号 RS12&TREND)
    tl = pd.DataFrame({"rb": rebal}, index=rebal)
    tl["小盘占优"] = ((ratio / ratio.shift(60) - 1) > 0).reindex(rebal)
    tl["趋势"] = sig["TREND"].values
    print("\n[状态时间线] (小盘占优=近3月 000852/000300 走强; 趋势=000852>MA120)")
    print(tl.to_string(index=False))

    # ---------- Q4: 因子组合 + 状态择时回测 ----------
    print("\n" + "=" * 100)
    print("Q4 因子组合(ENH 全期77月 / ENH_F_NI 38月) + 状态择时回测, 基准=000852指数 & 512100ETF")
    print("=" * 100)
    # 重建因子组合月度收益
    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    roe_pit = cb.build_roe_pit(rebal)
    ind_map = cb.load_industry_map()

    port = {k: {} for k in ["ENH", "ENH_F_NI"]}
    bench_m = {}
    bench_etf = {}
    etf_daily = load_idx("512100.SH")
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
            if len(row) >= 3:
                fvals[code] = row
        if len(fvals) < rv.TOP_N:
            continue
        fdf = pd.DataFrame(fvals).T
        zdf = fdf.apply(rv.winsorize).apply(lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
        # ENH: 3因子(ret_1m+ivol+turn) 无roe, 覆盖全期
        cols3 = ["ret_1m", "ivol", "turn"]
        if all(c in zdf.columns for c in cols3):
            has3 = zdf[cols3].dropna()
            if len(has3) >= rv.TOP_N:
                sc3 = has3.mean(axis=1)
                picks3 = sc3.nlargest(rv.TOP_N).index
                sub3 = pct_df.reindex(columns=picks3).reindex(hold).fillna(0.0) / 100.0
                port["ENH"][rb] = (1 + sub3.mean(axis=1)).prod() - 1 - COST
        # ENH_F_NI: 4因子 + 行业中性
        cols4 = ["ret_1m", "ivol", "turn", "roe"]
        if all(c in zdf.columns for c in cols4):
            has4 = zdf[cols4].dropna()
            if len(has4) >= rv.TOP_N:
                ind = pd.Series({c: ind_map.get(c, "NA") for c in has4.index}, index=has4.index)
                sc4 = has4.mean(axis=1).groupby(ind).transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
                picks4 = sc4.nlargest(rv.TOP_N).index
                sub4 = pct_df.reindex(columns=picks4).reindex(hold).fillna(0.0) / 100.0
                port["ENH_F_NI"][rb] = (1 + sub4.mean(axis=1)).prod() - 1 - COST
        b = sml["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_m[rb] = (1 + b).prod() - 1
        be = etf_daily["pct_chg"].reindex(hold).fillna(0.0) / 100.0
        bench_etf[rb] = (1 + be).prod() - 1
    print(f"[回测] 组合月数: ENH {len(port['ENH'])}, ENH_F_NI {len(port['ENH_F_NI'])}, 基准 {len(bench_m)}")

    def stats(pr, bm):
        pr, bm = pd.Series(pr), pd.Series(bm)
        nav = (1 + pr).prod()
        navs = (1 + pr).cumprod()
        years = len(pr) / 12.0
        return dict(cagr=nav ** (1 / years) - 1,
                    sharpe=pr.mean() / pr.std(ddof=1) * np.sqrt(12),
                    mdd=(navs.cummax() - navs).max(),
                    win=(pr > 0).mean(),
                    excess=(1 + pr).prod() / (1 + bm).prod() - 1)

    bm_idx = pd.Series(bench_m)
    bm_etf = pd.Series(bench_etf)
    sigq = sig  # 择时信号(调仓日取值)
    rows4 = []
    for pf_name, base in [("ENH", pd.Series(port["ENH"])), ("ENH_F_NI", pd.Series(port["ENH_F_NI"]))]:
        for bn_name, bm in [("000852", bm_idx), ("ETF512100", bm_etf)]:
            bm_s = bm.reindex(base.index)
            st0 = stats(base, bm_s)
            rows4.append((f"{pf_name} 无择时[{bn_name}]", st0["cagr"], st0["sharpe"], st0["mdd"], st0["win"], st0["excess"]))
            # 规则: 小盘强(RS12>0)->持因子组合; 小盘弱(RS12<=0)->持基准(吃反弹)
            pr_r = base.where(sigq["RS12"].reindex(base.index), bm_s)
            st = stats(pr_r, bm_s)
            rows4.append((f"{pf_name} RS12择时(弱->基准)[{bn_name}]", st["cagr"], st["sharpe"], st["mdd"], st["win"], st["excess"]))
            # 规则: 小盘强且趋势上->因子; 否则基准
            pr_r2 = base.where(sigq["RS12_TREND"].reindex(base.index), bm_s)
            st = stats(pr_r2, bm_s)
            rows4.append((f"{pf_name} RS12&TREND(弱->基准)[{bn_name}]", st["cagr"], st["sharpe"], st["mdd"], st["win"], st["excess"]))
    q4 = pd.DataFrame(rows4, columns=["策略", "年化", "Sharpe", "MaxDD", "月胜率", "超额vs基准"])
    for bn_name, bm in [("000852", bm_idx), ("ETF512100", bm_etf)]:
        bm_nav = bm.add(1).prod()
        bm_years = len(bm) / 12.0
        q4.loc[len(q4)] = [f"基准 {bn_name}", bm_nav ** (1 / bm_years) - 1,
                           bm.mean() / bm.std(ddof=1) * np.sqrt(12),
                           (bm.add(1).cumprod().cummax() - bm.add(1).cumprod()).max(), (bm > 0).mean(), 0.0]
    print(fmt_tbl(q4, pct=["年化", "MaxDD", "月胜率", "超额vs基准"], num=["Sharpe"]).to_string(index=False))

    # 因子组合月度收益 vs 小盘超额相关
    for pf_name, base in [("ENH", pd.Series(port["ENH"])), ("ENH_F_NI", pd.Series(port["ENH_F_NI"]))]:
        print(f"\n[Q4补充] {pf_name} 月收益 vs 小盘超额 rho = {base.corr(prem.reindex(base.index)):.3f}, "
              f"vs 000852 rho = {base.corr(sml_ret.reindex(base.index)):.3f}")

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    fp = os.path.join(OUT_DIR, "regime_study.txt")
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("Q1 因子 vs 小盘超额\n" + q1.to_string(index=False) + "\n\n")
        fh.write("Q2 时期归因\n" + q2.to_string(index=False) + "\n\n")
        fh.write("Q3 状态检测信号\n" + q3.to_string(index=False) + "\n\n")
        fh.write("Q4 择时回测\n" + q4.to_string(index=False) + "\n")
    print(f"\n[保存] {fp}")


if __name__ == "__main__":
    main()
