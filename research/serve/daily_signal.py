# -*- coding: utf-8 -*-
"""
每日信号生成：BASE+VAL+RS12+MA20三档(0.98) 最优策略 (research/serve 部署版)

复刻 risk_control_bt.py 的选股/择时/风控逻辑, 但只输出【最新持有期】的今日操作建议:
  - 当前调仓日 rb = 最近一个月的最后交易日 (含最新, 不去尾)
  - Top50 选股 (ret_1m + ivol + turnover_vol_20 + VAL, 截面 zscore 均值取 Top50)
  - RS12 择时 (000852/000300 过去240日相对强度, 弱时持 512100 ETF)
  - MA20 三档日频仓位 (close>=MA20 -> 1.0; MA20*0.98<=close<MA20 -> 0.5; 否则 0.0)
  - 估值数据缺失时 PIT fallback 到 <=调仓日 的最新估值快照 (与回测同口径)

落盘: research/serve/data/daily/YYYY-MM-DD.json (dashboard 历史记录)

用法:
    python research/serve/daily_signal.py              # 今日信号(落盘 + 打印)
    python research/serve/daily_signal.py --email      # 生成后发邮件
    python research/serve/daily_signal.py --rb 20260422  # 指定调仓日(调试/补历史)
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from research.factor_dic import run_validation as rv
from research.factor_dic import combo_backtest as cb
from research.factor_dic import style_factors as sf

SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SERVE_DIR, "data", "daily")
TOP_N = rv.TOP_N
DEEP = 0.98
NAME_MAP_PATH = os.path.join(ROOT, "stock_name_map.parquet")


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def load_valuation_pit(rebal_dates, all_codes):
    """估值面板(PIT fallback): {rb: (data_date, DataFrame)} —— 对每个调仓日取 <= rb 的最新估值文件
    回测期 (有精确文件) 行为与 sf.load_valuation 一致; 数据滞后时用旧快照并记录实际日期。
    """
    avail = sorted(f[:8] for f in os.listdir(sf.lf.PE_DIR) if f.endswith(".parquet"))
    out = {}
    for rb in rebal_dates:
        ok = [d for d in avail if d <= rb]
        if not ok:
            continue
        d = ok[-1]
        try:
            df = pd.read_parquet(os.path.join(sf.lf.PE_DIR, f"{d}.parquet"))
        except Exception:
            continue
        df = df.dropna(subset=["pe_ttm", "pb", "ps_ttm", "dv_ttm"], how="all")
        df = df[df["ts_code"].astype(str).isin(all_codes)]
        if not df.empty:
            out[rb] = (d, df.set_index("ts_code"))
    return out


def build_signal(rb, picks, sig_rs12, idx_close, ma20, name_map):
    """生成单期信号 dict"""
    rs12_on = bool(sig_rs12.loc[rb]) if rb in sig_rs12.index else True
    rs12_val = float(sig_rs12.loc[rb]) if rb in sig_rs12.index else np.nan

    # 最新交易日 = 调仓日当天 (无未来数据, PIT 干净)
    as_of = rb
    c = float(idx_close.loc[as_of]) if as_of in idx_close.index else np.nan
    m = float(ma20.loc[as_of]) if as_of in ma20.index else np.nan
    w = 1.0
    if np.isfinite(c) and np.isfinite(m):
        w = 1.0 if c >= m else (0.5 if c >= DEEP * m else 0.0)

    if not rs12_on:
        action = "持有 512100 ETF (全额, RS12 弱)"
        position = "512100 ETF"
    elif w >= 1.0:
        action = "满仓持有组合 (Top50 等权)"
        position = "股票组合"
    elif w >= 0.5:
        action = "半仓持有 (000852 跌破 MA20, 50% 组合 + 50% 现金)"
        position = "股票组合(半仓)"
    else:
        action = "空仓观望 (000852 跌破 MA20×0.98, 全部现金)"
        position = "现金"

    picks_out = []
    for code, score in picks:
        picks_out.append({
            "code": code,
            "name": name_map.get(code, ""),
            "score": round(float(score), 3),
        })

    return {
        "as_of_date": as_of,
        "rebalance_date": rb,
        "rs12_on": rs12_on,
        "rs12_value": round(rs12_val, 4) if rs12_val == rs12_val else None,
        "ma20": {"close": None if not np.isfinite(c) else round(c, 2),
                 "ma20": None if not np.isfinite(m) else round(m, 2),
                 "deep": DEEP, "weight": w},
        "action": action,
        "position": position,
        "picks": picks_out,
        "picks_count": len(picks_out),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", action="store_true", help="生成后发送邮件")
    ap.add_argument("--rb", default=None, help="指定调仓日 (YYYYMMDD), 默认最新")
    args = ap.parse_args()

    t0 = time.time()
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())  # 含最新一个月 (不回测去尾)

    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"[load] 调仓日 {len(rebal)} 个 ({rebal[0]}~{rebal[-1]}), 成分股 {len(all_codes)}", flush=True)

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, all_codes)
    val_map = load_valuation_pit(rebal, all_codes)   # {rb: (data_date, df)}
    funda_map = sf.build_funda_pit(rebal, all_codes)
    val_map_sf = {rb: df for rb, (_, df) in val_map.items()}
    panels = sf.build_factors(val_map_sf, funda_map, rebal)
    print(f"[load] 因子面板完成 ({time.time()-t0:.0f}s)", flush=True)

    sml = load_idx("000852.SH")
    big = load_idx("000300.SH")
    ratio = sml["close"] / big["close"].reindex(sml.index)
    sig_rs12 = ((ratio / ratio.shift(240)).rolling(5).mean() - 1.0 > 0).reindex(rebal)
    idx_close = sml["close"]
    ma20 = idx_close.rolling(20).mean()

    name_map = {}
    if os.path.exists(NAME_MAP_PATH):
        try:
            nd = pd.read_parquet(NAME_MAP_PATH)
            name_map = dict(zip(nd["ts_code"].astype(str), nd["name"].astype(str)))
        except Exception:
            pass

    # ---------- 最新(或指定)调仓日选股 ----------
    rb = args.rb if args.rb else rebal[-1]
    if rb not in rebal:
        print(f"[err] 调仓日 {rb} 不在列表 {rebal[0]}~{rebal[-1]}")
        sys.exit(1)
    members = rv.load_index_weight(rb)
    fvals = {}
    for code in members or []:
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
        for pname in panels:
            p = panels[pname].get(rb)
            if p is not None and code in p.index:
                v = p.loc[code]
                if np.isfinite(v):
                    row[pname] = v
        if len(row) >= 3:
            fvals[code] = row
    if len(fvals) < TOP_N:
        print(f"[err] 调仓日 {rb} 有效因子股数 {len(fvals)} < {TOP_N}")
        sys.exit(1)

    fdf = pd.DataFrame(fvals).T
    zdf = fdf.apply(sf.winsorize_series).apply(
        lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)
    cols = sf.BASE_COLS + ["VAL"]
    has = zdf[cols].dropna()
    if len(has) < TOP_N:
        print(f"[err] 调仓日 {rb} 完整因子股数 {len(has)} < {TOP_N} (VAL 数据滞后?)")
        sys.exit(1)
    scored = has.mean(axis=1).sort_values(ascending=False)
    picks = list(zip(scored.index.tolist(), scored.values.tolist()))
    top = picks[:TOP_N]

    sig = build_signal(rb, top, sig_rs12, idx_close, ma20, name_map)

    # 数据时效标注
    notes = []
    if rb in val_map:
        val_date = val_map[rb][0]
        if val_date < rb:
            notes.append(f"估值数据截至 {val_date} (VAL 因子使用该旧快照)")
    iw_dates = sorted(f[3:11] for f in os.listdir(rv.IW_DIR) if f.startswith("iw_"))
    iw_avail = [d for d in iw_dates if d <= rb]
    if iw_avail and iw_avail[-1] < rb:
        notes.append(f"成分股清单截至 {iw_avail[-1]} (使用最近一期)")
    sig["data_notes"] = notes
    sig["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---------- 落盘 ----------
    os.makedirs(DATA_DIR, exist_ok=True)
    fp = os.path.join(DATA_DIR, f"{time.strftime('%Y-%m-%d')}.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(sig, fh, ensure_ascii=False, indent=2)
    print(f"[saved] {fp}")

    # ---------- 打印摘要 ----------
    print("\n" + "=" * 78)
    print(f"今日操作建议  数据截至 {sig['as_of_date']}  生成 {sig['generated_at']}")
    print("=" * 78)
    print(f"当前调仓日: {sig['rebalance_date']}")
    print(f"RS12 择时:  {'强(持股)' if sig['rs12_on'] else '弱(持ETF)'}  value={sig['rs12_value']}")
    if sig["rs12_on"]:
        m20 = sig["ma20"]
        print(f"MA20 三档:  close={m20['close']}  MA20={m20['ma20']}  (deep={m20['deep']})  -> 仓位 w={m20['weight']}")
    print(f"操作: {sig['action']}")
    print(f"持仓数: {sig['picks_count']}")
    print("\nTop 10:")
    for i, p in enumerate(sig["picks"][:10], 1):
        nm = p["name"] or "?"
        print(f"  {i:>2}. {p['code']}  {nm:<8}  score={p['score']}")
    if notes:
        print("\n数据时效提示:")
        for n in notes:
            print(f"  - {n}")

    if args.email:
        try:
            from notify import send_email_html
            lines = [f"<h3>今日操作 ({sig['as_of_date']})</h3>",
                     f"<p><b>{sig['action']}</b></p>",
                     f"<p>RS12: {'强' if sig['rs12_on'] else '弱'} | 调仓日 {sig['rebalance_date']} | 持仓 {sig['picks_count']} 只</p>",
                     "<ul>"]
            for p in sig["picks"][:10]:
                lines.append(f"<li>{p['code']} {p['name']} (score {p['score']})</li>")
            lines.append("</ul>")
            for n in notes:
                lines.append(f"<p style='color:#a00'>⚠️ {n}</p>")
            body = "".join(lines)
            send_email_html(f"量化策略今日操作 {sig['as_of_date']}", body)
        except Exception as e:
            print(f"[warn] 邮件发送失败: {e}")


if __name__ == "__main__":
    main()
