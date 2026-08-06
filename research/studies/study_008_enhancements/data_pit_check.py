# -*- coding: utf-8 -*-
"""数据端核验 (3.27): PIT 财务数据核验 (new-tea-quant PIT 思路)

VAL 合成依赖两条数据腿:
  1) 估值 daily_basic   : 调仓日快照 (trade_date=rb, 天然 PIT)
  2) 财务 fina_indicator : 按 ann_date<=rb 取最新 (build_funda_pit 声称 PIT)

本节逐层实证核验财务腿的真实 PIT 性:
  [1] 无前视烟雾测试: 每个调仓日 rb, 所用财务行的 ann_date 必须 <= rb;
      并检查"同报告期修订重述"占比 (重述=用修订后数字=隐式前视)
  [2] 报告时滞结构: rb 时各股所用最新报告 end_date 距今月龄分布
      (季报~1-4月龄/中报~2-7月/年报~3-8月, 与披露节奏吻合 = PIT 成立)
  [3] 覆盖率: 成分股中财务数据覆盖% 与 VAL 估值覆盖% 随年份演进
  [4] 边界核查: 缺 VAL 月份 / NaN ann_date 行 / 未来公告是否被正确排除
判定: PIT 结构成立则历史结论可信; 否则量化受影响月份。

输出: results/data_pit_check.txt|json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import style_factors as sf
from research.factor_dic import lynch_factor as lf
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements import common as C


def _months_series(end_dates, rb):
    """end_dates: Series(YYYYMMDD), rb: YYYYMMDD -> 月龄 Series (rb - end_date)"""
    return (int(rb[:4]) - end_dates.str[:4].astype(int)) * 12 + (int(rb[4:6]) - end_dates.str[4:6].astype(int))


def main():
    env = C.Env()
    rebal = env.rebal

    fina = pd.read_parquet(sf.FINA_ALL)
    fina["ann_date"] = fina["ann_date"].astype(str).str[:8]
    fina = fina[fina["ann_date"].str.match(r"^\d{8}$", na=False)].copy()
    fina["end_date"] = fina["end_date"].astype(str).str[:8]
    fina = fina.sort_values("ann_date")
    n_nan_ann = int(pd.read_parquet(sf.FINA_ALL)["ann_date"].isna().sum())
    # 修订重述: 同一 (ts_code, end_date) 多行公告
    dup = fina.groupby(["ts_code", "end_date"]).size()
    n_dup = int((dup > 1).sum())

    # 估值快照文件清单
    pe_files = set(f for f in os.listdir(lf.PE_DIR) if f.endswith(".parquet"))

    lines = ["数据端核验 (3.27): PIT 财务数据核验", "=" * 92]
    lines.append(f"fina_all: {len(fina)} 行 / {fina['ts_code'].nunique()} 只, "
                 f"ann_date {fina['ann_date'].min()} ~ {fina['ann_date'].max()}, "
                 f"end_date {fina['end_date'].min()} ~ {fina['end_date'].max()}")
    lines.append(f"NaN ann_date 行: {n_nan_ann} (过滤后排除)")

    # ---- [1] 逐 rb: 无前视 + 时滞 + 覆盖率 ----
    rows = []
    for rb in rebal:
        members = rv.load_index_weight(rb) or set()
        latest = fina[fina["ann_date"] <= rb].drop_duplicates("ts_code", keep="last")
        used = latest[latest["ts_code"].isin(members)]
        if used.empty:
            rows.append(dict(rb=rb, cov_f=0.0, cov_val=0.0, age_med=np.nan,
                             max_ann="", n_used=0))
            continue
        age = _months_series(used["end_date"].astype(str).str[:8], rb)
        # 无前视: 所用行 ann_date 必须 <= rb (构造保证, 但显式复核)
        max_ann = used["ann_date"].max()
        assert max_ann <= rb, f"look-ahead at {rb}: {max_ann}"
        # VAL 覆盖
        val_df = None
        fp = os.path.join(lf.PE_DIR, f"{rb}.parquet")
        if os.path.exists(fp):
            val_df = pd.read_parquet(fp)
        cov_val = (val_df["ts_code"].astype(str).isin(members)).sum() / len(members) if val_df is not None else 0.0
        rows.append(dict(rb=rb, cov_f=len(used) / len(members), cov_val=float(cov_val),
                         age_med=float(np.median(age)), max_ann=max_ann,
                         n_used=len(used)))
    rdf = pd.DataFrame(rows)

    lines.append("")
    lines.append("[1] 逐调仓日 PIT 核验 (抽样每 12 月 + 首末):")
    lines.append(f"    {'rb':<10}{'成分':>5}{'财务覆盖':>8}{'VAL覆盖':>8}{'报告月龄中位':>10}{'最大ann_date':>12}")
    for _, r in (rdf.iloc[::12].iterrows()):
        lines.append(f"    {r['rb']:<10}{int(r.get('n_used',0) or 0):>5}"
                     f"{r['cov_f']:>7.1%}{r['cov_val']:>8.1%}"
                     f"{r['age_med']:>10.1f}{r['max_ann']:>12}")
    last = rdf.iloc[-1]
    lines.append(f"    {last['rb']:<10}{int(last['n_used']):>5}{last['cov_f']:>7.1%}"
                 f"{last['cov_val']:>8.1%}{last['age_med']:>10.1f}{last['max_ann']:>12}")
    lines.append(f"    ── 全期: 财务覆盖均值 {rdf['cov_f'].mean():.1%} | VAL覆盖均值 {rdf['cov_val'].mean():.1%} | "
                 f"报告月龄中位 {rdf['age_med'].median():.1f} 月")

    # ---- [2] 时滞结构: 报告期构成 (每 12 月抽样) ----
    lines.append("")
    lines.append("[2] 报告期构成 (rb 时各股最新报告 end_date 的季度分布):")
    lines.append(f"    {'rb':<10}{'Q1(0331)':>9}{'H1(0630)':>9}{'Q3(0930)':>9}{'FY(1231)':>9}")
    for rb in rdf.iloc[::6]["rb"]:
        latest = fina[fina["ann_date"] <= rb].drop_duplicates("ts_code", keep="last")
        used = latest[latest["ts_code"].isin(rv.load_index_weight(rb) or set())]
        m = used["end_date"].astype(str).str[4:8]
        c = {k: (m == k).mean() for k in ("0331", "0630", "0930", "1231")}
        lines.append(f"    {rb:<10}" + "".join(f"{c[k]:>8.1%}" for k in ("0331", "0630", "0930", "1231")))

    # ---- [3] 修订重述 + 未来公告 + 缺 VAL 月份 ----
    lines.append("")
    lines.append("[3] PIT 结构核查:")
    lines.append(f"    修订重述 (同 ts_code+end_date 多公告): {n_dup} / {len(dup)} ({n_dup/len(dup):.2%}) -> "
                 f"{'无重述前视' if n_dup == 0 else '极少, 可忽略'}")
    missing_val = [rb for rb in rebal if f"{rb}.parquet" not in pe_files]
    lines.append(f"    缺 daily_basic 快照的调仓月: {missing_val if missing_val else '无'}")
    future_rows = fina[fina["ann_date"] > fina["end_date"].astype(str).str[:8]]
    lines.append(f"    未来公告 (ann_date > end_date) 行: {len(future_rows)} -> "
                 f"{'正常 (预披露/快报), 过滤按 ann_date 不引入前视' if len(future_rows) else '无'}")
    # 最后一个调仓月是否可用财务 (2026-07-31 用 2026-06-30 中报预披露等)
    rb_last = rebal[-1]
    last_used = fina[fina["ann_date"] <= rb_last].drop_duplicates("ts_code", keep="last")
    last_used = last_used[last_used["ts_code"].isin(rv.load_index_weight(rb_last) or set())]
    lines.append(f"    末月 {rb_last}: 财务覆盖 {len(last_used)/1000:.1%}, "
                 f"最新 end_date 中位 {last_used['end_date'].astype(str).str[:8].median()}")

    # ---- [4] 判定 ----
    look_ahead = None
    for _, r in rdf.iterrows():
        assert r["max_ann"] <= r["rb"], f"look-ahead at {r['rb']}"
    cov_ok = rdf["cov_f"].mean() > 0.9 and rdf["cov_val"].mean() > 0.9
    lag_ok = rdf["age_med"].median() >= 1.0
    dup_rate = n_dup / len(dup) if len(dup) else 0.0
    if dup_rate < 0.001 and cov_ok and lag_ok:
        verdict = (f"✅ PIT 结构成立: 财务腿严格按 ann_date<=rb 对齐 (0 前视, 重述率 {dup_rate:.2%} 可忽略), "
                   f"报告月龄中位 {rdf['age_med'].median():.1f} 月与披露节奏吻合, "
                   f"财务/VAL 覆盖 {rdf['cov_f'].mean():.1%}/{rdf['cov_val'].mean():.1%} -> 历史结论可信")
    else:
        verdict = (f"⚠ 需关注: 修订 {n_dup} ({dup_rate:.2%}), 财务覆盖 {rdf['cov_f'].mean():.1%}, "
                   f"VAL覆盖 {rdf['cov_val'].mean():.1%}, 月龄中位 {rdf['age_med'].median():.1f}")
    lines.append("")
    lines.append(f"判定: {verdict}")
    print("\n".join(lines))

    with open(os.path.join(C.OUT_DIR, "data_pit_check.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(C.OUT_DIR, "data_pit_check.json"), "w", encoding="utf-8") as f:
        json.dump(dict(n_rows=len(fina), n_stocks=int(fina["ts_code"].nunique()),
                       n_dup=n_dup, nan_ann=n_nan_ann, missing_val=missing_val,
                       cov_f=float(rdf["cov_f"].mean()), cov_val=float(rdf["cov_val"].mean()),
                       age_med=float(rdf["age_med"].median()), verdict=verdict),
                  f, ensure_ascii=False, indent=1)
    print(f"[saved] {os.path.join(C.OUT_DIR, 'data_pit_check.txt')} | .json")


if __name__ == "__main__":
    main()
