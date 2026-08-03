# -*- coding: utf-8 -*-
"""
pit_data.py — PIT (Point-In-Time) 数据层构建脚本 / PIT data layer builder

修复 study_007 的前视偏差问题：
Fixes look-ahead bias in study_007:
  1. 财报按 ann_date 做 asof 对齐（不再用最新一期填全部历史）
     Fundamentals are asof-aligned by announcement date (no more filling all history with latest report)
  2. 收益率直接采用 pct_chg（已验证 pre_close 处理了除权除息，等价于复权收益）
     Returns use pct_chg directly (pre_close already handles corporate actions => adjusted returns)
  3. 使用真实流通市值 circ_mv（万元），不再用成交额冒充
     Uses real float market cap circ_mv (10k CNY) instead of amount

产出 / Outputs (relative to this script's directory):
  data/panel.parquet      — 日频面板 2019-07-01 ~ 2025-12-31 (2019H2 用于因子预热 / factor warm-up)
  data/funda_pit.parquet  — PIT 财报面板 2023-05-01 ~ 2025-12-31
  data/panel_stats.json   — 构建统计 / build statistics

运行 / Run:  "$DAIMON_USER_PYTHON" pit_data.py   (需要 pyarrow / requires pyarrow)
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------------- 配置 / Config -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TMP_DIR = os.path.join(DATA_DIR, "_tmp_yearly")

DAY1_DIR = r"D:\iquant_data\data_v2\data_day1"        # 日行情 / daily quotes
OTHER_DIR = r"D:\iquant_data\data_v2\other_day1"      # 估值/市值 / valuation & market cap
INDUSTRY_FP = r"D:\iquant_data\data_v2\industry1\industry.parquet"
FINA_FP = r"D:\iquant_data\data_v2\fundamental1\fina_indicator_cache.parquet"

PANEL_START = "20190701"   # 面板起点（含 2019H2 预热期 / warm-up）
PANEL_END = "20251231"     # 面板终点
FUNDA_START = "20230501"   # PIT 财报面板起点（受财报数据 ann_date 范围限制）
FUNDA_END = "20251231"

DAY1_COLS = ["ts_code", "trade_date", "open", "close", "pre_close",
             "pct_chg", "vol", "amount"]
OTHER_COLS = ["ts_code", "trade_date", "turnover_rate", "pe", "pb", "circ_mv"]
FUNDA_KEEP = ["roe", "or_yoy", "netprofit_yoy", "grossprofit_margin",
              "netprofit_margin", "debt_to_assets", "quick_ratio"]

PANEL_COLS = ["ts_code", "trade_date", "open", "close", "pre_close",
              "daily_ret", "amount", "vol", "circ_mv", "pe", "pb",
              "turnover_rate", "industry", "name", "list_date",
              "is_st", "limit_up", "limit_down", "mkt_ret"]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)


# ----------------------- 涨跌停阈值 / Limit thresholds -----------------------
def limit_threshold(codes: pd.Series, dates: pd.Series) -> pd.Series:
    """
    按板块返回涨跌停阈值(%) / Return price-limit threshold (%) by board.
      主板 60/000/001/002/003: 9.8
      创业板 300: 2020-08-24 起 19.8，此前 9.8
      科创板 688: 19.8
      北交所 4xx/8xx/920: 29.8
    """
    sym = codes.str[:6]
    th = pd.Series(9.8, index=codes.index)

    is_cyb = sym.str.startswith("300")
    th[is_cyb & (dates >= "20200824")] = 19.8

    is_kcb = sym.str.startswith("688")
    th[is_kcb] = 19.8

    is_bj = (sym.str.startswith("4") | sym.str.startswith("8")
             | sym.str.startswith("920"))
    th[is_bj] = 29.8
    return th


# --------------------------- 年文件列表 / File list ---------------------------
def list_files(folder, start, end):
    out = []
    for fp in glob.glob(os.path.join(folder, "*.parquet")):
        d = os.path.splitext(os.path.basename(fp))[0]
        if d.isdigit() and start <= d <= end:
            out.append((d, fp))
    out.sort()
    return out


def safe_read(fp, cols):
    """容错读取：缺列补 NaN / Tolerant read: fill missing columns with NaN."""
    try:
        df = pd.read_parquet(fp)
    except Exception as e:
        print(f"  [WARN] 读取失败 skip {fp}: {e}")
        return None
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


# --------------------------- 主构建流程 / Main build ---------------------------
def build_panel():
    print("== 读取行业快照 / loading industry snapshot ==")
    ind = pd.read_parquet(INDUSTRY_FP)
    ind = ind[["ts_code", "industry", "name", "list_date"]].drop_duplicates("ts_code")

    day_files = list_files(DAY1_DIR, PANEL_START, PANEL_END)
    print(f"日行情文件数 / daily files: {len(day_files)}")

    # 按年分组处理，控制内存 / process year-by-year to bound memory
    years = {}
    for d, fp in day_files:
        years.setdefault(d[:4], []).append((d, fp))

    n_rows_total = 0
    for year in sorted(years):
        files = years[year]
        parts = []
        for d, fp in files:
            df = safe_read(fp, DAY1_COLS)
            if df is not None and len(df):
                parts.append(df)
        if not parts:
            continue
        q = pd.concat(parts, ignore_index=True)
        q = q.drop_duplicates(["ts_code", "trade_date"])

        # 合并估值/市值 / merge valuation & market cap
        oparts = []
        for d, _ in files:
            ofp = os.path.join(OTHER_DIR, d + ".parquet")
            if os.path.exists(ofp):
                o = safe_read(ofp, OTHER_COLS)
                if o is not None and len(o):
                    oparts.append(o)
        if oparts:
            od = pd.concat(oparts, ignore_index=True).drop_duplicates(["ts_code", "trade_date"])
            q = q.merge(od, on=["ts_code", "trade_date"], how="left")
        else:
            for c in ["turnover_rate", "pe", "pb", "circ_mv"]:
                q[c] = np.nan

        # 合并行业快照（左连接，退市股保留，industry 标 UNKNOWN）
        # merge industry snapshot (left join; delisted stocks kept with UNKNOWN)
        q = q.merge(ind, on="ts_code", how="left")
        q["industry"] = q["industry"].fillna("UNKNOWN")
        q["name"] = q["name"].fillna("")
        q["list_date"] = q["list_date"].fillna("")

        # 派生列 / derived columns
        q["daily_ret"] = q["pct_chg"] / 100.0
        q["is_st"] = q["name"].str.contains("ST", case=False, na=False)
        th = limit_threshold(q["ts_code"], q["trade_date"])
        q["limit_up"] = q["pct_chg"] >= th
        q["limit_down"] = q["pct_chg"] <= -th
        q["mkt_ret"] = q.groupby("trade_date")["daily_ret"].transform("mean")

        q = q[PANEL_COLS]
        q["trade_date"] = q["trade_date"].astype(str)

        # float32 降精度省内存/磁盘（对研究足够）/ downcast to float32
        fcols = ["open", "close", "pre_close", "daily_ret", "amount", "vol",
                 "circ_mv", "pe", "pb", "turnover_rate", "mkt_ret"]
        q[fcols] = q[fcols].astype("float32")

        tmp_fp = os.path.join(TMP_DIR, f"panel_{year}.parquet")
        q.to_parquet(tmp_fp, index=False)
        n_rows_total += len(q)
        print(f"  {year}: {len(q):,} rows -> {tmp_fp}")
        del q, parts, oparts

    # 流式合并年度文件为单一 panel.parquet / stream-merge yearly files
    print("== 合并年度文件 / merging yearly files ==")
    panel_fp = os.path.join(DATA_DIR, "panel.parquet")
    tmp_files = sorted(glob.glob(os.path.join(TMP_DIR, "panel_*.parquet")))
    writer = None
    for fp in tmp_files:
        t = pq.read_table(fp)
        if writer is None:
            writer = pq.ParquetWriter(panel_fp, t.schema, compression="snappy")
        writer.write_table(t)
    if writer:
        writer.close()
    for fp in tmp_files:
        os.remove(fp)
    os.rmdir(TMP_DIR)
    print(f"panel.parquet 完成 / done: {n_rows_total:,} rows")
    return panel_fp, n_rows_total


def build_funda_pit(panel_fp):
    print("== 构建 PIT 财报面板 / building PIT fundamentals ==")
    fin = pd.read_parquet(FINA_FP)
    need = ["ts_code", "ann_date", "end_date"] + FUNDA_KEEP
    for c in need:
        if c not in fin.columns:
            fin[c] = np.nan
    fin = fin[need].copy()

    # 清洗 ann_date；同一 (ts_code, ann_date) 取 end_date 最新一期
    # clean ann_date; for same (ts_code, ann_date) keep the latest end_date
    fin["ann_dt"] = pd.to_datetime(fin["ann_date"], format="%Y%m%d", errors="coerce")
    fin = fin.dropna(subset=["ann_dt", "ts_code"])
    fin = fin.sort_values(["ts_code", "ann_dt", "end_date"])
    fin = fin.drop_duplicates(["ts_code", "ann_dt"], keep="last")

    # 面板内的交易日序列 / trade dates from panel
    dates = pd.read_parquet(panel_fp, columns=["trade_date"])["trade_date"].unique()
    dates = pd.Series(dates).sort_values()
    dates = dates[(dates >= FUNDA_START) & (dates <= FUNDA_END)]
    trade_dt = pd.to_datetime(dates, format="%Y%m%d")

    stocks = fin["ts_code"].unique()
    # 左表：funda 覆盖股票 × 交易日 / left frame: covered stocks x trade dates
    left = pd.MultiIndex.from_product(
        [stocks, trade_dt], names=["ts_code", "trade_dt"]).to_frame(index=False)

    # asof 对齐：ann_date <= trade_date 前一日历日  <=>  ann_dt < trade_dt（严格早于）
    # asof align: ann_date <= previous calendar day of trade_date <=> ann_dt < trade_dt
    left = left.sort_values("trade_dt")
    right = fin.sort_values("ann_dt")
    out = pd.merge_asof(
        left, right,
        left_on="trade_dt", right_on="ann_dt",
        by="ts_code", allow_exact_matches=False)

    out["trade_date"] = out["trade_dt"].dt.strftime("%Y%m%d")
    out = out[["ts_code", "trade_date"] + FUNDA_KEEP]
    out[FUNDA_KEEP] = out[FUNDA_KEEP].astype("float32")
    out = out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    fp = os.path.join(DATA_DIR, "funda_pit.parquet")
    out.to_parquet(fp, index=False)
    n_asof = out["roe"].notna().sum()
    print(f"funda_pit.parquet 完成 / done: {len(out):,} rows, "
          f"asof 命中行 / rows with fundamentals: {n_asof:,} "
          f"({n_asof/len(out)*100:.1f}%)")
    return fp, len(out), int(n_asof)


def main():
    panel_fp, n_rows = build_panel()
    funda_fp, n_funda, n_asof = build_funda_pit(panel_fp)

    # 基础统计 / basic stats
    pf = pq.ParquetFile(panel_fp)
    stats = {
        "panel_path": panel_fp,
        "funda_path": funda_fp,
        "panel_rows": n_rows,
        "funda_rows": n_funda,
        "funda_asof_hits": n_asof,
    }
    with open(os.path.join(DATA_DIR, "panel_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("统计已写入 / stats written:", stats)


if __name__ == "__main__":
    main()
