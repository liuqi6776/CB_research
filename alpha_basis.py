# -*- coding: utf-8 -*-
"""时变α模块: IM 贴水驱动的中证1000增强 α (2026-07-18)

α_t = -1 × 最近 60 日历日 IM 季月合约年化基差均值 (无前视, 只用 t 日及之前采样)

用法:
  python alpha_basis.py           # 只打印当前时变α统计
  python alpha_basis.py update    # 每日增量更新 (现货+期货采样), 然后打印统计

作为模块:
  from alpha_basis import load_alpha_series, update_basis_samples
  alpha, meta = load_alpha_series()        # -> pd.Series (date索引, 年化α小数), 前导NaN由调用方填充
  update_basis_samples()                   # 增量更新 im_ann_basis_samples.csv

数据文件 (workspace 根目录):
  im_ann_basis_samples.csv   采样点 (date, contract, dte, fut, spot, basis_pct, ann_basis_pct)
  zz1000_spot_daily.csv      现货日线缓存 (date, close), 首建自 zz1000_spot_a/b.csv, 之后新浪增量
"""
import os, sys, time, datetime as dt
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES_CSV = os.path.join(ROOT, "im_ann_basis_samples.csv")
SPOT_CACHE = os.path.join(ROOT, "zz1000_spot_daily.csv")

FALLBACK_ALPHA = 0.093   # 数据缺失时回退常数 (2023-01~2026-07 全样本均值)
ROLL_DAYS = 60           # 滚动窗口 (日历日)
MIN_PERIODS = 20
DTE_MIN, DTE_MAX = 20, 120   # 采样窗口: 到期前 20~120 天


def third_friday(year, month):
    """中金所合约到期日 = 合约月份第三个周五"""
    d = dt.date(year, month, 15)
    return d + dt.timedelta(days=(4 - d.weekday()) % 7)


def quarterly_contracts(y0, y1):
    """季月合约列表 [(year, month, symbol)], 如 IM2609"""
    out = []
    for y in range(y0, y1 + 1):
        for m in (3, 6, 9, 12):
            out.append((y, m, f"IM{str(y)[2:]}{m:02d}"))
    return out


# ---------------- 时变 α 序列 ----------------

def load_alpha_series(samples_csv=SAMPLES_CSV, roll_days=ROLL_DAYS,
                      min_periods=MIN_PERIODS, fallback=FALLBACK_ALPHA):
    """读取采样点 -> 时变α序列 (date 索引, 年化小数).

    返回 (alpha_series, meta)。序列前 ~min_periods 个有效日为 NaN, 由调用方
    ffill/fillna(fallback) 处理; 采样起点之前的历史一律用 fallback 常数。
    """
    smp = pd.read_csv(samples_csv)
    smp["date"] = pd.to_datetime(smp["date"])
    daily = smp.groupby("date")["ann_basis_pct"].mean()   # 当日各合约均值 (负=贴水)
    alpha = (-daily / 100.0).rolling(roll_days, min_periods=min_periods).mean()
    alpha.name = "alpha"
    valid = alpha.dropna()
    meta = {
        "start": str(alpha.index.min().date()), "end": str(alpha.index.max().date()),
        "valid_start": str(valid.index.min().date()) if len(valid) else None,
        "mean": float(valid.mean()), "min": float(valid.min()), "max": float(valid.max()),
        "last": float(valid.iloc[-1]) if len(valid) else fallback,
        "n_samples": len(smp), "fallback": fallback,
    }
    return alpha, meta


# ---------------- 现货缓存 (增量) ----------------

def _load_spot_cache():
    if os.path.exists(SPOT_CACHE):
        sp = pd.read_csv(SPOT_CACHE)
    else:  # 首建: 合并 iFinD 导出的 a/b 两段
        parts = []
        for f in ("zz1000_spot_a.csv", "zz1000_spot_b.csv"):
            p = os.path.join(ROOT, f)
            if os.path.exists(p):
                d = pd.read_csv(p)
                d = d.rename(columns={"time": "date"})[["date", "close"]]
                parts.append(d)
        if not parts:
            return pd.DataFrame(columns=["date", "close"])
        sp = pd.concat(parts, ignore_index=True)
    sp["date"] = pd.to_datetime(sp["date"])
    sp = sp.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return sp


def update_spot():
    """用新浪指数日线增量更新现货缓存。返回 (spot Series, 新增行数)。"""
    import akshare as ak
    sp = _load_spot_cache()
    last = sp["date"].max() if len(sp) else pd.Timestamp("2000-01-01")
    df = ak.stock_zh_index_daily(symbol="sh000852")
    df["date"] = pd.to_datetime(df["date"])
    new = df[df["date"] > last][["date", "close"]]
    if len(new):
        sp = pd.concat([sp, new], ignore_index=True)
        sp = sp.drop_duplicates("date").sort_values("date").reset_index(drop=True)
        sp.to_csv(SPOT_CACHE, index=False)
    print(f"现货缓存: {len(sp)} 行, 最新 {sp['date'].max().date()}, 新增 {len(new)} 行")
    return sp.set_index("date")["close"], len(new)


# ---------------- 期货采样 (增量) ----------------

def update_basis_samples():
    """每日增量更新: 现货 -> 各存活季月合约新采样点 -> 合并去重写盘。

    非破坏性: 仅在全部抓取完成后一次性重写 CSV; 任一合约失败只跳过该合约。
    """
    spot, _ = update_spot()

    existing = pd.DataFrame(columns=["date", "contract", "dte", "fut", "spot",
                                     "basis_pct", "ann_basis_pct"])
    if os.path.exists(SAMPLES_CSV):
        existing = pd.read_csv(SAMPLES_CSV)
        existing["date"] = pd.to_datetime(existing["date"])

    today = dt.date.today()
    max_by_contract = existing.groupby("contract")["date"].max().to_dict() if len(existing) else {}

    # 只抓取"采样窗口已开且未采完"的合约:
    #   窗口 = [expiry-120d, expiry-120d 之后最后一个 <= expiry-20d 的现货交易日]
    y_now = today.year
    todo = []
    for y, m, sym in quarterly_contracts(y_now - 1, y_now + 1):
        expiry = third_friday(y, m)
        window_open = pd.Timestamp(expiry) - pd.Timedelta(days=DTE_MAX)
        window_close = pd.Timestamp(expiry) - pd.Timedelta(days=DTE_MIN)
        if window_open > pd.Timestamp(today):
            continue  # 窗口未开 (远月合约)
        cand = spot.index[spot.index <= window_close]
        if not len(cand):
            continue
        last_possible = cand.max()           # 窗口内最后一个可能交易日
        have = max_by_contract.get(sym)
        if have is not None and have >= last_possible:
            continue  # 已采完
        todo.append((y, m, sym, expiry, have))

    print(f"待更新合约: {[s for _, _, s, _, _ in todo] if todo else '无 (已是最新)'}")
    import akshare as ak
    new_records = []
    for y, m, sym, expiry, have in todo:
        try:
            df = ak.futures_zh_daily_sina(symbol=sym)
            if df is None or len(df) == 0:
                print(f"{sym}: 无数据"); continue
            df["date"] = pd.to_datetime(df["date"])
            n0 = 0
            for d, f in df.set_index("date")["close"].items():
                if have is not None and d <= have:
                    continue
                dte = (pd.Timestamp(expiry) - d).days
                if DTE_MIN <= dte <= DTE_MAX and d in spot.index:
                    s = spot[d]
                    new_records.append({"date": d, "contract": sym, "dte": dte,
                                        "fut": f, "spot": s,
                                        "basis_pct": (f / s - 1) * 100,
                                        "ann_basis_pct": (f / s - 1) * 365 / dte * 100})
                    n0 += 1
            print(f"{sym}: 新增 {n0} 个采样点 (到期 {expiry})")
            time.sleep(0.3)
        except Exception as e:
            print(f"{sym}: 抓取失败, 跳过 ({e})")

    if new_records:
        out = pd.concat([existing, pd.DataFrame(new_records)], ignore_index=True)
        out = out.drop_duplicates(["date", "contract"]).sort_values(["date", "contract"])
        out.to_csv(SAMPLES_CSV, index=False)
        print(f"采样点: {len(existing)} -> {len(out)} (新增 {len(out) - len(existing)})")
    else:
        print(f"采样点: {len(existing)} (无新增)")
    return len(new_records)


if __name__ == "__main__":
    if "update" in sys.argv:
        update_basis_samples()
    alpha, meta = load_alpha_series()
    print(f"\n时变α: {meta['start']} ~ {meta['end']} (有效自 {meta['valid_start']})")
    print(f"  均值 {meta['mean']:.2%} | 区间 [{meta['min']:.2%}, {meta['max']:.2%}] | "
          f"最新 {meta['last']:.2%} | 采样点 {meta['n_samples']} | 回退常数 {meta['fallback']:.1%}")
