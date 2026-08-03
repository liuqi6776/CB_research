# -*- coding: utf-8 -*-
"""
从 tushare 补抓林奇因子(PEG)所需数据, 扩展到 2020-2026:
  1) pe_ttm: daily_basic 按 78 个调仓日拉全市场 -> data/pe_ttm/{rb}.parquet (ts_code,pe_ttm,pe)
  2) netprofit_yoy: fina_indicator 逐股拉全历史 -> data/fina_yoy.parquet (ts_code,ann_date,netprofit_yoy)
     (fina_indicator 强制 ts_code, 无法按日期批量; 逐股 2058 次, 限频 0.3s+重试)
"""
import os
import sys
import time

import pandas as pd
import tushare as ts

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings
from research.factor_dic import run_validation as rv

BASE = os.path.dirname(os.path.abspath(__file__))
PE_DIR = os.path.join(BASE, "data", "pe_ttm")
FIN_DIR = os.path.join(BASE, "data")
os.makedirs(PE_DIR, exist_ok=True)


def main():
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    parts = sys.argv[1:] or ["pe", "fina"]
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]

    if "pe" in parts:
        # 1) pe_ttm: 调仓日全市场
        print("== 1/2 pe_ttm (daily_basic) ==")
        need = [rb for rb in rebal if not os.path.exists(os.path.join(PE_DIR, f"{rb}.parquet"))]
        print(f"待拉 {len(need)} 个调仓日 (已缓存 {len(rebal)-len(need)})")
        for rb in need:
            for attempt in range(3):
                try:
                    df = pro.daily_basic(trade_date=rb, fields="ts_code,pe_ttm,pe")
                    if df is not None and not df.empty:
                        df.to_parquet(os.path.join(PE_DIR, f"{rb}.parquet"))
                        break
                except Exception as e:
                    if attempt == 2:
                        print(f"  [fail] {rb}: {e}")
                time.sleep(0.35)
            else:
                continue
        print(f"pe_ttm 完成: {len(os.listdir(PE_DIR))}/{len(rebal)}")

    if "fina" not in parts:
        return
    # 2) netprofit_yoy: 逐股全历史
    print("== 2/2 netprofit_yoy (fina_indicator 逐股) ==")
    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"成分股 {len(all_codes)} 只")

    FINA_FP = os.path.join(FIN_DIR, "fina_yoy.parquet")
    if os.path.exists(FINA_FP):
        have = set(pd.read_parquet(FINA_FP, columns=["ts_code"])["ts_code"].unique())
    else:
        have = set()
    todo = [c for c in all_codes if c not in have]
    print(f"待拉 {len(todo)} 只 (已缓存 {len(have)})")

    chunks = []
    fail = []
    last_save = 0
    for i, code in enumerate(todo):
        for attempt in range(3):
            try:
                df = pro.fina_indicator(ts_code=code, fields="ts_code,ann_date,netprofit_yoy")
                if df is not None and not df.empty:
                    chunks.append(df)
                break
            except Exception as e:
                if attempt == 2:
                    fail.append((code, str(e)))
                    print(f"  [fail] {code}: {e}", flush=True)
                time.sleep(0.5)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
            # 每 300 只 checkpoint 一次, 防中断丢全部
            if (i + 1) - last_save >= 300 and chunks:
                big = pd.concat(chunks, ignore_index=True)
                if os.path.exists(FINA_FP):
                    old = pd.read_parquet(FINA_FP)
                    big = pd.concat([old, big], ignore_index=True)
                big = big.drop_duplicates(subset=["ts_code", "ann_date"], keep="last")
                big.to_parquet(FINA_FP)
                chunks = []
                last_save = i + 1
                print(f"  [checkpoint] {big['ts_code'].nunique()} 只已存", flush=True)
        time.sleep(0.25)
    if chunks:
        big = pd.concat(chunks, ignore_index=True)
        if os.path.exists(FINA_FP):
            old = pd.read_parquet(FINA_FP)
            big = pd.concat([old, big], ignore_index=True)
        big = big.drop_duplicates(subset=["ts_code", "ann_date"], keep="last")
        big.to_parquet(FINA_FP)
        print(f"fina_yoy 保存: {len(big)} 行, 覆盖 {big['ts_code'].nunique()} 只", flush=True)
    print(f"失败 {len(fail)} 只: {fail[:10]}", flush=True)


if __name__ == "__main__":
    main()
