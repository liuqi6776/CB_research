# -*- coding: utf-8 -*-
"""
补抓风格因子(Buffett/价值/质量/成长)所需全字段数据, 2020-2026:
  1) daily_basic 全字段(78 调仓日): pe,pe_ttm,pb,ps,ps_ttm,dv_ttm,dv_ratio,total_mv,circ_mv
     -> data/pe_ttm/{rb}.parquet (覆盖旧文件, 列更全)
  2) fina_indicator 全字段(2058 只): roe,eps,grossprofit_margin,debt_to_assets,
     q_sales_yoy,q_profit_yoy,ocfps 等 -> data/fina_all.parquet
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
FINA_ALL = os.path.join(BASE, "data", "fina_all.parquet")
os.makedirs(PE_DIR, exist_ok=True)

BB_FIELDS = "ts_code,pe,pe_ttm,pb,ps,ps_ttm,dv_ttm,dv_ratio,total_mv,circ_mv"
FIN_FIELDS = ("ts_code,ann_date,end_date,eps,roe,roe_dt,netprofit_yoy,dt_netprofit_yoy,"
              "grossprofit_margin,debt_to_assets,q_sales_yoy,q_profit_yoy,ocfps")


def main():
    parts = sys.argv[1:] or ["bb", "fina"]
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal = sorted(months.values())[:-1]

    if "bb" in parts:
        print("== 1/2 daily_basic 全字段 ==", flush=True)
        for rb in rebal:
            for attempt in range(3):
                try:
                    df = pro.daily_basic(trade_date=rb, fields=BB_FIELDS)
                    if df is not None and not df.empty:
                        df.to_parquet(os.path.join(PE_DIR, f"{rb}.parquet"))
                        break
                except Exception as e:
                    if attempt == 2:
                        print(f"  [fail] {rb}: {e}", flush=True)
                time.sleep(0.35)
        print(f"daily_basic 完成: {len(os.listdir(PE_DIR))}/{len(rebal)}", flush=True)

    if "fina" not in parts:
        return
    print("== 2/2 fina_indicator 全字段 ==", flush=True)
    all_codes = set()
    for rb in rebal:
        m = rv.load_index_weight(rb)
        if m:
            all_codes |= m
    all_codes = sorted(all_codes)
    print(f"成分股 {len(all_codes)} 只", flush=True)

    if os.path.exists(FINA_ALL):
        have = set(pd.read_parquet(FINA_ALL, columns=["ts_code"])["ts_code"].unique())
    else:
        have = set()
    todo = [c for c in all_codes if c not in have]
    print(f"待拉 {len(todo)} 只 (已缓存 {len(have)})", flush=True)

    chunks = []
    fail = []
    last_save = 0
    for i, code in enumerate(todo):
        for attempt in range(3):
            try:
                df = pro.fina_indicator(ts_code=code, fields=FIN_FIELDS)
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
            if (i + 1) - last_save >= 300 and chunks:
                big = pd.concat(chunks, ignore_index=True)
                if os.path.exists(FINA_ALL):
                    old = pd.read_parquet(FINA_ALL)
                    big = pd.concat([old, big], ignore_index=True)
                big = big.drop_duplicates(subset=["ts_code", "ann_date"], keep="last")
                big.to_parquet(FINA_ALL)
                chunks = []
                last_save = i + 1
                print(f"  [checkpoint] {big['ts_code'].nunique()} 只已存", flush=True)
        time.sleep(0.25)
    if chunks:
        big = pd.concat(chunks, ignore_index=True)
        if os.path.exists(FINA_ALL):
            old = pd.read_parquet(FINA_ALL)
            big = pd.concat([old, big], ignore_index=True)
        big = big.drop_duplicates(subset=["ts_code", "ann_date"], keep="last")
        big.to_parquet(FINA_ALL)
        print(f"fina_all 保存: {len(big)} 行, 覆盖 {big['ts_code'].nunique()} 只", flush=True)
    print(f"失败 {len(fail)} 只: {fail[:10]}", flush=True)


if __name__ == "__main__":
    main()
