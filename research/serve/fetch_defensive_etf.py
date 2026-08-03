# -*- coding: utf-8 -*-
"""
拉取避险/替代资产 ETF 日线 (akshare) -> {serve}/data/etf/{code}.parquet
用于验证: RS12 弱市段从"持 512100"改为"持货基/国债/黄金/红利"能否压低回撤

注意: 输出目录与 defensive_asset_bt.py 的 ETF_DIR 保持同一表达式,
      保证两端脚本可组成端到端流程 (数据源: akshare 不复权, 与工作区
      tushare fund_daily 数据可能存在差异, 使用前需对齐).
"""
import os
import time

import akshare as ak
import pandas as pd

# 与 defensive_asset_bt.py 的 ETF_DIR 一致: <serve>/data/etf
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "serve", "data", "etf")
os.makedirs(OUT, exist_ok=True)

# code: (akshare symbol, 名称)
TARGETS = {
    "511880.SH": ("511880", "银华日利(货币)"),
    "511990.SH": ("511990", "华宝添益(货币)"),
    "511010.SH": ("511010", "国债ETF"),
    "511260.SH": ("511260", "十年国债ETF"),
    "518880.SH": ("518880", "黄金ETF"),
    "510880.SH": ("510880", "红利ETF"),
    "512890.SH": ("512890", "红利低波ETF"),
    "512400.SH": ("512400", "有色金属ETF"),
    "515790.SH": ("515790", "光伏ETF"),
}


def main():
    for code, (sym, name) in TARGETS.items():
        fp = os.path.join(OUT, f"{code}.parquet")
        if os.path.exists(fp):
            print(f"[skip] {code} {name} 已存在")
            continue
        try:
            df = ak.fund_etf_hist_em(symbol=sym, period="daily",
                                     start_date="20191201", end_date="20260802",
                                     adjust="")
            if df is None or df.empty:
                print(f"[warn] {code} {name} 空数据")
                continue
            df = df.rename(columns={"日期": "trade_date", "收盘": "close"})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
            df = df[["trade_date", "close"]].dropna().set_index("trade_date").sort_index()
            df.to_parquet(fp)
            n = len(df)
            ret = df["close"].pct_change().dropna()
            print(f"[ok]   {code} {name}: {df.index[0]}~{df.index[-1]} n={n} "
                  f"累计={((1+ret).prod()-1):.1%} 年化≈{((1+ret).prod())**(242/n)-1:.2%} "
                  f"最大回撤={((df['close'].cummax()-df['close'])/df['close'].cummax()).max():.2%}")
        except Exception as e:
            print(f"[err]  {code} {name}: {e}")
        time.sleep(0.8)


if __name__ == "__main__":
    main()
