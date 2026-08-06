# -*- coding: utf-8 -*-
"""快速诊断: 512100 ETF 数据边界 + 20260803 开盘价是否存在"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E

env = C.Env()
td = env.trade_dates
etf = C.load_idx("512100.SH")
print("ETF 列:", list(etf.columns))
print("ETF 数据范围:", etf.index.min(), "→", etf.index.max())
last5 = etf.tail(5)
print(last5[["open", "close"]])
print("td 最后 5 个交易日:", list(td[-5:]))
print("20260803 在 td 中:", "20260803" in td)
open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)
print("日频面板范围:", close_df.index.min(), "→", close_df.index.max())
print("日频面板是否含 20260803:", "20260803" in close_df.index)
print("512100 在 e_open_s 最后日期: ", end="")
eo = etf["open"].astype(float)
print(eo.tail(3).to_dict())
