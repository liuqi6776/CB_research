# -*- coding: utf-8 -*-
"""诊断: amount 列量级与覆盖 — 为何全判 LOW_LIQ"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements.tradability import load_amount_df

env = C.Env()
td = env.trade_dates
amt = load_amount_df(env, td)
print("amount 宽表 shape:", amt.shape)
print("amount 全局统计:")
print(amt.stack().describe())
print("\n20260731 picks 中的 amount 覆盖:")
picks = env.picks_map.get("20260731", [])
print("picks 数:", len(picks))
in_amt = [c for c in picks if c in amt.columns]
print("在 amount 面板:", len(in_amt))
if in_amt:
    c = in_amt[0]
    win = td[max(0, td.index("20260731")-60):td.index("20260731")]
    print(f"\n示例 {c} 2026-05~07 amount:")
    print(amt[c].reindex(win).dropna().head(5).to_string())
    print("均值:", amt[c].reindex(win).dropna().mean())
