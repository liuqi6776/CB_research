# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.factor_dic import run_validation as rv
from research.factor_dic import lynch_factor as lf

trade_dates = rv.load_trade_dates()
months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
rebal = sorted(months.values())[:-1]
all_codes = set()
for rb in rebal:
    m = rv.load_index_weight(rb)
    if m:
        all_codes |= m
all_codes = sorted(all_codes)
print("all_codes n:", len(all_codes), "sample:", all_codes[:3])

pe_map = lf.load_pe_ttm(rebal, all_codes)
print("pe_map n:", len(pe_map))
if pe_map:
    k = list(pe_map)[0]
    print("  sample rb:", k, "n_codes:", len(pe_map[k]), "sample:", list(pe_map[k].items())[:2])

yoy_map = lf.build_yoy_pit(rebal, all_codes)
print("yoy_map n:", len(yoy_map))
for rb in ["20230428", "20231229", "20250127", "20260422"]:
    y = yoy_map.get(rb)
    print("  rb:", rb, "n:", len(y) if y is not None else None,
          "sample:", list(y.items())[:2] if (y is not None and len(y)) else "-")

peg = lf.build_peg(pe_map, yoy_map, rebal, all_codes)
print("peg n:", len(peg))
for rb in ["20230428", "20231229", "20250127", "20260422"]:
    p = peg.get(rb)
    print("  rb:", rb, "n:", len(p) if p else 0, "sample:", list(p.items())[:2] if p else "-")
