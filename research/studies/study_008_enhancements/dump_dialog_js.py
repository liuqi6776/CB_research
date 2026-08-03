# -*- coding: utf-8 -*-
"""生成 dialog_navs.js (紧凑 JS 数据, 供对话内收益曲线 widget 内联)"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "dialog_navs.json"), "r", encoding="utf-8") as f:
    d = json.load(f)

keys = [k for k in d["BASE+VAL"]]
parts = []
for label in ["BASE+VAL", "+MA20三档098", "+HRP", "+HRP+MA20三档098", "+HRP+MA20五档098", "512100ETF"]:
    s = d[label]
    vals = ",".join(f"{s[k]:.4f}" for k in keys)
    parts.append(f'  "{label}":[{vals}]')
out = "const DATES=" + json.dumps([k for k in keys], ensure_ascii=False) + ";\n"
out += "const NAVS={\n" + ",\n".join(parts) + "\n};\n"
out += "const ORDER=['512100ETF','BASE+VAL','+MA20三档098','+HRP','+HRP+MA20三档098','+HRP+MA20五档098'];\n"
with open(os.path.join(HERE, "data", "dialog_navs.js"), "w", encoding="utf-8") as f:
    f.write(out)
print("saved", len(out), "chars")
