# -*- coding: utf-8 -*-
"""
因子冗余检测 + 正交性比对

Part A - 内部冗余: 对已保存的 ic_*.csv (21个因子) 做两两 IC 时序相关
Part B - 正交性:   计算现有研究因子(ivol/ret_1m/mom_20d)的月频 IC 序列,
                   与因子字典有效/反向因子做 IC 时序相关, 判断是否翻版

判定: |rho| >= 0.7 -> 同源冗余; 0.5~0.7 -> 高度相关; <0.5 -> 相对正交

用法:
    python research/factor_dic/redundancy_check.py
"""
import os
import sys
import glob
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research.factor_dic import run_validation as rv
from research.factor_dic.factor_lib import FACTOR_REGISTRY

OUT_DIR = rv.OUT_DIR
RHO_REDUNDANT = 0.7   # 同源阈值
RHO_HIGH = 0.5        # 高度相关阈值

# 现有研究因子(study_007 / chip_momentum 口径, 用复权 pct_chg 计算)
EXISTING_FACTORS = [
    ("ivol_std20", "低特质波动率(现有)", lambda d: -d["pct_chg"].fillna(0.0).rolling(20, min_periods=10).std() / 100.0),
    ("ret_1m",     "1月反转(现有)",      lambda d: -(1.0 + d["pct_chg"].fillna(0.0) / 100.0).rolling(20, min_periods=10).apply(np.prod, raw=True) + 1.0),
    ("mom_20d",    "20d动量(现有)",      lambda d: (1.0 + d["pct_chg"].fillna(0.0) / 100.0).rolling(20, min_periods=10).apply(np.prod, raw=True) - 1.0),
]


def load_ic_series():
    """读 results/ic_*.csv -> {fkey: Series(index=调仓日, value=IC)}"""
    out = {}
    for fp in glob.glob(os.path.join(OUT_DIR, "ic_*.csv")):
        fkey = os.path.basename(fp)[3:-4]
        try:
            s = pd.read_csv(fp, index_col=0).iloc[:, 0]
            s.index = s.index.astype(str)
            out[fkey] = s.astype(float)
        except Exception as e:
            print(f"[warn] 读取 {fp} 失败: {e}")
    return out


def part_a_internal(ics):
    """Part A: 21个因子 IC 序列两两相关"""
    names = {e[0]: e[1] for e in FACTOR_REGISTRY}
    df = pd.DataFrame(ics)
    corr = df.corr()
    print("\n" + "=" * 60)
    print("Part A - 内部冗余检测 (21因子 IC时序两两相关)")
    print("=" * 60)
    os.makedirs(OUT_DIR, exist_ok=True)
    corr.to_csv(os.path.join(OUT_DIR, "redundancy_matrix.csv"))
    pairs = []
    for i, f1 in enumerate(corr.index):
        for f2 in corr.columns[i + 1:]:
            r = corr.loc[f1, f2]
            if abs(r) >= RHO_HIGH:
                pairs.append((f1, f2, r))
    pairs.sort(key=lambda x: -abs(x[2]))
    print(f"{'因子1':<24}{'因子2':<24}{'|rho|':>8}  判定")
    print("-" * 60)
    for f1, f2, r in pairs:
        tag = "🔴同源" if abs(r) >= RHO_REDUNDANT else "🟠高相关"
        print(f"{names.get(f1,f1):<24}{names.get(f2,f2):<24}{r:>8.3f}  {tag}")
    if not pairs:
        print("  无 |rho|>=0.5 的因子对")
    return corr


def part_b_orthogonality(ics):
    """Part B: 现有因子 vs 因子字典因子的 IC 时序相关"""
    print("\n" + "=" * 60)
    print("Part B - 正交性比对 (现有 ivol/ret_1m/mom_20d vs 新因子)")
    print("=" * 60)

    trade_dates = rv.load_trade_dates()
    months = {d[:6]: d for d in trade_dates if d[:4] >= str(rv.START_YEAR)}
    rebal_dates = sorted(months.values())[:-1]

    all_codes = set()
    for rb in rebal_dates:
        members = rv.load_index_weight(rb)
        if members:
            all_codes |= members
    all_codes = sorted(all_codes)

    stocks, pct_df, _, _, _ = rv.load_panels(trade_dates, all_codes, None)

    # 逐股计算现有因子序列 + 未来收益
    t0 = time.time()
    ex_series = {name: {} for name, _, _ in EXISTING_FACTORS}
    fwd_by_code = {}
    for i, code in enumerate(all_codes):
        df = stocks.get(code)
        if df is None or len(df) < 60:
            continue
        pct = df["pct_chg"].fillna(0.0)
        cum = (1 + pct / 100.0).cumprod()
        fwd_by_code[code] = cum.shift(-rv.FORWARD_DAYS) / cum - 1.0
        for name, _, fn in EXISTING_FACTORS:
            try:
                s = fn(df)
                if s is not None and len(s.dropna()) > 0:
                    ex_series[name][code] = s.astype(float)
            except Exception:
                pass
        if (i + 1) % 500 == 0:
            print(f"  [calc] {i+1}/{len(all_codes)} ({time.time()-t0:.0f}s)")
    print(f"[calc] 现有因子+未来收益完成 ({time.time()-t0:.0f}s)")

    # 现有因子月度 Rank IC
    ex_ic = {}
    for name, label, _ in EXISTING_FACTORS:
        ic_list = []
        for rb in rebal_dates:
            members = rv.load_index_weight(rb)
            if members is None:
                continue
            fvals, rvals = {}, {}
            for code in members:
                fs = ex_series[name].get(code)
                fr = fwd_by_code.get(code)
                if fs is None or fr is None or rb not in fs.index or rb not in fr.index:
                    continue
                fv, rv_ = fs.loc[rb], fr.loc[rb]
                if pd.notna(fv) and pd.notna(rv_):
                    fvals[code] = fv
                    rvals[code] = rv_
            if len(fvals) < 50:
                continue
            f = rv.winsorize(pd.Series(fvals))
            ic_list.append((rb, f.rank().corr(pd.Series(rvals).rank())))
        ex_ic[name] = pd.Series([x[1] for x in ic_list], index=[x[0] for x in ic_list])
        print(f"  [IC] {label}: n={len(ex_ic[name])}  IC均值={ex_ic[name].mean():.4f}")

    # 与新因子对齐求相关
    focus = ["csad_std_21", "csad_ratio_20_120", "turnover_vol_20", "volume_surge_vol",
             "illiq_money_20", "net_support_vol", "chip_bandwidth"]
    names = {e[0]: e[1] for e in FACTOR_REGISTRY}
    rows = []
    print(f"\n{'现有因子':<16}{'新因子':<24}{'rho':>8}  判定")
    print("-" * 60)
    for ename, elabel, _ in EXISTING_FACTORS:
        for fkey in focus:
            if fkey not in ics or ics[fkey] is None:
                continue
            a, b = ex_ic[ename].align(ics[fkey], join="inner")
            if len(a) < 20:
                continue
            r = a.corr(b)
            tag = "🔴翻版" if abs(r) >= RHO_REDUNDANT else ("🟠高相关" if abs(r) >= RHO_HIGH else "🟢正交")
            rows.append((ename, fkey, r, tag))
            print(f"{elabel:<16}{names.get(fkey, fkey):<24}{r:>8.3f}  {tag}")
    if rows:
        out = pd.DataFrame(rows, columns=["existing", "new", "rho", "judge"])
        out.to_csv(os.path.join(OUT_DIR, "orthogonality_matrix.csv"), index=False, encoding="utf-8-sig")
    else:
        print("  无有效比对对")


def main():
    ics = load_ic_series()
    print(f"[load] 读入 {len(ics)} 个因子 IC 序列")
    part_a_internal(ics)
    part_b_orthogonality(ics)
    print("\n[done] 结果: redundancy_matrix.csv / orthogonality_matrix.csv")


if __name__ == "__main__":
    main()
