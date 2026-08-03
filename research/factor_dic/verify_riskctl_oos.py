# -*- coding: utf-8 -*-
"""
MA20 三档回撤控制的独立样本(2018-2019) OOS 检验 (v-riskctl)

背景: risk_control.md 的 walk-forward 验证只能说明 deep 阈值(0.98)调参风险低,
不能排除"MA20 三档规则本身是在 2020-2026 样本期内被发现/确认"的样本内偏差。
review 建议用 2020 年之前的数据做独立 OOS。

限制: index_weight 成分股数据最早只有 2020-01-23, 无法做 2018-2019 的完整
Top50 组合回测。因此本检验退而检验 MA20 三档**择时规则本身**:
对 000852(中证1000) 指数日收益直接施加 w 仓位 (close>=MA20 -> 1.0;
MA20*deep<=close<MA20 -> 0.5; close<MA20*deep -> 0; 降仓部分现金0收益),
对比无风控持有指数。规则与组合构成无关, 独立于选股, 可干净地检验
"该趋势跟随规则在 2018-2019(熊市+反弹) 是否同样降低回撤/提升卡玛"。

分段:
  - 2018-2019: 独立样本 (OOS, 与主策略 2020-2026 无重叠)
  - 2020-2026: 主策略样本内区间
  - 2018-2026: 全样本
同时用 512100 ETF 日收益做一遍(实际可投资标的, 与指数收益几乎一致)。

口径: 日频, 无交易成本(规则有效性检验, 非精确回测——MA20 切换摩擦未计,
但三档切换频率低, 且 2018-2019 主要结论是"降回撤", 对该结论不敏感)。

输出: results/riskctl_oos_2018.txt
"""
import os
import numpy as np
import pandas as pd

sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(sys_path)

from research.factor_dic import run_validation as rv

OUT_DIR = rv.OUT_DIR
DEEPS = [0.97, 0.98]


def load_idx(code):
    df = pd.read_parquet(os.path.join(rv.IDX_DIR, f"{code}.parquet"))
    df["trade_date"] = df["trade_date"].astype(str).str[:8]
    return df.set_index("trade_date").sort_index()


def stats(pr, per_year=242.0):
    pr = pr.dropna()
    navs_c = (1 + pr).cumprod()
    years = len(pr) / per_year
    cagr = navs_c.iloc[-1] ** (1 / years) - 1
    sharpe = pr.mean() / pr.std(ddof=1) * np.sqrt(per_year)
    mdd = ((navs_c.cummax() - navs_c) / navs_c.cummax()).max()
    calmar = cagr / mdd if mdd > 0 else np.nan
    return cagr, sharpe, mdd, calmar


def main():
    sml = load_idx("000852.SH")
    etf = load_idx("512100.SH")
    idx_ret = sml["pct_chg"] / 100.0
    idx_close = sml["close"]
    ma20 = idx_close.rolling(20).mean()
    etf_ret = etf["pct_chg"] / 100.0
    idx_ret = idx_ret.reindex(etf_ret.index).dropna()  # 对齐可投资区间

    # 三档仓位序列(日频, 无前置平移——T 日收盘可知 MA20, T+1 生效为保守处理)
    # 保守化: 用 T-1 日信号在 T 日生效 (shift(1)), 避免同日收盘信号偷价
    w_map = {}
    for deep in DEEPS:
        w = pd.Series(1.0, index=idx_ret.index)
        c_prev, m_prev = idx_close.shift(1), ma20.shift(1)
        below = c_prev < m_prev
        deep_below = c_prev < deep * m_prev
        w[below] = 0.5
        w[deep_below] = 0.0
        w_map[deep] = w

    segs = [
        ("2018-2019 独立OOS", "20180101", "20191231"),
        ("2020-2026 样本内", "20200101", "20261231"),
        ("2018-2026 全样本",  "20180101", "20261231"),
    ]

    lines = []
    lines.append("=" * 96)
    lines.append("MA20 三档择时规则 独立样本 OOS 检验 (v-riskctl)")
    lines.append("标的: 000852 指数日收益 / 512100 ETF 日收益 (T-1 信号 T 日生效, 降仓部分现金0收益, 无交易成本)")
    lines.append("=" * 96)
    for (sname, s0, s1) in segs:
        m = (idx_ret.index >= s0) & (idx_ret.index <= s1)
        r_idx = idx_ret[m]
        r_etf = etf_ret.reindex(r_idx.index).fillna(0.0)
        lines.append(f"\n### {sname}  (n={len(r_idx)})")
        hdr = f"{'策略':<22}{'年化':>9}{'Sharpe':>8}{'MaxDD':>9}{'卡玛':>8}{'平均仓位':>9}"
        lines.append(hdr)
        for tag, r in (("指数 无风控", r_idx),):
            cagr, sh, mdd, cm = stats(r)
            lines.append(f"{tag:<22}{cagr:>9.2%}{sh:>8.2f}{mdd:>9.2%}{cm:>8.2f}{'100%':>9}")
        for deep in DEEPS:
            w = w_map[deep].reindex(r_idx.index).fillna(1.0)
            r_w = w * r_idx
            cagr, sh, mdd, cm = stats(r_w)
            lines.append(f"{'指数 +MA20三档'+f'({deep})':<22}{cagr:>9.2%}{sh:>8.2f}{mdd:>9.2%}{cm:>8.2f}{w.mean():>8.1%}")
        for tag, r in (("ETF  无风控", r_etf),):
            cagr, sh, mdd, cm = stats(r)
            lines.append(f"{tag:<22}{cagr:>9.2%}{sh:>8.2f}{mdd:>9.2%}{cm:>8.2f}{'100%':>9}")
        for deep in DEEPS:
            w = w_map[deep].reindex(r_etf.index).fillna(1.0)
            r_w = w * r_etf
            cagr, sh, mdd, cm = stats(r_w)
            lines.append(f"{'ETF +MA20三档'+f'({deep})':<22}{cagr:>9.2%}{sh:>8.2f}{mdd:>9.2%}{cm:>8.2f}{w.mean():>8.1%}")
        # 三档明细
        for deep in DEEPS:
            w = w_map[deep].reindex(r_idx.index).fillna(1.0)
            p1 = (w == 1.0).mean()
            p05 = (w == 0.5).mean()
            p0 = (w == 0.0).mean()
            lines.append(f"  仓位分布 deep={deep}: 满仓 {p1:.1%} / 半仓 {p05:.1%} / 空仓 {p0:.1%}")

    text = "\n".join(lines) + "\n"
    print(text)
    fp = os.path.join(OUT_DIR, "riskctl_oos_2018.txt")
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"[保存] {fp}")


if __name__ == "__main__":
    main()
