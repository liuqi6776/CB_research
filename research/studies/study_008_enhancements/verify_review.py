# -*- coding: utf-8 -*-
"""一次性核查脚本: 验证外部审查意见
1. HRP 权重 vs 纯逆波动权重 (相关性是否影响权重)
2. HRP 单股权重集中度 (max w / top5 / 行业集中)
3. 日频 MaxDD vs 月末 MaxDD (最优变体 +HRP+MA20五档098)
4. MA20 五档仓位切换次数 (隐含换手)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements.direction2_hrp import _hrp_weights, WINDOW
from research.studies.study_008_enhancements.dump_dialog_navs import _ma20_w, TIER5_W, TIER5_BND

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def main():
    env = C.Env()
    td = env.trade_dates
    # ---------- 1 & 2: HRP vs 纯逆波动 + 权重集中度 ----------
    rows = []
    for rb in env.picks_map:
        picks = env.picks_map[rb]
        hi = td.index(rb)
        win = td[max(0, hi - WINDOW):hi]
        rets = env.pct_df.reindex(columns=picks).reindex(win)
        w = _hrp_weights(rets)
        vol = rets.std()
        iv = (1.0 / (vol + 1e-9))
        iv = iv / iv.sum()
        mae = (w - iv.reindex(w.index)).abs().mean()
        corr = w.corr(iv.reindex(w.index))
        # 波动率下限测试: 若 vol 极低股票权重
        vmin = vol.min()
        wmax = w.max()
        top5 = w.nlargest(5).sum()
        rows.append(dict(rb=rb, mae=mae, corr=corr, wmax=wmax, top5=top5,
                         n_iv_gt2x=(w > 2 * iv.reindex(w.index)).sum()))
    r = pd.DataFrame(rows)
    say("=" * 70)
    say("[1] HRP 权重 vs 纯逆波动权重 (相关性的影响)")
    say(f"跨 {len(r)} 个调仓月: 平均 |w_hrp - w_iv| = {r['mae'].mean():.4f}  "
        f"(等权为 {2/50:.4f})")
    say(f"w_hrp 与 w_iv 相关系数: 均值 {r['corr'].mean():.3f}  中位 {r['corr'].median():.3f}")
    say("[2] 权重集中度")
    say(f"单股最大权重: 均值 {r['wmax'].mean():.2%}  最大 {r['wmax'].max():.2%}  "
        f"(>5% 的月份: {(r['wmax'] > 0.05).mean():.0%})")
    say(f"Top5 权重合计: 均值 {r['top5'].mean():.2%}  最大 {r['top5'].max():.2%}")

    # ---------- 3: 日频 vs 月末 MaxDD (最优变体) ----------
    say("")
    say("=" * 70)
    say("[3] 日频 vs 月末 MaxDD  (+HRP+MA20五档098)")
    nav_d = {}
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        nav = nav_d.get(rb, 1.0)
        if picks is None:
            nav_d[rb_next] = nav
            continue
        hi = td.index(rb)
        win = td[max(0, hi - WINDOW):hi]
        rets = env.pct_df.reindex(columns=picks).reindex(win)
        w = _hrp_weights(rets)
        cr = (comb * w.reindex(comb.columns)).sum(axis=1, min_count=1)
        for t in hold:
            r_t = e_ret.loc[t]
            if rs12_on:
                c = env.idx_close_1.get(t, np.nan)
                m = env.ma20_1.get(t, np.nan)
                ww = 1.0
                if np.isfinite(c) and np.isfinite(m):
                    ww = _ma20_w(c, m, "t5")
                r_t = ww * cr.loc[t]
            nav *= (1.0 + r_t)
            nav_d[t] = nav
    s = pd.Series(nav_d).sort_index()
    # 日频 MaxDD (逐交易日)
    dd_d = (s.cummax() - s) / s.cummax()
    mdd_daily = dd_d.max()
    dd_bottom = dd_d.idxmax()
    # 月末采样 MaxDD (取每月最后一个调仓/月末点)
    sm = s.groupby(s.index.str[:6]).last()
    mdd_monthly = ((sm.cummax() - sm) / sm.cummax()).max()
    say(f"日频样本 {len(s)} 个交易日; 日频 MaxDD {mdd_daily:.2%} (发生在 {dd_bottom})")
    say(f"月末采样 MaxDD: {mdd_monthly:.2%}  (差 {mdd_daily - mdd_monthly:+.2%})")
    # 日频 3 大回撤谷
    tops = dd_d.nlargest(3)
    for d_, v_ in tops.items():
        say(f"  回撤谷 {d_}: {v_:.2%}")

    # ---------- 4: 五档仓位切换次数 ----------
    say("")
    say("=" * 70)
    say("[4] MA20 五档仓位切换 (隐含换手)")
    # 统计每笔持仓期内仓位序列的切换次数 (0.25/0.5/0.75/1.0 之间变化)
    sw_total = 0
    sw_days = 0
    for rb, rb_next, hold, picks, comb, e_ret, rs12_on in env.month_segments():
        if picks is None:
            continue
        ws = []
        for t in hold:
            c = env.idx_close_1.get(t, np.nan)
            m = env.ma20_1.get(t, np.nan)
            ww = 1.0
            if rs12_on and np.isfinite(c) and np.isfinite(m):
                ww = _ma20_w(c, m, "t5")
            ws.append(ww)
        sw = sum(1 for a, b in zip(ws[:-1], ws[1:]) if a != b)
        sw_total += sw
        sw_days += len(ws)
    say(f"全区间 {sw_days} 个交易日中, 相邻日仓位发生切换 {sw_total} 次 "
        f"({sw_total / sw_days:.1%} 的交易日)")
    say(f"切换次数中涉及部分降仓(0.75/0.5/0.25)的比例无法从计数区分, 见 README")

    fp = os.path.join(C.OUT_DIR, "verify_review.txt")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT) + "\n")
    say(f"\n[saved] {fp}")


if __name__ == "__main__":
    main()
