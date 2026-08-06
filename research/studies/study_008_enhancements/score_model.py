# -*- coding: utf-8 -*-
"""模型层实验 (3.26): 线性等权 vs Ridge 线性学习 vs GBDT 非线性合成

与基线完全相同的 4 因子信息集 (ret_1m/ivol/turn/VAL, winsorize+截面 zscore),
只替换"合成函数", 直接回答"非线性合成能否超越线性等权":

  ew   : 等权均值 zscore (基线, engine.score_at 同口径)
  ridge: Ridge 回归 (学习型线性权重, 允许负权重)
  hgb  : HistGradientBoostingRegressor (sklearn 原生 GBDT, 非线性+因子交互)

walk-forward 无前视: 第 t 月模型只用 <t 月的样本训练 (扩展窗, warmup=18);
warmup 前打分返回 None -> fail-closed 现金, 基线同窗同结构, 保证对比公平。

标签: fwd 持有期收益, 训练窗口内 winsorize(1%,99%) 防极端值。
阈值对齐: 快速算各候选阈值平均持仓 -> 选最接近基线 54.5 档 -> 全量回测。

输出: results/score_model.txt|json + score_model_ic.png + score_model_nav.png + score_model_imp.png
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from research.studies.study_008_enhancements import common as C
from research.studies.study_008_enhancements import engine as E
from research.studies.study_008_enhancements.tradability import Tradability, load_amount_df
from research.factor_dic import run_validation as rv
from research.factor_dic import style_factors as sf

FACTORS = ["ret_1m", "ivol", "turn", "VAL"]
WARMUP = 18
THR_EW = 0.93
HGB_PARAMS = dict(max_iter=300, learning_rate=0.04, max_depth=3,
                  min_samples_leaf=100, l2_regularization=1.0, random_state=42)
RIDGE_ALPHA = 1.0


def _metrics(s):
    if s is None or len(s) < 2:
        return None
    n = len(s)
    cagr = (s.iloc[-1] / s.iloc[0]) ** (242.0 / (n - 1)) - 1.0
    ret = s.pct_change().dropna()
    shp = ret.mean() / (ret.std() + 1e-12) * np.sqrt(242.0)
    dd = ((s.cummax() - s) / s.cummax()).max()
    m_ret = s.groupby(s.index.str[:6]).last().pct_change().dropna()
    return dict(final=float(s.iloc[-1]), ann=cagr, sharpe=shp, mdd=float(dd),
                calmar=float(cagr / dd) if dd > 0 else 0.0,
                m_mean=float(m_ret.mean()), m_win=float((m_ret > 0).mean()))


def _zdf(fdf):
    return fdf.apply(sf.winsorize_series).apply(
        lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)


def build_samples(env, rebal):
    """长表样本: (rb, code, 4 因子 zscore, label=fwd 收益); 与基线同成员/同列筛选"""
    rows = []
    for rb in rebal:
        fdf = E.build_fdf(env, rb, None)
        if fdf is None:
            continue
        z = _zdf(fdf)
        cols = [c for c in FACTORS if c in z.columns]
        has = z[cols].dropna()
        for code, r in has.iterrows():
            rows.append({"rb": rb, "code": code, **r.to_dict(),
                         "label": float(env.fwd[code].loc[rb])})
    df = pd.DataFrame(rows)
    return df.dropna(subset=["label"])


def walk_forward(samples, rebal, model_name):
    """扩展窗 walk-forward: 返回 {rb: predicted Series(code->score)} + 末次模型"""
    idx = {rb: i for i, rb in enumerate(rebal)}
    samples = samples.assign(_i=samples["rb"].map(idx))
    feats = [c for c in FACTORS if c in samples.columns]
    # VAL 缺失月 (整月无 VAL 列) 的该列特征补 0 (中性 zscore); 其余月份 dropna 已保证无缺
    samples = samples.copy()
    for f in feats:
        samples[f] = samples[f].fillna(0.0)
    preds, last_model = {}, None
    for i, rb in enumerate(rebal):
        if i < WARMUP:
            continue
        train = samples[samples["_i"] < i]
        test = samples[samples["_i"] == i]
        if len(train) < 500 or len(test) < 30:
            continue
        lo, hi = np.percentile(train["label"], [1, 99])
        y = np.clip(train["label"].to_numpy(), lo, hi)
        X, Xt = train[feats], test[feats]
        if model_name == "ridge":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=RIDGE_ALPHA, random_state=42).fit(X, y)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(**HGB_PARAMS).fit(X, y)
        last_model = model
        s = pd.Series(model.predict(Xt).astype(float), index=test["code"].values)
        preds[rb] = s
    return preds, last_model, feats


def main():
    env = C.Env()
    td = env.trade_dates
    open_df, close_df, high_df, low_df, pct_df, e_ovn, e_intra = E.load_prices(env, td)
    rebal = [rb for rb, *_ in env.month_segments()]
    st_map = E.load_st_intervals()
    one_up, one_dn = E.build_limit_sets(open_df, high_df, low_df, pct_df, env.all_codes)
    amount_df = load_amount_df(env, td)
    tf5 = Tradability(td, amount_df, lookback=60, min_amount=3e6, min_px_days=20,
                      st_map=st_map, min_vol=12.0, pct_df=pct_df)
    rb2idx = {rb: i for i, rb in enumerate(rebal)}

    print("[m] 构建样本集 (4 因子 zscore + fwd 标签) ...", flush=True)
    samples = build_samples(env, rebal)
    print(f"[m] 样本 {len(samples)} 行 / {samples['rb'].nunique()} 月; "
          f"标签均值 {samples['label'].mean()*100:.2f}%")

    # ---- walk-forward ----
    preds = {}
    last_model, imp = {}, {}
    for mn in ("ridge", "hgb"):
        print(f"[m] walk-forward 训练 {mn} ...", flush=True)
        p, lm, feats = walk_forward(samples, rebal, mn)
        preds[mn] = p
        last_model[mn] = lm
        if lm is not None and hasattr(lm, "feature_importances_"):
            imp[mn] = dict(zip(feats, lm.feature_importances_))

    # ---- IC / Top60 (同窗: warmup 后) ----
    def _fwd_at(env, rb):
        out = {}
        for code, fr in env.fwd.items():
            if rb in fr.index:
                v = fr.loc[rb]
                if np.isfinite(v):
                    out[code] = float(v)
        return pd.Series(out)

    ics = {"ew": [], "ridge": [], "hgb": []}
    pts = {"ew": [], "ridge": [], "hgb": []}
    for i, rb in enumerate(rebal):
        if i < WARMUP:
            continue
        fr = _fwd_at(env, rb)
        sc0 = E.score_at(env, rb, None)
        scoreds = {"ew": sc0}
        for mn in ("ridge", "hgb"):
            scoreds[mn] = preds[mn].get(rb)
        for mn, scored in scoreds.items():
            if scored is None:
                continue
            df = pd.DataFrame({"s": scored, "f": fr}).dropna()
            if len(df) > 30:
                rho, _ = spearmanr(df["s"], df["f"])
                ics[mn].append(rho)
            if len(scored) >= 60:
                top = scored.sort_values(ascending=False).head(60).index
                df2 = pd.DataFrame({"s": scored.reindex(top), "f": fr.reindex(top)}).dropna()
                for code, row in df2.iterrows():
                    pts[mn].append((rb, code, row["s"], row["f"]))

    lines = ["模型层实验 (3.26): 线性等权 vs Ridge vs GBDT 非线性合成",
             "=" * 92]
    lines.append(f"样本 {len(samples)} 行 ({samples['rb'].nunique()} 月 × 成分股), "
                 f"因子 {FACTORS}; warmup {WARMUP} 月, 同窗对比 {len(ics['ew'])} 月")
    lines.append("")
    lines.append("[3] 打分 IC / Top60 散点 (同窗 warmup 后):")
    d = {}
    for mn in ("ew", "ridge", "hgb"):
        a = np.asarray(ics[mn])
        df = pd.DataFrame(pts[mn], columns=["rb", "code", "s", "f"])
        d[mn] = df
        rho = spearmanr(df["s"], df["f"])[0] if len(df) else np.nan
        lines.append(f"    {mn:<6} IC {a.mean():+.4f} | ICIR {a.mean()/(a.std()+1e-12):+.3f} | "
                     f"正占比 {(a>0).mean():.0%} | Top60 Spearman {rho:+.4f} (n={len(df)})")
    lines.append(f"    与等权差值: IC {np.mean(ics['ridge'])-np.mean(ics['ew']):+.4f} (ridge) | "
                 f"{np.mean(ics['hgb'])-np.mean(ics['ew']):+.4f} (hgb); "
                 f"Top60 {spearmanr(d['ridge']['s'], d['ridge']['f'])[0]-spearmanr(d['ew']['s'], d['ew']['f'])[0]:+.4f} (ridge) | "
                 f"{spearmanr(d['hgb']['s'], d['hgb']['f'])[0]-spearmanr(d['ew']['s'], d['ew']['f'])[0]:+.4f} (hgb)")
    if imp:
        lines.append("")
        lines.append("    HGB 特征重要性 (末次模型): " +
                     " ".join(f"{k}={v:.3f}" for k, v in imp["hgb"].items()))

    # ---- 阈值对齐: 快速持仓估算 ----
    lines.append("")
    lines.append("[4] 阈值选股回测 (同窗, 阶段4, 万1; 阈值对齐 ~54.5 持仓):")
    ew_scores = {rb: E.score_at(env, rb, None) for i, rb in enumerate(rebal) if i >= WARMUP}

    def _holdings(series_map, thr):
        hs = []
        for rb, s in series_map.items():
            if s is None or len(s) == 0:
                continue
            hs.append(float((s >= thr).sum()))
        return float(np.mean(hs)) if hs else 0.0

    bt = {}
    # 基线: 直接用 0.93 (全窗持仓 54.5), 同时打印同窗实际持仓
    ns_ew = _holdings(ew_scores, THR_EW)
    lines.append(f"    基线 ew >= {THR_EW}: 同窗平均持仓 {ns_ew:.1f}")

    def _align(model_scores, target):
        pool = pd.concat([s for s in model_scores.values() if s is not None])
        cands = [float(v) for v in np.percentile(pool, np.arange(80, 99, 1))]
        best, best_d = None, 1e9
        for thr in cands:
            ns = _holdings(model_scores, thr)
            if abs(ns - target) < best_d:
                best, best_d = (thr, ns), abs(ns - target)
        return best[0], best[1]

    fns = {"ew": (lambda env, rb, i=0: E.score_at(env, rb, None) if rb2idx[rb] >= WARMUP else None)}
    for mn in ("ridge", "hgb"):
        thr, ns = _align(preds[mn], ns_ew)
        bt[mn] = dict(thr=thr, ns=ns)
        fns[mn] = (lambda env, rb, mn=mn: preds[mn].get(rb) if rb2idx[rb] >= WARMUP else None)
        lines.append(f"    模型 {mn}: 对齐阈值 {thr:.4f} -> 同窗平均持仓 {ns:.1f}")

    # ---- 全量回测 (各方案对齐档 + 基线) ----
    navs = {}
    for mn in ("ew", "ridge", "hgb"):
        nav, stt = E.run_backtest(env, td, open_df, close_df, high_df, low_df, pct_df,
                                  e_ovn, e_intra, use_hrp=False, use_ma20=False,
                                  st_map=st_map, limit_sets=(one_up, one_dn),
                                  tradable=tf5, score_thr=THR_EW if mn == "ew" else bt[mn]["thr"],
                                  score_fn=fns[mn])
        m = _metrics(nav)
        bt[mn] = dict(thr=THR_EW if mn == "ew" else bt[mn]["thr"],
                      ns=ns_ew if mn == "ew" else bt[mn]["ns"], m=m, nav=nav)
        lines.append(f"    {mn:<6} >= {bt[mn]['thr']:<6} (持仓 {bt[mn]['ns']:.1f}): "
                     f"终值 {m['final']:.4f} | 年化 {m['ann']:.2%} | Sharpe {m['sharpe']:.2f} | "
                     f"卡玛 {m['calmar']:.2f} | MaxDD {m['mdd']:.2%}")
    lines.append(f"    vs 基线: 年化 {bt['ridge']['m']['ann']-bt['ew']['m']['ann']:+.2%} (ridge) | "
                 f"{bt['hgb']['m']['ann']-bt['ew']['m']['ann']:+.2%} (hgb); "
                 f"卡玛 {bt['ridge']['m']['calmar']-bt['ew']['m']['calmar']:+.2f} (ridge) | "
                 f"{bt['hgb']['m']['calmar']-bt['ew']['m']['calmar']:+.2f} (hgb)")
    print("\n".join(lines))

    # ---- 图1: IC ----
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = {"ew": "#333", "ridge": "#1f77b4", "hgb": "#d62728"}
    for mn in ("ew", "ridge", "hgb"):
        rol = pd.Series(ics[mn]).rolling(12, min_periods=6).mean()
        ax.plot(rol.index, rol.values, lw=1.6, color=colors[mn], label=f"{mn} 12月滚动 IC")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_title("月度打分 IC (同窗): 等权 vs Ridge vs GBDT")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "score_model_ic.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    # ---- 图2: 净值 ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for mn in ("ew", "ridge", "hgb"):
        ax.plot(bt[mn]["nav"].index, bt[mn]["nav"].values, lw=1.2, color=colors[mn],
                label=f"{mn} (>= {bt[mn]['thr']}, 卡玛 {bt[mn]['m']['calmar']:.2f})")
    ax.set_title("阈值选股净值对比 (同窗 warmup 后, 持仓对齐)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = os.path.join(C.OUT_DIR, "score_model_nav.png")
    fig.savefig(fp, dpi=130)
    plt.close(fig)
    print(f"[saved] {fp}")

    # ---- 图3: 特征重要性 ----
    if "hgb" in imp:
        fig, ax = plt.subplots(figsize=(7, 4))
        fi = pd.Series(imp["hgb"]).sort_values()
        ax.barh(fi.index, fi.values, color="#d62728", alpha=0.8)
        ax.set_title("HGB 特征重要性 (末次模型)")
        fig.tight_layout()
        fp = os.path.join(C.OUT_DIR, "score_model_imp.png")
        fig.savefig(fp, dpi=130)
        plt.close(fig)
        print(f"[saved] {fp}")

    out = dict(warmup=WARMUP, n_samples=len(samples),
               ic={mn: dict(n=len(ics[mn]), mean=float(np.mean(ics[mn])),
                            icir=float(np.mean(ics[mn]) / (np.std(ics[mn]) + 1e-12)),
                            pos=float((np.asarray(ics[mn]) > 0).mean()),
                            top60_sp=float(spearmanr(d[mn]["s"], d[mn]["f"])[0]))
                  for mn in ("ew", "ridge", "hgb")},
               imp=imp,
               bt={mn: dict(thr=bt[mn]["thr"], m=bt[mn]["m"], n_sel=bt[mn]["ns"]) for mn in ("ew", "ridge", "hgb")})
    with open(os.path.join(C.OUT_DIR, "score_model.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(C.OUT_DIR, "score_model.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[saved] {os.path.join(C.OUT_DIR, 'score_model.txt')} | .json")


if __name__ == "__main__":
    main()
