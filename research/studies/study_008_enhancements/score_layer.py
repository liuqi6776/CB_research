# -*- coding: utf-8 -*-
"""打分层实验模块 (3.23): 可注入打分函数 — 行业中性化 / 分位数 A/B

engine.run_backtest 已支持 score_fn 注入 (score_fn(env, rb) -> scored Series,
见 engine.py 分数阈值分支); 本模块实现三种打分层:

  zscore   : 基线 — winsorize(1%/99%) -> 截面 zscore -> 等权列均值
             (与 engine.score_at 完全一致, 可做回归基准)
  ind_neut : 行业中性化 — 在 zscore 合成均值后, 按东财行业去均值
             (OpenAlpha cs_indneut / combo_backtest 的 neut="ind" 同口径:
              先因子合成, 再对合成分做行业内去均值, 消除行业暴露)
  quantile : 分位数打分 — winsorize -> 截面分位排名 [0,1] -> 等权列均值
             (秩变换抗极值, 对 zscore 假设分布不敏感)

usage:
    from research.studies.study_008_enhancements import score_layer as SL
    fn = SL.make_score_fn("ind_neut", ind_map=ind_map)
    scored = fn(env, rb)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import run_validation as rv
from research.factor_dic import style_factors as sf
from research.studies.study_008_enhancements import engine as E

MODES = ("zscore", "ind_neut", "quantile")


def _zdf(fdf):
    """winsorize + 截面 zscore (逐列)"""
    return fdf.apply(sf.winsorize_series).apply(
        lambda s: (s - s.mean()) / (s.std() + 1e-8), axis=0)


def score_at_mode(env, rb, ext_panels=None, mode="zscore", ind_map=None):
    """调仓日 rb 三种打分层之一, 返回 scored Series (高=好) 或 None (数据不足)"""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}, expect {MODES}")
    fdf = E.build_fdf(env, rb, ext_panels)
    if fdf is None:
        return None
    ext_cols = list(ext_panels.keys()) if ext_panels else []
    cols = [c for c in sf.BASE_COLS + ["VAL"] + ext_cols if c in fdf.columns]
    if mode == "quantile":
        q = fdf[cols].apply(sf.winsorize_series).rank(pct=True)
        has = q.dropna()
    else:
        z = _zdf(fdf)
        has = z[cols].dropna()
        if mode == "ind_neut":
            # 合成均值后按行业去均值 (OpenAlpha cs_indneut 简化版; 无映射归入 "NA" 组)
            ind = pd.Series({c: (ind_map or {}).get(c, "NA") for c in has.index}, index=has.index)
            comp = has.mean(axis=1)
            comp = comp - comp.groupby(ind).transform("mean")
            has = pd.DataFrame({"s": comp}).dropna()
            if len(has) < rv.TOP_N:
                return None
            return has["s"]
    if len(has) < rv.TOP_N:
        return None
    return has.mean(axis=1)


def make_score_fn(mode="zscore", ext_panels=None, ind_map=None):
    """工厂: 返回 score_fn(env, rb) -> scored Series, 供 engine.run_backtest(score_fn=...)"""
    def _fn(env, rb):
        return score_at_mode(env, rb, ext_panels=ext_panels, mode=mode, ind_map=ind_map)
    return _fn
