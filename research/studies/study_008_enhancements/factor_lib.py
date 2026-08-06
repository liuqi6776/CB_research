# -*- coding: utf-8 -*-
"""增强量价因子库 (月度调仓日截面, 方向统一为"高值=好")

参照 qlib Alpha158 + GTJA191 风格, 从本地日频 OHLCV/amount 面板计算,
与现有 BASE(4因子: ret_1m/ivol/turnover_vol_20/VAL) 低相关的 8 个低频量价因子:

  ret_3m      : -60日累计收益        (长周期反转, 与 ret_1m 的 20 日窗去相关)
  vol_60      : -60日收益波动率      (长窗低波动)
  skew_20     : -20日收益偏度        (负偏度溢价)
  amt_ratio_20:  20日均额/60日均额   (量能趋势)
  vol_ratio   :  20日均量/60日均量   (量比)
  illiq_20    : -Amihud 非流动性均值  (高流动性好)
  max_ret_20  : -20日最大单日收益    (MAX 彩票效应)
  hl_range_20 : -20日平均振幅        (低波动延伸)

输出: {name: {rb: Series(code -> value)}}  (与 VAL 面板同结构, 供 engine.score_at ext_panels 使用)
"""
import numpy as np
import pandas as pd

FACTOR_DEFS = [
    ("ret_3m",      -1, lambda df: np.exp(np.log1p(df["pct_chg"] / 100.0).rolling(60, min_periods=30).sum()) - 1.0),
    ("vol_60",      -1, lambda df: df["pct_chg"].rolling(60, min_periods=30).std()),
    ("skew_20",     -1, lambda df: df["pct_chg"].rolling(20, min_periods=10).skew()),
    ("amt_ratio_20", 1, lambda df: df["amount"].rolling(20, min_periods=10).mean() / df["amount"].rolling(60, min_periods=30).mean()),
    ("vol_ratio",    1, lambda df: df["vol"].rolling(20, min_periods=10).mean() / df["vol"].rolling(60, min_periods=30).mean()),
    ("illiq_20",    -1, lambda df: (df["pct_chg"].abs() / df["amount"].replace(0, np.nan)).rolling(20, min_periods=10).mean()),
    ("max_ret_20",  -1, lambda df: df["pct_chg"].rolling(20, min_periods=10).max()),
    ("hl_range_20", -1, lambda df: ((df["high"] - df["low"]) / df["close"].replace(0, np.nan)).rolling(20, min_periods=10).mean()),
]


def build_ext_panels(stocks, rebal, all_codes):
    """计算 8 个增强因子面板 {name: {rb: Series(code -> value)}}

    stocks: code -> 日频 DataFrame(列含 pct_chg/open/high/low/close/vol/amount)
    rebal: 调仓日列表 (YYYYMMDD 字符串)
    """
    panels = {name: {} for name, _, _ in FACTOR_DEFS}
    dates = pd.Index(rebal)
    for code in all_codes:
        df = stocks.get(code)
        if df is None or len(df) < 80:
            continue
        for name, sign, fn in FACTOR_DEFS:
            try:
                s = fn(df)
            except Exception:
                continue
            s = s * sign  # 方向统一为"高值=好"
            if s is None or len(s.dropna()) < 30:
                continue
            s = s.reindex(dates)  # 调仓日无值(停牌等) → NaN (由打分 dropna 处理)
            panel = panels[name]
            for rb, v in s.items():
                if np.isfinite(v):
                    panel.setdefault(rb, {})[code] = float(v)
    out = {}
    for name, _, _ in FACTOR_DEFS:
        out[name] = {rb: pd.Series(vals) for rb, vals in panels[name].items()}
    return out


# =====================================================================
# V2 因子集 (3.22): 动量分层 / 波动率偏度 / 量价背离
# 与 V1 (3.21) 不同族, 刻意避开 ret_1m/ivol/turn/VAL 已捕获的信号:
#   动量分层: 多窗口动量 + 动量加速度 (与 20 日反转 ret_1m 窗口错开)
#   波动率偏度: 下行/上行半方差比 + 超额峰度 (与 ivol 的 std 测度不同)
#   量价背离: 收益-量变相关 / 量能斜率 / 收盘区间位置 / OBV 斜率 (纯量价交互)
# 方向统一为"高值=好" (sign 由 IC 方向检验自动修正, 此处为初值)。
# =====================================================================
FACTOR_DEFS_V2 = [
    # ---- 动量分层 (momentum stratification) ----
    ("mom_2m_ex_1m", 1, lambda df: _cumret(df, 40, 20)),      # 40-20 日动量 (跳过最近 20 日)
    ("mom_6m_ex_20d", 1, lambda df: _cumret(df, 120, 20)),    # 120-20 日动量 (GTJA191 中期)
    ("mom_acc_60",    1, lambda df: _cumret(df, 60, 20) - _cumret(df, 20, 0)),  # 60日动量-20日动量 (加速)
    # ---- 波动率偏度 (volatility skewness) ----
    ("vol_down_up_60", -1, lambda df: _down_up_ratio(df, 60)),  # -下行/上行半方差比
    ("kurt_60",       -1, lambda df: df["pct_chg"].rolling(60, min_periods=30).kurt()),  # -超额峰度
    # ---- 量价背离 (volume-price divergence) ----
    ("vp_corr_20",     1, lambda df: _vp_corr(df, 20)),       # 收益与量变相关
    ("amt_slope_60",   1, lambda df: _amt_slope(df, 60)),     # 成交额线性趋势斜率/均值
    ("hl_pos_60",      1, lambda df: _hl_pos(df, 60)),        # 收盘在高低区间的平均位置
    ("obv_slope_20",   1, lambda df: _obv_slope(df, 20)),     # OBV 斜率/收盘价
]


def _cumret(df, win, skip):
    """win 日累计收益, 跳过最近 skip 日 (动量分层, 与 ret_1m 的 20 日窗去重叠)"""
    pct = df["pct_chg"]
    r = (pct * 0.01).shift(skip).rolling(win - skip, min_periods=max(8, (win - skip) // 2)).sum()
    return r


def _down_up_ratio(df, win):
    """下行半方差 / 上行半方差 (波动率偏度: >1 = 下行风险更大)"""
    r = df["pct_chg"]
    lo = r.rolling(win, min_periods=max(8, win // 3)).apply(
        lambda x: np.sqrt((x[x < 0] ** 2).mean()) if (x < 0).any() else np.nan, raw=True)
    hi = r.rolling(win, min_periods=max(8, win // 3)).apply(
        lambda x: np.sqrt((x[x >= 0] ** 2).mean()) if (x >= 0).any() else np.nan, raw=True)
    return lo / hi.replace(0, np.nan)


def _vp_corr(df, win):
    """20 日 日收益 与 日成交量变化 的滚动相关系数 (量价背离)"""
    r = df["pct_chg"]
    dv = df["vol"].pct_change()
    return r.rolling(win, min_periods=max(8, win // 2)).corr(dv)


def _amt_slope(df, win):
    """成交额线性趋势斜率 / 均值 (量能趋势, 无量纲)"""
    a = df["amount"]
    y = a.rolling(win, min_periods=max(8, win // 2)).mean()
    x = np.arange(win)
    sl = a.rolling(win, min_periods=max(8, win // 2)).apply(
        lambda v: np.polyfit(x[:len(v)], v, 1)[0], raw=True)
    return sl / y.replace(0, np.nan)


def _hl_pos(df, win):
    """收盘价在 (low, high) 区间内的位置均值: (close-low)/(high-low)"""
    c, lo, hi = df["close"], df["low"], df["high"]
    pos = (c - lo) / (hi - lo).replace(0, np.nan)
    return pos.rolling(win, min_periods=max(8, win // 2)).mean()


def _obv_slope(df, win):
    """OBV (On-Balance Volume) 线性斜率 / 收盘价 (能量潮, 无量纲)"""
    pct = df["pct_chg"]
    vol = df["vol"]
    obv = (np.sign(pct) * vol).rolling(win, min_periods=1).sum()
    x = np.arange(win)
    sl = obv.rolling(win, min_periods=max(8, win // 2)).apply(
        lambda v: np.polyfit(x[:len(v)], v, 1)[0], raw=True)
    return sl / df["close"].replace(0, np.nan)


def build_ext_panels_v2(stocks, rebal, all_codes):
    """计算 9 个 V2 因子面板 {name: {rb: Series(code -> value)}} (结构同 V1)"""
    panels = {name: {} for name, _, _ in FACTOR_DEFS_V2}
    dates = pd.Index(rebal)
    for code in all_codes:
        df = stocks.get(code)
        if df is None or len(df) < 140:
            continue
        for name, sign, fn in FACTOR_DEFS_V2:
            try:
                s = fn(df)
            except Exception:
                continue
            s = s * sign
            if s is None or len(s.dropna()) < 30:
                continue
            s = s.reindex(dates)
            panel = panels[name]
            for rb, v in s.items():
                if np.isfinite(v):
                    panel.setdefault(rb, {})[code] = float(v)
    out = {}
    for name, _, _ in FACTOR_DEFS_V2:
        out[name] = {rb: pd.Series(vals) for rb, vals in panels[name].items()}
    return out
