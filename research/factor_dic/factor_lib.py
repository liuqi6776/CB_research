# -*- coding: utf-8 -*-
"""
因子字典验证 - 因子计算库（日频近似）

来源因子字典: quant_conclusion/factor_dic/Workbook1.csv (600+ 因子, Fitness 排序)
本库实现其中可用**日频数据**验证/近似的因子族:
  1. 流动性/非流动性族   (Amihud 类: money/轨迹非流动性/郑兆磊非流动性)
  2. 换手率族            (特质换手波动率/邪修换手率/换手率极大值幅度)
  3. 波动率族            (耀眼波动率/适度耀眼波动率/量涌波动率 的日频近似)
  4. 羊群/CSAD 族        (std_21 CSAD / ratio_20_120 CSAD)
  5. 筹码分布族          (获利比率/筹码乖离率/当天新增筹码盈利占比/成本带宽)
  6. 资金流族            (净支撑成交量/大单净流入/待著而救 的日频近似)
  7. 反转/动量族         (高频反转 rev_hf 日频近似/长端动量/周五动量)

每个因子函数输入: 单只股票的日频 DataFrame(按 trade_date 升序, 含 open/high/low/close/vol/amount/pct_chg)
输出: 以 trade_date 为索引的因子 Series(可含 NaN, 末尾需 NaN 供未来收益对齐)
"""
import numpy as np
import pandas as pd


def _safe_div(a, b, fill=np.nan):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out = out.replace([np.inf, -np.inf], fill)
    return out


# ==================== 1. 流动性/非流动性族 ====================

def illiq_money_20(d: pd.DataFrame, window: int = 20) -> pd.Series:
    """money 高频非流动性 日频近似: 20日均值(|ret|/amount) * 1e9
    字典 Fitness=2.06; 方向: 负向(高非流动 → 未来收益低)"""
    ret = d["pct_chg"].abs() / 100.0
    amt = d["amount"].replace(0, np.nan)
    amihud = _safe_div(ret, amt) * 1e9
    return amihud.rolling(window, min_periods=10).mean()


def illiq_traj_20(d: pd.DataFrame, window: int = 20) -> pd.Series:
    """轨迹非流动性 日频近似: 20日均值(|ret|/sqrt(amount)) * 1e6
    字典 Fitness=1.84; 方向: 负向"""
    ret = d["pct_chg"].abs() / 100.0
    amt = np.sqrt(d["amount"].replace(0, np.nan))
    amihud = _safe_div(ret, amt) * 1e6
    return amihud.rolling(window, min_periods=10).mean()


def illiq_vol_20(d: pd.DataFrame, window: int = 20) -> pd.Series:
    """郑兆磊非流动性(vol 口径) 日频近似: 20日均值(|ret|/vol)
    字典 Fitness=1.54; 方向: 负向"""
    ret = d["pct_chg"].abs() / 100.0
    vol = d["vol"].replace(0, np.nan)
    return _safe_div(ret, vol).rolling(window, min_periods=10).mean()


# ==================== 2. 换手率族 ====================

def turnover_vol_20(d: pd.DataFrame, window: int = 20) -> pd.Series:
    """特质换手波动率 日频近似: 换手率(vol/流通股本近似用 amount/close 代理量能) 的滚动波动
    字典 Fitness=2.02; 方向: 负向(换手波动大 → 散户博弈 → 未来收益低)"""
    # 无流通股本时用量能代理: 换手代理 = vol * close (金额量)
    proxy = d["vol"] * d["close"]
    t = proxy.replace(0, np.nan)
    # 对数化降低量级, 再取滚动波动
    lt = np.log(t + 1)
    return lt.rolling(window, min_periods=10).std()


def turnover_extreme(d: pd.DataFrame, window: int = 20) -> pd.Series:
    """换手率极大值幅度 日频近似: 当日量能 / 20日均量能 - 1
    字典 Fitness=1.52; 方向: 负向(放量异动后回落)"""
    proxy = d["vol"] * d["close"]
    ma = proxy.rolling(window, min_periods=10).mean().replace(0, np.nan)
    return _safe_div(proxy, ma) - 1.0


def turnover_evil(d: pd.DataFrame, window: int = 20) -> pd.Series:
    """邪修换手率 日频近似: 换手率与价格变化相关性残差的负值(量价背离度)
    字典 Fitness=1.54; 方向: 负向(量价背离 → 不健康)"""
    proxy = np.log(d["vol"] * d["close"] + 1)
    ret = d["pct_chg"] / 100.0
    corr = proxy.rolling(window, min_periods=10).corr(ret)
    return -corr  # 相关性高=健康; 取负 → 高值=背离


# ==================== 3. 波动率族 ====================

def loud_vol(d: pd.DataFrame, window: int = 20, amp_k: float = 1.5) -> pd.Series:
    """耀眼波动率 日频近似: 放量日(amount>20日均额*k)的收益波动
    字典 Fitness=1.95; 方向: 负向"""
    amt = d["amount"]
    ma = amt.rolling(window, min_periods=10).mean()
    mask = amt > ma * amp_k
    ret = d["pct_chg"] / 100.0
    vol = ret.where(mask).rolling(window, min_periods=5).std()
    return vol


def moderate_loud_vol(d: pd.DataFrame, window: int = 20) -> pd.Series:
    """适度耀眼波动率 日频近似: 放量日的波动减去全样本波动(度量"额外波动")
    字典 Fitness=2.16; 方向: 负向"""
    amt = d["amount"]
    ma = amt.rolling(window, min_periods=10).mean()
    mask = amt > ma * 1.5
    ret = d["pct_chg"] / 100.0
    vol_all = ret.rolling(window, min_periods=10).std()
    vol_loud = ret.where(mask).rolling(window, min_periods=5).std()
    return (vol_loud - vol_all).fillna(0.0)


def volume_surge_vol(d: pd.DataFrame, window: int = 20) -> pd.Series:
    """量涌波动率 日频近似: 量价同涨日(价涨且量增)的波动
    字典 Fitness=1.92; 方向: 负向"""
    ret = d["pct_chg"] / 100.0
    vol_grow = d["vol"] > d["vol"].shift(1)
    price_grow = d["close"] > d["open"]
    mask = vol_grow & price_grow
    return ret.where(mask).rolling(window, min_periods=5).std()


# ==================== 4. 羊群/CSAD 族 ====================

def csad_std_21(d: pd.DataFrame, mkt_ret: pd.Series, window: int = 21) -> pd.Series:
    """std_21 CSAD 羊群模型: 21日滚动(|个股收益-市场收益|的均值)
    字典 Fitness=2.21; 方向: 负向(分歧大 → 波动大 → 未来收益低)
    注意: mkt_ret 需为同索引的市场收益(如中证1000日收益)"""
    ret = d["pct_chg"] / 100.0
    dev = (ret - mkt_ret).abs()
    return dev.rolling(window, min_periods=10).mean()


def csad_ratio_20_120(d: pd.DataFrame, mkt_ret: pd.Series) -> pd.Series:
    """ratio_20/120 CSAD: 20日偏离 / 120日偏离
    字典 Fitness=1.60; 方向: 负向(短期分歧扩大 → 风险)"""
    ret = d["pct_chg"] / 100.0
    dev = (ret - mkt_ret).abs()
    s20 = dev.rolling(20, min_periods=10).mean()
    s120 = dev.rolling(120, min_periods=60).mean()
    return _safe_div(s20, s120)


# ==================== 5. 筹码分布族 ====================

def chip_winner_rate(chip: pd.DataFrame) -> pd.Series:
    """获利比率(winner_rate): 直接取筹码数据列
    字典 Fitness=0.41; 方向: 负向(获利盘>70% → 抛压)"""
    return chip["winner_rate"]


def chip_bias(chip: pd.DataFrame) -> pd.Series:
    """筹码乖离率: close / weight_avg - 1
    字典 Fitness=1.58; 方向: 负向(乖离大 → 回归)"""
    wa = chip["weight_avg"].replace(0, np.nan)
    return _safe_div(chip["close"], wa) - 1.0


def chip_new_profit(chip: pd.DataFrame) -> pd.Series:
    """当天新增筹码盈利占比: 用 (close - cost_15) / (cost_85 - cost_15) 近似
    字典 Fitness=1.62; 方向: 正向(当日获利筹码占比高 → 后续延续)"""
    spread = chip["cost_85pct"] - chip["cost_15pct"]
    spread = spread.replace(0, np.nan)
    return _safe_div(chip["close"] - chip["cost_15pct"], spread)


def chip_bandwidth(chip: pd.DataFrame) -> pd.Series:
    """成本带宽: (cost_85 - cost_15) / cost_50
    字典 Fitness=0.29; 方向: 负向(筹码分散 → 上行压力)"""
    c50 = chip["cost_50pct"].replace(0, np.nan)
    return _safe_div(chip["cost_85pct"] - chip["cost_15pct"], c50)


# ==================== 6. 资金流族 ====================

def net_mf_5d(mf: pd.DataFrame, window: int = 5) -> pd.Series:
    """净资金流 5 日累计(资金流数据 net_mf_amount)
    方向: 正向(持续流入 → 支撑)"""
    return mf["net_mf_amount"].rolling(window, min_periods=3).sum()


def lg_net_5d(mf: pd.DataFrame, window: int = 5) -> pd.Series:
    """大单净流入 5 日累计(buy_lg - sell_lg + buy_elg - sell_elg)
    方向: 正向"""
    cols = ["buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"]
    if not all(c in mf.columns for c in cols):
        if "lg_net_amount" in mf.columns:
            return mf["lg_net_amount"].rolling(window, min_periods=3).sum()
        return pd.Series(np.nan, index=mf.index)
    net = (mf["buy_lg_amount"] - mf["sell_lg_amount"] + mf["buy_elg_amount"] - mf["sell_elg_amount"])
    return net.rolling(window, min_periods=3).sum()


def net_support_vol(d: pd.DataFrame, window: int = 10) -> pd.Series:
    """净支撑成交量 日频近似: (收阳量 - 收阴量) 的滚动累计 / 总成交量
    字典 Fitness=1.90; 方向: 正向(支撑位买盘强)"""
    up = d["close"] >= d["open"]
    vol = d["vol"]
    net = (vol * up.astype(float)) - (vol * (~up).astype(float))
    tot = vol.rolling(window, min_periods=5).sum().replace(0, np.nan)
    return net.rolling(window, min_periods=5).sum() / tot


# ==================== 7. 反转/动量族 ====================

def reversal_5(d: pd.DataFrame, window: int = 5) -> pd.Series:
    """高频反转 日频近似: -5日累计收益
    字典 rev_hf Fitness=0.55; 方向: 正向(反转, 高值=近期跌 → 未来反弹)"""
    return -d["pct_chg"].rolling(window).sum()


def momentum_long(d: pd.DataFrame, long_w: int = 250, skip: int = 20) -> pd.Series:
    """长端动量: 250日收益 - 近20日收益(剔除短期反转污染)
    字典 Fitness=0.30; 方向: 正向"""
    r = d["pct_chg"].rolling(long_w).sum() - d["pct_chg"].rolling(skip).sum()
    return r


def friday_momentum(d: pd.DataFrame) -> pd.Series:
    """周五动量: 仅周五收益的累计(一周动量变体)
    字典 Fitness=0.05; 方向: 正向"""
    td = d["trade_date"].astype(str)
    idx = pd.to_datetime(td, format="%Y%m%d", errors="coerce")
    if idx.isna().all():
        idx = pd.to_datetime(td, errors="coerce")  # 兼容 "2020-01-02" 等格式
    fri = idx.dt.weekday == 4
    r = d["pct_chg"].where(fri, 0.0)
    return r.rolling(20).sum()


# ==================== 注册表(按重要性排序) ====================
# 每项: (key, 名称, 方向, 计算函数, 所需额外数据)
FACTOR_REGISTRY = [
    # P0: 流动性族 (最契合现有研究)
    ("illiq_money_20",    "非流动性_money_20d",   "neg", illiq_money_20,  None),
    ("illiq_traj_20",     "轨迹非流动性_20d",     "neg", illiq_traj_20,   None),
    ("illiq_vol_20",      "非流动性_vol_20d",     "neg", illiq_vol_20,    None),
    # P0: 筹码分布族
    ("chip_bias",         "筹码乖离率",           "neg", chip_bias,       "chip"),
    ("chip_new_profit",   "当天新增筹码盈利占比", "pos", chip_new_profit, "chip"),
    ("chip_winner_rate",  "获利比率",             "neg", chip_winner_rate,"chip"),
    ("chip_bandwidth",    "成本带宽",             "neg", chip_bandwidth,  "chip"),
    # P1: 换手率族
    ("turnover_vol_20",   "特质换手波动率",       "neg", turnover_vol_20, None),
    ("turnover_extreme",  "换手率极大值幅度",     "neg", turnover_extreme,None),
    ("turnover_evil",     "邪修换手率",           "neg", turnover_evil,   None),
    # P1: 波动率族
    ("moderate_loud_vol", "适度耀眼波动率",       "neg", moderate_loud_vol, None),
    ("loud_vol",          "耀眼波动率",           "neg", loud_vol,        None),
    ("volume_surge_vol",  "量涌波动率",           "neg", volume_surge_vol,None),
    # P1: 羊群族 (需要市场收益)
    ("csad_std_21",       "CSAD_std_21",          "neg", csad_std_21,     "mkt"),
    ("csad_ratio_20_120", "CSAD_ratio_20_120",    "neg", csad_ratio_20_120, "mkt"),
    # P2: 资金流族
    ("net_mf_5d",         "净资金流5d",           "pos", net_mf_5d,       "mf"),
    ("lg_net_5d",         "大单净流入5d",         "pos", lg_net_5d,       "mf"),
    ("net_support_vol",   "净支撑成交量",         "pos", net_support_vol, None),
    # P2: 反转/动量族
    ("reversal_5",        "反转5d",               "pos", reversal_5,      None),
    ("momentum_long",     "长端动量250_20",       "pos", momentum_long,   None),
    ("friday_momentum",   "周五动量",             "pos", friday_momentum, None),
]
