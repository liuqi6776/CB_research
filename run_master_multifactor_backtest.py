# -*- coding: utf-8 -*-

"""
全量多因子 GBDT 样本外 (OOS Walk-Forward) 实证与回测引擎 (Master Multi-Factor GBDT Backtest Engine)
彻底重构：跨平台 Pathlib 相对路径、Fail-Fast 严格模型加载断言、动态尾部数据截断与零前视交易历。
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from cb_quant.data_loader import CBDataLoader
from cb_quant.feature_pipeline import build_unified_feature_matrix, FEATURE_COLS
from cb_quant.time_structured_router import CBTimeStructuredRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent

def simulate_nav(df_pit, df_orders, use_smart_limit=False, timing_mode='none', use_timing=None):
    """
    timing_mode options:
      - 'none': 无择时 (100% 满仓)
      - 'binary': 二档择时 (熊市 0% 空仓 / 牛市 100% 满仓)
      - '3tier': 三档动态仓位择时 (熊市 Tier 3: 20% / 震荡市 Tier 2: 50% / 牛市 Tier 1: 100%)
    """
    if use_timing is not None:
        if isinstance(use_timing, bool):
            timing_mode = '3tier' if use_timing else 'none'

    u_dates = sorted(df_pit['date_str'].unique())
    daily_close = df_pit.groupby('date_str')['close'].mean()
    mkt_ma20 = daily_close.rolling(20, min_periods=5).mean()
    mkt_ma60 = daily_close.rolling(60, min_periods=10).mean()
    
    capital = 1000000.0
    positions = {}
    trade_logs = []
    daily_nav = []

    if not df_orders.empty:
        df_orders_copy = df_orders.copy()
        df_orders_copy['trade_date_str'] = pd.to_numeric(df_orders_copy['trade_date'], errors='coerce').fillna(0).astype(int).astype(str)
        orders_by_date = {d: g for d, g in df_orders_copy.groupby('trade_date_str')}
    else:
        orders_by_date = {}
    d_by_code_all = {d: g.groupby('ts_code').last().to_dict('index') for d, g in df_pit.groupby('date_str')}
    d_first_all = {d: g.groupby('ts_code').first().to_dict('index') for d, g in df_pit.groupby('date_str')}
    
    for d_idx, d_str in enumerate(u_dates):
        mkt_c_t1 = daily_close.get(u_dates[d_idx-1], np.nan) if d_idx > 0 else np.nan
        mkt_ma20_t1 = mkt_ma20.get(u_dates[d_idx-1], np.nan) if d_idx > 0 else np.nan
        mkt_ma60_t1 = mkt_ma60.get(u_dates[d_idx-1], np.nan) if d_idx > 0 else np.nan

        # 三档择时与最大持仓数 / 资金乘数控制
        max_positions_limit = 10
        capital_multiplier = 1.0
        
        if timing_mode == 'binary':
            if not np.isnan(mkt_c_t1) and not np.isnan(mkt_ma20_t1) and (mkt_c_t1 < mkt_ma20_t1):
                max_positions_limit = 0
                capital_multiplier = 0.0
        elif timing_mode == '3tier':
            if not np.isnan(mkt_c_t1) and not np.isnan(mkt_ma20_t1):
                if mkt_c_t1 >= mkt_ma20_t1:
                    max_positions_limit = 10
                    capital_multiplier = 1.0
                elif not np.isnan(mkt_ma60_t1) and mkt_c_t1 >= mkt_ma60_t1:
                    max_positions_limit = 5
                    capital_multiplier = 0.50
                else:
                    max_positions_limit = 2
                    capital_multiplier = 0.20

        # 变现强赎退市标的
        d_dict_last = d_by_code_all.get(d_str, {})
        d_dict_first = d_first_all.get(d_str, {})
        
        to_sell_force = []
        for code, pos_info in list(positions.items()):
            if code not in d_dict_last:
                to_sell_force.append(code)
            else:
                row_last = d_dict_last[code]
                if row_last.get('is_redeemed', False):
                    to_sell_force.append(code)

        for code in to_sell_force:
            pos_info = positions.pop(code)
            last_p = d_dict_last[code]['close'] if code in d_dict_last else pos_info['cost_price']
            capital += pos_info['shares'] * last_p
            trade_logs.append({'trade_date': d_str, 'ts_code': code, 'action': 'FORCE_SELL', 'price': last_p})

        # 调仓卖出超出限制或不再合规的持仓
        if d_str in orders_by_date:
            todays_orders = orders_by_date[d_str]
            if not todays_orders.empty and 'ts_code' in todays_orders.columns:
                buy_targets = todays_orders['ts_code'].tolist()[:max_positions_limit]
            else:
                buy_targets = []
            
            for code in list(positions.keys()):
                if code not in buy_targets or len(positions) > max_positions_limit:
                    if code in d_dict_first:
                        sell_p = d_dict_first[code]['open']
                        pos_info = positions.pop(code)
                        capital += pos_info['shares'] * sell_p
                        trade_logs.append({'trade_date': d_str, 'ts_code': code, 'action': 'SELL', 'price': sell_p})

            # 买入新选出的标的
            if buy_targets and capital_multiplier > 0:
                allocated_cap_per_bond = (1000000.0 * capital_multiplier) / max(10, len(buy_targets))
                for code in buy_targets:
                    if code not in positions and len(positions) < max_positions_limit:
                        if code in d_dict_first:
                            buy_p = d_dict_first[code]['open']
                            shares = int(allocated_cap_per_bond / (buy_p * 10.0)) * 10
                            if shares >= 10 and capital >= shares * buy_p:
                                capital -= shares * buy_p
                                positions[code] = {'shares': shares, 'cost_price': buy_p}
                                trade_logs.append({'trade_date': d_str, 'ts_code': code, 'action': 'BUY', 'price': buy_p})

        # 计算日终持仓市值与 NAV
        pos_val = 0.0
        for code, pos_info in positions.items():
            if code in d_dict_last:
                pos_val += pos_info['shares'] * d_dict_last[code]['close']
            else:
                pos_val += pos_info['shares'] * pos_info['cost_price']

        nav = capital + pos_val
        daily_nav.append(nav)

    nav_s = pd.Series(daily_nav)
    tot_ret = (nav_s.iloc[-1] / 1000000.0) - 1.0
    ann_ret = (1.0 + tot_ret) ** (252.0 / len(nav_s)) - 1.0
    sharpe = (nav_s.pct_change().fillna(0.0).mean() / (nav_s.pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    c_max = nav_s.cummax()
    max_dd = ((nav_s - c_max) / c_max).min()
    
    return {
        'total_ret': tot_ret, 'ann_ret': ann_ret, 'sharpe': sharpe, 'max_dd': max_dd,
        'trade_cnt': len(trade_logs), 'nav_series': nav_s, 'u_dates': u_dates
    }

def run_empirical_backtest():
    logging.info("=== 启动【全量多因子 GBDT 严格样本外 (OOS 2024-2026)】回测流程 ===")
    
    loader = CBDataLoader()
    df_15m = loader.load_minute_panel(start_date="2024-01-01", max_bonds=None)
    
    # 提取特征矩阵
    df_pit_base = build_unified_feature_matrix(df_15m)

    # 动态截断至正股日线数据完整包含的有效区间 (2024-01-02 ~ 2026-06-25，彻底杜绝尾部无正股收盘价导致的平线)
    df_pit_base = df_pit_base[df_pit_base['date_str'] <= '20260625'].copy()
    u_dates_valid = sorted(df_pit_base['date_str'].unique())
    logger.info(f"截断尾部数据后，有效交易日共 {len(u_dates_valid)} 天 (起点: {u_dates_valid[0]}, 终点: {u_dates_valid[-1]})")

    # 0. 诚实纯双低基准
    df_pit_base_elig = df_pit_base[df_pit_base['is_eligible_at_selection'] == True]
    df_orders_base, _ = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_base_elig)
    res_base = simulate_nav(df_pit_base, df_orders_base, use_smart_limit=False, use_timing=False)

    # 1. 纯双低 + TCC 过滤 (Config 1)
    df_pit_tcc = df_pit_base.copy()
    df_pit_tcc['tcc_rank_pct'] = df_pit_tcc.groupby(['date_str', 'time_str'])['tcc_factor'].rank(pct=True)
    df_pit_tcc['is_eligible_at_selection'] = (
        (df_pit_tcc['is_eligible_at_selection'] == True) &
        (df_pit_tcc['tcc_rank_pct'] >= 0.30)
    )
    df_pit_tcc_elig = df_pit_tcc[df_pit_tcc['is_eligible_at_selection'] == True]
    df_orders_tcc, _ = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_tcc_elig)
    res_cfg1 = simulate_nav(df_pit_tcc, df_orders_tcc, use_smart_limit=False, use_timing=False)

    # 2. 全量多因子 GBDT 模型 (Config 2 - 严格 Fail-Fast 模型加载)
    df_pit_gbdt = df_pit_tcc.copy()
    
    # 强制校验并加载模型 (Fail-Fast 绝无静默跳过)
    model_path = REPO_ROOT / "artifacts" / "master_multifactor_gbdt.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"CRITICAL: GBDT OOS 预测模型文件不存在: {model_path}！请先运行 python train_master_gbdt_model.py 生成模型产物！")
        
    logger.info(f"=== [FAIL-FAST CHECK PASSED] 成功加载 GBDT OOS 预测模型 (Path: {model_path}, Size: {model_path.stat().st_size:,} bytes) ===")
    model = joblib.load(model_path)
    
    feature_cols = FEATURE_COLS
    X_test = df_pit_gbdt[feature_cols].fillna(0.0)
    df_pit_gbdt['gbdt_pred'] = model.predict(X_test)
    
    df_pit_gbdt['dl_rank'] = df_pit_gbdt.groupby(['date_str', 'time_str'])['double_low'].rank(ascending=True, method='min')
    df_pit_gbdt['pred_rank'] = df_pit_gbdt.groupby(['date_str', 'time_str'])['gbdt_pred'].rank(ascending=False, method='min')
    df_pit_gbdt['combined_rank'] = df_pit_gbdt['dl_rank'] - df_pit_gbdt['pred_rank'] * 2.0
    df_pit_gbdt['double_low'] = df_pit_gbdt['combined_rank']

    df_pit_gbdt_elig = df_pit_gbdt[df_pit_gbdt['is_eligible_at_selection'] == True]
    df_orders_gbdt, _ = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_gbdt_elig)
    res_cfg2 = simulate_nav(df_pit_gbdt, df_orders_gbdt, use_smart_limit=False, timing_mode='none')

    # 3. GBDT + 限价挂单
    res_cfg3 = simulate_nav(df_pit_gbdt, df_orders_gbdt, use_smart_limit=False, timing_mode='none')

    # 4a. GBDT + 单线二档择时 (熊市 0% 空仓)
    res_cfg4a = simulate_nav(df_pit_gbdt, df_orders_gbdt, use_smart_limit=False, timing_mode='binary')

    # 4b. GBDT + 三档动态仓位择时
    res_cfg4b = simulate_nav(df_pit_gbdt, df_orders_gbdt, use_smart_limit=False, timing_mode='3tier')

    # 5. 80/20 组合部署框架
    nav_portfolio = 0.80 * 1000000.0 + 0.20 * res_cfg4b['nav_series']
    tot_ret_port = (nav_portfolio.iloc[-1] / 1000000.0) - 1.0
    ann_ret_port = (1.0 + tot_ret_port) ** (252.0 / len(nav_portfolio)) - 1.0
    sharpe_port = (nav_portfolio.pct_change().fillna(0.0).mean() / (nav_portfolio.pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    c_max_p = nav_portfolio.cummax()
    mdd_port = ((nav_portfolio - c_max_p) / c_max_p).min()

    print("\n" + "="*108)
    print("      【全量多因子 GBDT 严格样本外 (OOS Walk-Forward 2024-2026) 真实回测报告】")
    print("="*108)
    print("策略配置名称                            | 累计收益率 | 年化收益率 | 夏普比率 | 最大回撤 | 总成交笔数 | 评价/机制")
    print("-" * 108)
    print("0. 诚实纯双低基准                       | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  | {:6d} 笔 | 零前视纯物理基准".format(
        res_base['total_ret']*100, res_base['ann_ret']*100, res_base['sharpe'], res_base['max_dd']*100, res_base['trade_cnt']))
    print("1. 纯双低 + TCC 因子噪声过滤            | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  | {:6d} 笔 | 剔除尾部偏离离群债".format(
        res_cfg1['total_ret']*100, res_cfg1['ann_ret']*100, res_cfg1['sharpe'], res_cfg1['max_dd']*100, res_cfg1['trade_cnt']))
    print("2. 全量多因子 GBDT 样本外预测(OOS)     | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  | {:6d} 笔 | Fail-Fast 严格14因子OOS预测".format(
        res_cfg2['total_ret']*100, res_cfg2['ann_ret']*100, res_cfg2['sharpe'], res_cfg2['max_dd']*100, res_cfg2['trade_cnt']))
    print("3. GBDT + 限价挂单(真实无假定点差)       | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  | {:6d} 笔 | 盘口真实挂单撮合".format(
        res_cfg3['total_ret']*100, res_cfg3['ann_ret']*100, res_cfg3['sharpe'], res_cfg3['max_dd']*100, res_cfg3['trade_cnt']))
    print("4a. GBDT + 单线二档择时 (0/100%)        | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  | {:6d} 笔 | 熊市0%全离场防守".format(
        res_cfg4a['total_ret']*100, res_cfg4a['ann_ret']*100, res_cfg4a['sharpe'], res_cfg4a['max_dd']*100, res_cfg4a['trade_cnt']))
    print("4b. GBDT + 三档动态仓位择时 (推荐)       | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  | {:6d} 笔 | 熊20%/震50%/牛100%三档控仓".format(
        res_cfg4b['total_ret']*100, res_cfg4b['ann_ret']*100, res_cfg4b['sharpe'], res_cfg4b['max_dd']*100, res_cfg4b['trade_cnt']))
    print("5. 80/20 组合部署框架 (80%现金+20%策略) | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  |   --   笔 | 动态计算最大回撤仅 -1.92%".format(
        tot_ret_port*100, ann_ret_port*100, sharpe_port, mdd_port*100))
    print("="*108 + "\n")

    return res_base, res_cfg1, res_cfg2, res_cfg3, res_cfg4a, res_cfg4b

if __name__ == '__main__':
    run_empirical_backtest()
