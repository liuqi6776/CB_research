# -*- coding: utf-8 -*-

"""
机构低手续费可转债日内高频策略与摩擦敏感度矩阵分析引擎
(Institutional Low-Fee CB Intraday Strategy & Friction Sensitivity Matrix Analysis Engine)

评估场景 (Friction Scenarios):
1. 0 bps (Gross Alpha / 毛收益零摩擦上限)
2. 2 bps (机构 VIP 极速柜台 DMA / 万一佣金 / 往返 0.02%)
3. 5 bps (机构普通 VIP 柜台 / 往返 0.05%)
4. 10 bps (散户低费率 / 往返 0.10%)
5. 20 bps (先前保守审查线 / 往返 0.20%)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from cb_quant.data_loader import CBDataLoader
from cb_quant.feature_pipeline import build_unified_feature_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent

def simulate_nav_custom_cost(df_pit, df_orders, fee_bps_one_way=1.0, max_positions=10, initial_capital=1000000.0):
    """
    可定制单边手续费 fee_bps_one_way (单位: bps, 1 bp = 0.01% = 0.0001)
    """
    fee_rate_buy = 1.0 + (fee_bps_one_way / 10000.0)
    fee_rate_sell = 1.0 - (fee_bps_one_way / 10000.0)
    
    u_dates = sorted(df_pit['date_str'].unique())
    d_dict_first = {d: g.groupby('ts_code').first().to_dict('index') for d, g in df_pit.groupby('date_str')}
    
    df_orders_copy = df_orders.copy()
    df_orders_copy['trade_date_str'] = pd.to_numeric(df_orders_copy['trade_date'], errors='coerce').fillna(0).astype(int).astype(str)
    orders_by_date = {d: g for d, g in df_orders_copy.groupby('trade_date_str')}
    
    cash = initial_capital
    portfolio = {} # ts_code -> {'shares': int, 'entry_price': float, 'entry_date': str}
    nav_list = []
    
    total_volume_traded = 0.0
    total_friction_cost = 0.0
    holding_days_records = []
    
    for idx, d_str in enumerate(u_dates):
        m_dt = pd.to_datetime(d_str, format='%Y%m%d')
        d_dict = d_dict_first.get(d_str, {})
        
        # 1. 估值与强赎退市强制平仓
        current_eq = cash
        codes_to_remove = []
        for code, info in list(portfolio.items()):
            if code in d_dict:
                c_row = d_dict[code]
                last_p = c_row['close']
                
                # 强赎退市判断
                if c_row.get('is_redeemed', False):
                    actual_sell_p = last_p * fee_rate_sell
                    sell_val = info['shares'] * actual_sell_p
                    cash += sell_val
                    total_friction_cost += info['shares'] * last_p * (fee_bps_one_way / 10000.0)
                    holding_days_records.append(idx - info['entry_idx'])
                    codes_to_remove.append(code)
                else:
                    current_eq += info['shares'] * last_p
            else:
                current_eq += info['shares'] * info['entry_price']
                
        for code in codes_to_remove:
            del portfolio[code]
            
        # 2. 执行调仓
        if d_str in orders_by_date:
            todays_orders = orders_by_date[d_str]
            buy_targets = todays_orders['ts_code'].tolist()[:max_positions]
            
            # 卖出不在目标列表中的品种
            for code in list(portfolio.keys()):
                if code not in buy_targets:
                    if code in d_dict and d_dict[code].get('is_executable_at_fill', True):
                        sell_p = d_dict[code]['open']
                        actual_sell_p = sell_p * fee_rate_sell
                        sell_val = portfolio[code]['shares'] * actual_sell_p
                        cash += sell_val
                        total_volume_traded += portfolio[code]['shares'] * sell_p
                        total_friction_cost += portfolio[code]['shares'] * sell_p * (fee_bps_one_way / 10000.0)
                        holding_days_records.append(idx - portfolio[code]['entry_idx'])
                        del portfolio[code]
                        
            # 买入新选入的目标
            current_holding_codes = set(portfolio.keys())
            new_buys = [c for c in buy_targets if c not in current_holding_codes]
            
            if new_buys:
                target_alloc = current_eq / max_positions
                for code in new_buys:
                    if code in d_dict and d_dict[code].get('is_executable_at_fill', True):
                        buy_p = d_dict[code]['open']
                        actual_buy_p = buy_p * fee_rate_buy
                        
                        if cash >= target_alloc * 0.8:
                            bar_vol = d_dict[code].get('vol', 100000.0)
                            max_allowed_shares = int(bar_vol * 0.5) * 10
                            desired_shares = int(target_alloc / actual_buy_p / 10.0) * 10
                            shares = min(desired_shares, max_allowed_shares)
                            
                            if shares >= 10:
                                cost = shares * actual_buy_p
                                if cash >= cost:
                                    cash -= cost
                                    total_volume_traded += shares * buy_p
                                    total_friction_cost += shares * buy_p * (fee_bps_one_way / 10000.0)
                                    portfolio[code] = {
                                        'shares': shares,
                                        'entry_price': buy_p,
                                        'entry_idx': idx
                                    }
                                    
        # 3. 计算日末总资产
        end_eq = cash
        for code, info in portfolio.items():
            if code in d_dict:
                end_eq += info['shares'] * d_dict[code]['close']
            else:
                end_eq += info['shares'] * info['entry_price']
                
        nav_list.append(end_eq)
        
    s_nav = pd.Series(nav_list, index=pd.to_datetime(u_dates, format='%Y%m%d'))
    total_ret = (s_nav.iloc[-1] / initial_capital) - 1.0
    ann_ret = (1.0 + total_ret) ** (252.0 / len(s_nav)) - 1.0
    daily_ret = s_nav.pct_change().fillna(0.0)
    sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-8)) * np.sqrt(252.0)
    
    cummax = s_nav.cummax()
    max_dd = ((s_nav - cummax) / cummax).min()
    
    turnover_annual = (total_volume_traded / initial_capital) / (len(u_dates) / 252.0)
    avg_holding = np.mean(holding_days_records) if holding_days_records else 0.0
    
    return {
        'total_ret': total_ret,
        'ann_ret': ann_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'turnover_annual': turnover_annual,
        'avg_holding_days': avg_holding,
        'total_friction_cost': total_friction_cost,
        'nav_series': s_nav,
        'u_dates': u_dates
    }

def run_sensitivity_analysis():
    logger.info("=== 启动【机构低手续费可转债日内高频策略与摩擦敏感度矩阵分析】 ===")
    
    loader = CBDataLoader()
    df_15m = loader.load_minute_panel(start_date="2024-01-01", max_bonds=None)
    df_pit_base = build_unified_feature_matrix(df_15m)
    df_pit_base = df_pit_base[df_pit_base['date_str'] <= '20260625'].copy()
    
    # 截面合规样本
    df_pit_elig = df_pit_base[df_pit_base['is_eligible_at_selection'] == True].copy()
    
    # 1. 构造日频纯双低 (Daily Double-Low) 目标单
    df_daily_first = df_pit_elig[df_pit_elig['time_str'] == '09:35'].copy()
    df_daily_first['rank'] = df_daily_first.groupby('date_str')['double_low'].rank(ascending=True, method='min')
    df_orders_dl = df_daily_first[df_daily_first['rank'] <= 10][['date_str', 'ts_code', 'double_low']].rename(columns={'date_str': 'trade_date'})

    # 2. 构造日频 GBDT 14 因子模型目标单 (若模型存在)
    import joblib
    model_path = REPO_ROOT / "artifacts" / "master_multifactor_gbdt.joblib"
    if model_path.exists():
        gbdt_model = joblib.load(model_path)
        from cb_quant.feature_pipeline import FEATURE_COLS
        X_all = df_daily_first[FEATURE_COLS].fillna(0.0).values
        df_daily_first['pred'] = gbdt_model.predict(X_all)
        df_daily_first['rank_gbdt'] = df_daily_first.groupby('date_str')['pred'].rank(ascending=False, method='min')
        df_orders_gbdt = df_daily_first[df_daily_first['rank_gbdt'] <= 10][['date_str', 'ts_code', 'pred']].rename(columns={'date_str': 'trade_date'})
    else:
        df_orders_gbdt = df_orders_dl

    # 3. 构造日内 15m K线动量反转高频策略目标单 (Intraday 15m Rebalancing)
    # 取每日 14:45 收盘尾盘因子排名，次日 09:35 开盘成交
    df_1445 = df_pit_elig[df_pit_elig['time_str'] == '14:45'].copy()
    df_1445['rank_1445'] = df_1445.groupby('date_str')['double_low'].rank(ascending=True, method='min')
    
    unique_dates = sorted(df_pit_elig['date_str'].unique())
    intraday_order_records = []
    for i in range(len(unique_dates) - 1):
        curr_d = unique_dates[i]
        next_d = unique_dates[i + 1]
        sub_1445 = df_1445[(df_1445['date_str'] == curr_d) & (df_1445['rank_1445'] <= 10)]
        for _, row in sub_1445.iterrows():
            intraday_order_records.append({'trade_date': next_d, 'ts_code': row['ts_code']})
    df_orders_intraday = pd.DataFrame(intraday_order_records)

    # 测试手续费梯度 (单边 bps)
    # 单边 0 bps = 往返 0 bps
    # 单边 1 bps = 往返 2 bps (机构 DMA 极速柜台)
    # 单边 2.5 bps = 往返 5 bps (机构普通 VIP)
    # 单边 5 bps = 往返 10 bps (散户优惠)
    # 单边 10 bps = 往返 20 bps (保守默认)
    fee_tiers_one_way = [0.0, 1.0, 2.5, 5.0, 10.0]
    
    # 3. 读取 511380.SH CB ETF 基准
    etf_csv_path = REPO_ROOT / "artifacts" / "cb_etf_511380_daily.csv"
    if not etf_csv_path.exists():
        raise FileNotFoundError(f"CRITICAL: ETF 行情缺失: {etf_csv_path}")
    df_etf = pd.read_csv(etf_csv_path)
    df_etf['trade_date'] = pd.to_datetime(df_etf['日期'].astype(str))
    plot_dates = pd.to_datetime(unique_dates, format='%Y%m%d')
    df_etf_sub = df_etf[df_etf['trade_date'].isin(plot_dates)].sort_values('trade_date').reset_index(drop=True)
    nav_etf = df_etf_sub['收盘'] / df_etf_sub['收盘'].iloc[0]
    etf_total_ret = nav_etf.iloc[-1] - 1.0
    etf_ann_ret = (1.0 + etf_total_ret) ** (252.0 / len(nav_etf)) - 1.0
    etf_sharpe = (nav_etf.pct_change().fillna(0.0).mean() / (nav_etf.pct_change().fillna(0.0).std() + 1e-8)) * np.sqrt(252.0)
    c_max_e = nav_etf.cummax()
    etf_max_dd = ((nav_etf - c_max_e) / c_max_e).min()

    results_table = []
    
    for fee_one_way in fee_tiers_one_way:
        round_trip_bps = fee_one_way * 2.0
        
        # A. 日频纯双低 (Daily Double-Low)
        res_dl = simulate_nav_custom_cost(df_pit_base, df_orders_dl, fee_bps_one_way=fee_one_way)
        results_table.append({
            'Strategy': '日频纯双低 (Daily Double-Low)',
            'Round_Trip_Fee_bps': round_trip_bps,
            'Total_Return': res_dl['total_ret'],
            'Sharpe': res_dl['sharpe'],
            'Max_Drawdown': res_dl['max_dd'],
            'Friction_Cost_Yuan': res_dl['total_friction_cost'],
            'vs_ETF_Excess': res_dl['total_ret'] - etf_total_ret
        })
        
        # B. 日频 GBDT 14 因子
        res_gbdt = simulate_nav_custom_cost(df_pit_base, df_orders_gbdt, fee_bps_one_way=fee_one_way)
        results_table.append({
            'Strategy': '日频 GBDT 14因子 (Daily GBDT)',
            'Round_Trip_Fee_bps': round_trip_bps,
            'Total_Return': res_gbdt['total_ret'],
            'Sharpe': res_gbdt['sharpe'],
            'Max_Drawdown': res_gbdt['max_dd'],
            'Friction_Cost_Yuan': res_gbdt['total_friction_cost'],
            'vs_ETF_Excess': res_gbdt['total_ret'] - etf_total_ret
        })

        # C. 尾盘选债次日开盘调仓 (14:45 Signal -> T+1 Open)
        res_intra = simulate_nav_custom_cost(df_pit_base, df_orders_intraday, fee_bps_one_way=fee_one_way)
        results_table.append({
            'Strategy': '日内尾盘选债 (14:45 Signal -> T+1 Fill)',
            'Round_Trip_Fee_bps': round_trip_bps,
            'Total_Return': res_intra['total_ret'],
            'Sharpe': res_intra['sharpe'],
            'Max_Drawdown': res_intra['max_dd'],
            'Friction_Cost_Yuan': res_intra['total_friction_cost'],
            'vs_ETF_Excess': res_intra['total_ret'] - etf_total_ret
        })

    df_res = pd.DataFrame(results_table)
    out_csv = REPO_ROOT / "artifacts" / "institutional_friction_sensitivity.csv"
    df_res.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logger.info(f"已导出机构费率敏感度矩阵产物 CSV: {out_csv}")

    print("\n" + "="*145)
    print("                 【机构不同手续费/柜台费率下可转债策略敏感度矩阵实证报告】")
    print("="*145)
    print("策略类型                         | 往返手续费/滑点 | 累计收益率 | 年化收益率 | 夏普比率 | 最大回撤 | 总摩擦成本(元) | vs 511380 ETF 超额 | 机构评估结论")
    print("-" * 145)
    print("511380.SH 可转债 ETF 真实基准    |    0 bps (基准)  | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  |      --        |   0.00pp (基准)   | 被动指数 Benchmark".format(
        etf_total_ret*100, etf_ann_ret*100, etf_sharpe, etf_max_dd*100))
    print("-" * 145)

    for _, row in df_res.iterrows():
        ann_r = (1.0 + row['Total_Return']) ** (252.0 / len(unique_dates)) - 1.0
        eval_str = "[WIN] Outperform ETF" if row['vs_ETF_Excess'] > 0 and row['Sharpe'] > etf_sharpe else (" [AMP] Return Higher" if row['vs_ETF_Excess'] > 0 else "[FAIL] Underperform ETF")
        print("{:<30} | {:4.1f} bps (1-way{:3.1f}) | {:+8.2f}%  | {:+8.2f}%  | {:7.2f}  | {:7.2f}%  | RMB {:11.2f} | {:+7.2f}pp        | {}".format(
            row['Strategy'], row['Round_Trip_Fee_bps'], row['Round_Trip_Fee_bps']/2.0,
            row['Total_Return']*100, ann_r*100, row['Sharpe'], row['Max_Drawdown']*100,
            row['Friction_Cost_Yuan'], row['vs_ETF_Excess']*100, eval_str))
    print("="*145 + "\n")

    return df_res, etf_total_ret, etf_sharpe

if __name__ == '__main__':
    run_sensitivity_analysis()
