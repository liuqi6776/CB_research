# -*- coding: utf-8 -*-
"""
measure_q3_correlation_matrix.py
Q3: 测量多资产（股票ETF、转债ETF/指数、国债ETF）真实相关性矩阵、滚动相关性及 40/40/20 组合抗跌能力
"""
import os
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'etf-valuation-strategy', 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'research', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

REPORT_FILE = os.path.join(RESULTS_DIR, 'q3_correlation_report.txt')

def load_asset_data():
    files = {
        'Stock_HS300': os.path.join(DATA_DIR, 'hs300_daily.csv'),
        'Stock_ZZ1000': os.path.join(DATA_DIR, 'zz1000_daily.csv'),
        'CB_Index': os.path.join(DATA_DIR, 'cbond_daily.csv'),
        'Treasury_Bond': os.path.join(DATA_DIR, 'bond_etf_daily.csv')
    }
    
    dfs = []
    for name, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # 找到日期与收盘价列
            date_col = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower() or '日期' in c][0]
            close_col = [c for c in df.columns if 'close' in c.lower() or '收盘' in c][0]
            
            df['date'] = pd.to_datetime(df[date_col].astype(str))
            df = df.set_index('date')[[close_col]].rename(columns={close_col: name})
            dfs.append(df)
            
    panel = pd.concat(dfs, axis=1, join='inner').sort_index().dropna()
    returns = panel.pct_change().dropna()
    return panel, returns

def run_q3_measurement():
    print(">>> Running Q3: Cross-Asset Correlation & 40/40/20 Portfolio Simulation...", flush=True)
    panel, returns = load_asset_data()
    print(f"Loaded Asset Data: {len(returns)} days ({returns.index.min().date()} to {returns.index.max().date()})")
    
    report = []
    report.append("==========================================================================")
    report.append("     Q3 Report: Cross-Asset Correlation Matrix & 40/40/20 Portfolio      ")
    report.append("==========================================================================")
    report.append(f"Sample Period: {returns.index.min().date()} to {returns.index.max().date()} ({len(returns)} trading days)")
    report.append("==========================================================================")
    
    # 1. 全样本相关性矩阵
    report.append("\n[1. Full-Sample Daily Return Correlation Matrix]")
    report.append("--------------------------------------------------------------------------")
    corr_full = returns.corr()
    report.append(corr_full.to_string())
    
    # 2. 压力时期相关性矩阵 (Stress Period Analysis)
    stress_periods = {
        '2015 Crash': ('2015-06-01', '2015-12-31'),
        '2018 Bear': ('2018-01-01', '2018-12-31'),
        '2022 Drawdown': ('2022-01-01', '2022-12-31'),
        '2024 Squeeze': ('2024-01-01', '2024-03-31')
    }
    
    report.append("\n==========================================================================")
    report.append("[2. Stress-Period Correlation Analysis (Stock vs Treasury & CB)]")
    report.append("--------------------------------------------------------------------------")
    
    for name, (start, end) in stress_periods.items():
        sub_ret = returns.loc[start:end]
        if len(sub_ret) > 10:
            c = sub_ret.corr()
            report.append(f"\n--- {name} ({start} to {end}, N={len(sub_ret)}) ---")
            report.append(f"Stock_ZZ1000 vs Treasury_Bond Corr: {c.loc['Stock_ZZ1000', 'Treasury_Bond']:+.4f}")
            report.append(f"Stock_ZZ1000 vs CB_Index Corr:      {c.loc['Stock_ZZ1000', 'CB_Index']:+.4f}")
            report.append(f"CB_Index vs Treasury_Bond Corr:      {c.loc['CB_Index', 'Treasury_Bond']:+.4f}")
            
    # 3. 40/40/20 组合模拟 (40% Stock ZZ1000, 40% CB Index, 20% Treasury Bond)
    report.append("\n==========================================================================")
    report.append("[3. Portfolio Backtest Comparison: 100% Stock vs 40/40/20 Diversified]")
    report.append("--------------------------------------------------------------------------")
    
    # 100% 股票 (ZZ1000)
    ret_stock = returns['Stock_ZZ1000']
    nav_stock = (1.0 + ret_stock).cumprod()
    
    # 40/40/20 每日重平衡
    ret_p40 = 0.40 * returns['Stock_ZZ1000'] + 0.40 * returns['CB_Index'] + 0.20 * returns['Treasury_Bond']
    nav_p40 = (1.0 + ret_p40).cumprod()
    
    def calc_metrics(nav, ret):
        ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / len(nav)) - 1.0
        ann_vol = ret.std() * np.sqrt(252.0)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        dd = (nav / nav.cummax() - 1.0).min()
        return ann_ret, ann_vol, sharpe, dd

    ar_s, av_s, sh_s, dd_s = calc_metrics(nav_stock, ret_stock)
    ar_p, av_p, sh_p, dd_p = calc_metrics(nav_p40, ret_p40)
    
    report.append(f"{'Portfolio Strategy':<30} | {'Ann. Ret':<10} | {'Ann. Vol':<10} | {'Sharpe':<8} | {'Max DD':<9}")
    report.append("--------------------------------------------------------------------------")
    report.append(f"{'100% Stock (ZZ1000)':<30} | {ar_s:<+10.2%} | {av_s:<10.2%} | {sh_s:<8.3f} | {dd_s:<+9.2%}")
    report.append(f"{'40/40/20 (Stock/CB/Treasury)':<30} | {ar_p:<+10.2%} | {av_p:<10.2%} | {sh_p:<8.3f} | {dd_p:<+9.2%}")
    report.append("--------------------------------------------------------------------------")
    report.append(f"Volatility Reduction: {av_s:.2%} -> {av_p:.2%} (Cut by {1 - av_p/av_s:.1%})")
    report.append(f"Max Drawdown Improvement: {dd_s:.2%} -> {dd_p:.2%} (Cut by {1 - dd_p/dd_s:.1%})")
    report.append(f"Sharpe Improvement: {sh_s:.3f} -> {sh_p:.3f} (+{sh_p - sh_s:.3f})")
    report.append("==========================================================================")
    
    report_text = "\n".join(report)
    print(report_text, flush=True)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Saved Q3 report to {REPORT_FILE}")

if __name__ == '__main__':
    run_q3_measurement()
