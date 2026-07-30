# -*- coding: utf-8 -*-
"""
measure_q1_im_basis.py
Q1: 测量 IM (中证1000股指期货) 贴水历史分布、滚仓捕获收益与市场状态演化
"""
import os
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_DIR, 'research', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FUT_FILE = os.path.join(PROJECT_DIR, 'im_main_sina.csv')
SPOT_FILE = os.path.join(PROJECT_DIR, 'zz1000_spot_daily.csv')
REPORT_FILE = os.path.join(RESULTS_DIR, 'q1_im_basis_report.txt')

def run_q1_measurement():
    print(">>> Running Q1: IM Futures Basis & Rollover Yield Measurement...", flush=True)
    
    if not os.path.exists(FUT_FILE) or not os.path.exists(SPOT_FILE):
        raise FileNotFoundError(f"Missing data files: {FUT_FILE} or {SPOT_FILE}")
        
    df_fut = pd.read_csv(FUT_FILE)
    df_fut['date'] = pd.to_datetime(df_fut['日期'])
    df_fut = df_fut.set_index('date')[['收盘价']].rename(columns={'收盘价': 'fut_close'})
    
    df_spot = pd.read_csv(SPOT_FILE)
    df_spot['date'] = pd.to_datetime(df_spot['date'])
    df_spot = df_spot.set_index('date')[['close']].rename(columns={'close': 'spot_close'})
    
    df = df_fut.join(df_spot, how='inner').dropna()
    print(f"Aligned Data Points: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")
    
    # 收益率与基差计算
    df['fut_ret'] = df['fut_close'].pct_change().fillna(0.0)
    df['spot_ret'] = df['spot_close'].pct_change().fillna(0.0)
    df['basis_pct'] = (df['fut_close'] / df['spot_close'] - 1.0) * 100.0  # 基差 %
    df['excess_daily'] = df['fut_ret'] - df['spot_ret']
    
    # 分年度滚仓捕获 (Annualized Roll Capture)
    df['year'] = df.index.year
    years = sorted(df['year'].unique())
    
    report = []
    report.append("==========================================================================")
    report.append("          Q1 Report: IM Futures Basis & Rollover Yield Measurement         ")
    report.append("==========================================================================")
    report.append(f"Sample Period: {df.index.min().date()} to {df.index.max().date()} ({len(df)} trading days)")
    report.append("==========================================================================")
    
    report.append("\n[1. Year-by-Year Rollover Capture (IM Futures vs CSI 1000 Spot)]")
    report.append(f"{'Year':<6} | {'IM Fut Ret':<12} | {'Spot Ret':<10} | {'Roll Capture':<13} | {'Ann. Yield':<12} | {'Days':<6}")
    report.append("--------------------------------------------------------------------------")
    
    year_stats = []
    for y in years:
        g = df[df['year'] == y]
        fut_cum = (1.0 + g['fut_ret']).prod() - 1.0
        spot_cum = (1.0 + g['spot_ret']).prod() - 1.0
        roll_cap = (1.0 + fut_cum) / (1.0 + spot_cum) - 1.0
        n_days = len(g)
        ann_yield = (1.0 + roll_cap) ** (252.0 / n_days) - 1.0 if n_days > 20 else np.nan
        year_stats.append((y, fut_cum, spot_cum, roll_cap, ann_yield, n_days))
        report.append(f"{y:<6} | {fut_cum:<+12.2%} | {spot_cum:<+10.2%} | {roll_cap:<+13.2%} | {ann_yield:<+12.2%} | {n_days:<6}")
        
    full_fut_cum = (1.0 + df['fut_ret']).prod() - 1.0
    full_spot_cum = (1.0 + df['spot_ret']).prod() - 1.0
    full_roll_cap = (1.0 + full_fut_cum) / (1.0 + full_spot_cum) - 1.0
    full_days = len(df)
    full_ann_yield = (1.0 + full_roll_cap) ** (252.0 / full_days) - 1.0
    
    report.append("--------------------------------------------------------------------------")
    report.append(f"{'Full':<6} | {full_fut_cum:<+12.2%} | {full_spot_cum:<+10.2%} | {full_roll_cap:<+13.2%} | {full_ann_yield:<+12.2%} | {full_days:<6}")
    
    # 2. 基差水平分位数
    report.append("\n==========================================================================")
    report.append("[2. IM Futures Basis Distribution (Futures / Spot - 1)]")
    report.append("--------------------------------------------------------------------------")
    b_mean = df['basis_pct'].mean()
    b_std = df['basis_pct'].std()
    b_p10 = df['basis_pct'].quantile(0.10)
    b_p25 = df['basis_pct'].quantile(0.25)
    b_median = df['basis_pct'].median()
    b_p75 = df['basis_pct'].quantile(0.75)
    b_p90 = df['basis_pct'].quantile(0.90)
    
    report.append(f"Mean Basis:   {b_mean:+.2f}%  (Std: {b_std:.2f}%)")
    report.append(f"Median Basis: {b_median:+.2f}%")
    report.append(f"Quantiles:    P10: {b_p10:+.2f}% | P25: {b_p25:+.2f}% | P75: {b_p75:+.2f}% | P90: {b_p90:+.2f}%")
    report.append(f"Min / Max:    Min: {df['basis_pct'].min():+.2f}% | Max: {df['basis_pct'].max():+.2f}%")
    
    # 3. 市场状态下的基差变化 (Bull vs Bear Regime)
    report.append("\n==========================================================================")
    report.append("[3. Basis & Rollover Yield by Market Regimes]")
    report.append("--------------------------------------------------------------------------")
    bull_days = df[df['spot_ret'] > 0]
    bear_days = df[df['spot_ret'] <= 0]
    
    report.append(f"Bull Days (Spot > 0, N={len(bull_days)}): Mean Basis = {bull_days['basis_pct'].mean():+.2f}%, Avg Daily Excess = {bull_days['excess_daily'].mean()*100:+.3f}%")
    report.append(f"Bear Days (Spot <= 0, N={len(bear_days)}): Mean Basis = {bear_days['basis_pct'].mean():+.2f}%, Avg Daily Excess = {bear_days['excess_daily'].mean()*100:+.3f}%")
    
    report.append("\n==========================================================================")
    report.append("[4. Strategic Findings for Lever 4 (IM Basis Capture)]")
    report.append("--------------------------------------------------------------------------")
    report.append(f"1. Average Annualized Rollover Yield: {full_ann_yield:+.2%}/year")
    report.append("2. Under 15% Futures Margin + 85% Cash Carry (e.g. 2.0% Treasury repo):")
    report.append(f"   Total Expected Equity Return ≈ Spot Return + {full_ann_yield:+.2%} (Basis) + {0.85*2.0:.2%}% (Cash Carry) = Spot + {full_ann_yield + 0.85*0.02:+.2%}/year")
    report.append("==========================================================================")
    
    report_text = "\n".join(report)
    print(report_text, flush=True)
    
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"Saved Q1 report to {REPORT_FILE}")

if __name__ == '__main__':
    run_q1_measurement()
