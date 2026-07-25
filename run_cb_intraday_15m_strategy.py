# -*- coding: utf-8 -*-

"""
15分钟日内 T+0 零隔夜策略主程序 (结合双低风控围栏、早盘高波聚焦与 +1.5% 止盈 / -0.8% 止损)
15-Minute Intraday T+0 Strategy Execution Runner (Zero-Overnight)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cb_quant.data_loader import CBDataLoader
from cb_quant.traditional_factor_engine import CBTraditionalFactorEngine
from cb_quant.intraday_15m_engine import CBIntraday15mEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def analyze_trades_detail(df_trades):
    if df_trades.empty or 'action' not in df_trades.columns:
        return {}
        
    sells = df_trades[df_trades['action'] != 'INTRA_BUY'].copy()
    if sells.empty:
        return {}

    total_sells = len(sells)
    wins = sells[sells['pnl'] > 0]
    losses = sells[sells['pnl'] < 0]

    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_sells if total_sells > 0 else 0.0

    avg_win = wins['pnl'].mean() if win_count > 0 else 0.0
    avg_loss = abs(losses['pnl'].mean()) if loss_count > 0 else 1.0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    max_win = sells['pnl'].max()
    max_loss = sells['pnl'].min()
    action_counts = sells['action'].value_counts().to_dict()

    return {
        'total_sells': total_sells,
        'win_count': win_count,
        'loss_count': loss_count,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'max_win': max_win,
        'max_loss': max_loss,
        'action_counts': action_counts
    }

def main():
    logging.info("=== 开始运行 15分钟日内 T+0 零隔夜策略 ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    # 1. 转换为 15 分钟 K 线
    df_panel['trade_time'] = pd.to_datetime(df_panel['trade_time'])
    df_15m = df_panel.groupby(['ts_code', pd.Grouper(key='trade_time', freq='15min')]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'amount': 'sum'
    }).dropna().reset_index()
    
    # 2. 算双低因子与风控围栏 Top 30
    logging.info("计算双低风控围栏 (Top 30)...")
    df_trad = CBTraditionalFactorEngine.compute_traditional_factors(df_15m)
    df_15m = CBTraditionalFactorEngine.select_double_low_pool(df_trad, pool_size=30)
    df_15m['ret_15m'] = df_15m.groupby('ts_code')['close'].pct_change(1)
    
    # 3. 运行 15m 日内零隔夜策略 (+1.5% 止盈, -0.8% 止损, 14:45 强平)
    strategy_engine = CBIntraday15mEngine(
        initial_capital=1000000.0,
        top_n=5,
        take_profit=0.007,       # +0.7% 快吃止盈
        stop_loss=0.005,         # -0.5% 严割止损
        max_hold_bars=2,         # 最多持仓 30分钟
        single_slippage=0.0005,  # 0.05% 单边贴近真实盘口滑点
        commission=0.00005
    )
    
    df_equity, df_trades = strategy_engine.run_backtest(df_15m)
    trade_stats = analyze_trades_detail(df_trades)
    
    if not df_equity.empty:
        df_eq = df_equity.copy()
        df_eq['trade_time'] = pd.to_datetime(df_eq['trade_time'])
        df_eq['date'] = df_eq['trade_time'].dt.date
        
        daily_nav = df_eq.groupby('date')['nav'].last()
        daily_ret = daily_nav.pct_change().dropna()
        
        tot_ret = (daily_nav.iloc[-1] - 1000000.0) / 1000000.0
        num_days = len(daily_nav)
        ann_ret = (1.0 + tot_ret) ** (252.0 / num_days) - 1.0 if num_days > 0 else 0.0
        
        rf_daily = 0.02 / 252.0
        ex_ret = daily_ret - rf_daily
        sharpe = (ex_ret.mean() / (daily_ret.std() + 1e-8)) * np.sqrt(252.0)
        
        cummax = daily_nav.cummax()
        drawdown = (daily_nav - cummax) / cummax
        max_dd = drawdown.min()
        
        calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
        total_trades = trade_stats.get('total_sells', 0)
        daily_t = total_trades / float(num_days) if num_days > 0 else 0.0
    else:
        tot_ret, ann_ret, sharpe, max_dd, calmar, daily_t, num_days = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    print("\n" + "="*65)
    print("   15分钟日内 T+0 零隔夜策略 (双低围栏+硬止盈止损) 绩效分析")
    print("="*65)
    print(f"回测测试区间:       2025-01-01 至 2026-07-24 (共 {num_days} 个交易日)")
    print(f"初始资金:           1,000,000.00 元")
    print(f"期末资金:           {df_equity['nav'].iloc[-1] if not df_equity.empty else 0:,.2f} 元")
    print(f"绝对累计收益率:     {tot_ret*100:+.2f}%")
    print(f"年化收益率:         {ann_ret*100:+.2f}%")
    print(f"夏普比率 (Sharpe):  {sharpe:.2f}")
    print(f"卡玛比率 (Calmar):  {calmar:.2f}")
    print(f"最大回撤 (MaxDD):   {max_dd*100:.2f}%")
    print(f"日均平仓交易笔数:   {daily_t:.2f} 笔/天")
    print(f"平仓交易胜率:       {trade_stats.get('win_rate', 0.0)*100:.2f}%")
    print(f"★ 盈亏比:          {trade_stats.get('profit_loss_ratio', 0.0):.2f} (目标 > 1.20)")
    print(f"平均单笔盈利:       {trade_stats.get('avg_win', 0.0):,.2f} 元")
    print(f"平均单笔亏损:       {trade_stats.get('avg_loss', 0.0):,.2f} 元")
    print(f"隔夜持仓比例:       0.00% (每日 14:45 强制 100% 现金空仓)")
    print(f"离场类型分布:       {trade_stats.get('action_counts', {})}")
    print("="*65 + "\n")
    
    if not df_equity.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(df_equity['trade_time']), df_equity['nav'], label='15-Min Intraday T+0 Strategy NAV (Zero-Overnight)', color='#1f77b4', linewidth=2.0)
        plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
        plt.title("A-Share CB 15-Min Intraday T+0 Strategy Equity Curve", fontsize=14)
        plt.xlabel("Trade Time", fontsize=12)
        plt.ylabel("Net Asset Value (RMB)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        plt.savefig("cb_intraday_15m_equity.png", dpi=300)

if __name__ == "__main__":
    main()
