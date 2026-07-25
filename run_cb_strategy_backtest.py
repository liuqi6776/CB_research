# -*- coding: utf-8 -*-

"""
可转债 5 分钟线多因子策略回测与绩效评估主程序
Main Entry for A-Share CB Minute Multi-Factor Quantitative Backtest & Evaluation
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cb_quant.data_loader import CBDataLoader
from cb_quant.factor_engine import CBFactorEngine
from cb_quant.scoring_engine import CBScoringEngine
from cb_quant.backtest_engine import CBMinuteBacktestEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def calculate_performance_metrics(df_equity, df_trades, initial_capital=1000000.0):
    """计算完整的量化交易绩效指标"""
    if df_equity.empty:
        return {}
        
    nav_series = df_equity['nav']
    total_return = (nav_series.iloc[-1] - initial_capital) / initial_capital
    
    # 按照 5 分钟 K 线计算收益率序列 (每天 48 根 K 线，一年 242 交易日 = 11,616 根 K 线)
    returns = nav_series.pct_change().dropna()
    
    ann_factor = 11616.0
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    sharpe_ratio = (mean_ret / (std_ret + 1e-8)) * np.sqrt(ann_factor) if std_ret > 0 else 0.0
    
    # 最大回撤计算
    cummax = nav_series.cummax()
    drawdowns = (nav_series - cummax) / cummax
    max_drawdown = drawdowns.min()
    
    # 年化收益率计算
    num_bars = len(nav_series)
    years = num_bars / ann_factor
    ann_return = (1.0 + total_return) ** (1.0 / max(years, 0.01)) - 1.0 if total_return > -1.0 else -1.0
    
    calmar_ratio = ann_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0

    # 交易胜率与盈亏比统计
    if not df_trades.empty and 'action' in df_trades.columns:
        sell_trades = df_trades[df_trades['action'].isin(['STOP_LOSS', 'RANK_DECAY_SELL'])]
        total_trades = len(sell_trades)
        win_trades = len(sell_trades[sell_trades['pnl'] > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0
        
        avg_win = sell_trades[sell_trades['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0.0
        avg_loss = abs(sell_trades[sell_trades['pnl'] < 0]['pnl'].mean()) if (total_trades - win_trades) > 0 else 1.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    else:
        total_trades = 0
        win_rate = 0.0
        win_loss_ratio = 0.0

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'win_loss_ratio': win_loss_ratio,
        'final_nav': nav_series.iloc[-1]
    }

def main():
    logging.info("=== 开始运行 A股可转债 5分钟线多因子策略回测 ===")
    
    # 1. 加载数据面板 (默认加载 2024 至 2026 最新数据)
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=150)
    
    # 2. 计算分钟级多因子
    logging.info("计算分钟级多因子指标 (Smooth Momentum, Alpha, Amplitude, Volume Ratio)...")
    df_factors = CBFactorEngine.compute_factors(df_panel)
    
    # 3. 横截面百分位打分与选债
    logging.info("执行横截面 Percentile 打分与选债引擎...")
    scoring_engine = CBScoringEngine(top_n=8, min_amount=300000.0)
    df_scored = scoring_engine.compute_cross_sectional_scores(df_factors)
    
    # 4. 运行分钟级 T+0 事件驱动回测
    logging.info("启动 T+0 分钟级事件驱动回测引擎 (含 0.5% 动态止损与离场补位)...")
    bt_engine = CBMinuteBacktestEngine(
        initial_capital=1000000.0,
        top_n=8,
        rank_decay_buffer=5,
        stop_loss_pct=0.005,      # 0.5% 动态止损
        single_slippage=0.001,    # 单边 0.1% 滑点
        commission=0.00005        # 万0.5 佣金
    )
    
    df_equity, df_trades = bt_engine.run_backtest(df_scored)
    
    # 5. 计算并打印绩效分析报告
    metrics = calculate_performance_metrics(df_equity, df_trades)
    
    print("\n" + "="*50)
    print("      可转债 5 分钟线多因子策略绩效评估报告")
    print("="*50)
    print(f"初始资金:           1,000,000.00 元")
    print(f"期末资金:           {metrics.get('final_nav', 0.0):,.2f} 元")
    print(f"累计收益率:         {metrics.get('total_return', 0.0)*100:.2f}%")
    print(f"年化收益率:         {metrics.get('ann_return', 0.0)*100:.2f}%")
    print(f"夏普比率 (Sharpe):   {metrics.get('sharpe_ratio', 0.0):.2f}")
    print(f"卡尔玛比率 (Calmar): {metrics.get('calmar_ratio', 0.0):.2f}")
    print(f"最大回撤 (MaxDD):   {metrics.get('max_drawdown', 0.0)*100:.2f}%")
    print(f"平仓交易次数:       {metrics.get('total_trades', 0)} 次")
    print(f"交易胜率 (WinRate): {metrics.get('win_rate', 0.0)*100:.2f}%")
    print(f"盈亏比 (Profit/Loss):{metrics.get('win_loss_ratio', 0.0):.2f}")
    print("="*50 + "\n")
    
    # 6. 绘制并保存策略净值走势图
    if not df_equity.empty:
        plt.figure(figsize=(12, 6))
        plt.plot(pd.to_datetime(df_equity['trade_time']), df_equity['nav'], label='Strategy Net Asset Value (NAV)', color='#1f77b4', linewidth=1.5)
        plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
        plt.title("A-Share Convertible Bond 5-Min Multi-Factor Strategy Equity Curve", fontsize=14)
        plt.xlabel("Trade Time", fontsize=12)
        plt.ylabel("Net Asset Value (RMB)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        
        output_img = "cb_quant_strategy_equity.png"
        plt.savefig(output_img, dpi=300)
        logging.info(f"策略净值曲线图已生成存至: {os.path.abspath(output_img)}")

if __name__ == "__main__":
    main()
