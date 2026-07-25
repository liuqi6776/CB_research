# -*- coding: utf-8 -*-

"""
LightGBM 可转债 60 分钟超额预测与排名策略回测主程序
Main Execution for LightGBM Rank-Based 60-Min Alpha Convertible Bond Strategy
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cb_quant.data_loader import CBDataLoader
from cb_quant.ml_factor_engine import CBMLFactorEngine
from cb_quant.ml_model import CBMLModelEngine
from cb_quant.ml_strategy_engine import CBMLStrategyEngine

# 配置日志 / Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def evaluate_ml_performance(df_equity, df_trades, initial_capital=1000000.0):
    if df_equity.empty:
        return {}
        
    nav_series = df_equity['nav']
    total_return = (nav_series.iloc[-1] - initial_capital) / initial_capital
    
    returns = nav_series.pct_change().dropna()
    ann_factor = 11616.0 # 每天 48 根 5min 棒 * 242 交易日 = 11,616
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe_ratio = (mean_ret / (std_ret + 1e-8)) * np.sqrt(ann_factor) if std_ret > 0 else 0.0
    
    cummax = nav_series.cummax()
    drawdowns = (nav_series - cummax) / cummax
    max_drawdown = drawdowns.min()
    
    num_bars = len(nav_series)
    num_days = num_bars / 48.0
    years = num_bars / ann_factor
    
    ann_return = (1.0 + total_return) ** (1.0 / max(years, 0.01)) - 1.0 if total_return > -1.0 else -1.0
    calmar_ratio = ann_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0.0

    if not df_trades.empty and 'action' in df_trades.columns:
        sell_trades = df_trades[df_trades['action'] != 'BUY']
        total_trades = len(sell_trades) # 平仓卖出笔数
        all_trades_count = len(df_trades) # 总买卖笔数 (包含买入和卖出)
        
        daily_trades = all_trades_count / max(num_days, 1.0)
        daily_sells = total_trades / max(num_days, 1.0)
        
        win_trades = len(sell_trades[sell_trades['pnl'] > 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0
        
        avg_win = sell_trades[sell_trades['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0.0
        avg_loss = abs(sell_trades[sell_trades['pnl'] < 0]['pnl'].mean()) if (total_trades - win_trades) > 0 else 1.0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    else:
        total_trades = 0
        all_trades_count = 0
        daily_trades = 0.0
        daily_sells = 0.0
        win_rate = 0.0
        win_loss_ratio = 0.0

    return {
        'total_return': total_return,
        'ann_return': ann_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'total_trades': total_trades,
        'all_trades_count': all_trades_count,
        'daily_trades': daily_trades,
        'daily_sells': daily_sells,
        'win_rate': win_rate,
        'win_loss_ratio': win_loss_ratio,
        'final_nav': nav_series.iloc[-1],
        'total_days': num_days
    }

def main():
    logging.info("=== 开始运行 LightGBM 排名 60 分钟超额预测策略 ===")
    
    # 1. 加载 5 分钟 K 线行情面板
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2024-06-01", end_date="2026-07-25", max_bonds=200)
    
    # 2. 计算特征工程与 60 分钟超额标签
    logging.info("构建 LightGBM 多时段特征、日内时空特征与 60 分钟超额收益 Label...")
    df_ml_factors = CBMLFactorEngine.compute_ml_features(df_panel)
    
    # 3. 划分样本内训练集 (In-Sample Train) 与样本外测试集 (Out-of-Sample Test)
    train_mask = (df_ml_factors['trade_time'] >= '2024-06-01') & (df_ml_factors['trade_time'] < '2025-06-01')
    test_mask = (df_ml_factors['trade_time'] >= '2025-06-01')
    
    df_train = df_ml_factors[train_mask]
    df_test = df_ml_factors[test_mask]
    
    logging.info(f"样本内训练集样本数: {len(df_train):,} 行 | 样本外测试集样本数: {len(df_test):,} 行")
    
    # 4. 训练 LightGBM 模型并预测样本外排名
    model_engine = CBMLModelEngine(n_estimators=120, learning_rate=0.03, max_depth=5)
    model_engine.train(df_train)
    
    logging.info("推断样本外 60 分钟超额预测值并生成横截面排名...")
    df_predicted = model_engine.predict_ranks(df_test)
    
    # 5. 运行基于 Rank > 12 软退出、-1.5% 硬止损、30分钟买入冷却及即时补位的事件驱动回测
    strategy_engine = CBMLStrategyEngine(
        initial_capital=1000000.0,
        top_n=8,                 # 持仓 8 只
        exit_rank_k=18,          # 掉出 Top 18 名软退出
        min_hold_bars=6,         # 最小持仓 30 分钟 (6 根 K 线)
        hard_stop_loss=0.015,    # -1.5% 硬止损
        cooldown_bars=6,         # 卖出后 30 分钟冷却
        single_slippage=0.001,   # 单边 0.1% 滑点
        commission=0.00005       # 万0.5 佣金
    )
    
    df_equity, df_trades = strategy_engine.run_backtest(df_predicted)
    
    # 6. 计算绩效指标与每日交易笔数
    metrics = evaluate_ml_performance(df_equity, df_trades)
    
    print("\n" + "="*65)
    print("      LightGBM 排名 60 分钟超额预测策略 (样本外) 绩效报告")
    print("="*65)
    print(f"测试时间段:         2025-06-01 至 2026-07-24 (共 {metrics.get('total_days', 0):.1f} 个交易日)")
    print(f"初始资金:           1,000,000.00 元")
    print(f"期末净值:           {metrics.get('final_nav', 0.0):,.2f} 元")
    print(f"累计收益率:         {metrics.get('total_return', 0.0)*100:.2f}%")
    print(f"年化收益率:         {metrics.get('ann_return', 0.0)*100:.2f}%")
    print(f"夏普比率 (Sharpe):   {metrics.get('sharpe_ratio', 0.0):.2f}")
    print(f"卡尔玛比率 (Calmar): {metrics.get('calmar_ratio', 0.0):.2f}")
    print(f"最大回撤 (MaxDD):   {metrics.get('max_drawdown', 0.0)*100:.2f}%")
    print(f"总买卖交易次数:     {metrics.get('all_trades_count', 0)} 笔 (买入+卖出)")
    print(f"日均总交易笔数:     {metrics.get('daily_trades', 0.0):.2f} 笔/天 [目标: 10 ~ 20 笔/天]")
    print(f"日均轮换卖出笔数:   {metrics.get('daily_sells', 0.0):.2f} 笔/天 [约每只债每天轮换1次]")
    print(f"交易胜率 (WinRate): {metrics.get('win_rate', 0.0)*100:.2f}%")
    print(f"盈亏比 (Profit/Loss):{metrics.get('win_loss_ratio', 0.0):.2f}")
    print("="*65 + "\n")
    
    # 7. 导出净值曲线图及交易日志
    if not df_equity.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(df_equity['trade_time']), df_equity['nav'], label='LightGBM Rank Strategy NAV (Out-of-Sample)', color='#1f77b4', linewidth=1.8)
        plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
        plt.title("LightGBM Rank-Based 60-Min Alpha Convertible Bond Strategy Equity Curve", fontsize=14)
        plt.xlabel("Trade Time", fontsize=12)
        plt.ylabel("Net Asset Value (RMB)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        
        output_img = "cb_lgb_strategy_equity.png"
        plt.savefig(output_img, dpi=300)
        logging.info(f"策略净值曲线图已生成存至: {os.path.abspath(output_img)}")
        
    if not df_trades.empty:
        df_trades.to_csv("cb_lgb_trades.csv", index=False, encoding='utf-8-sig')
        logging.info("策略交易日志已存至: cb_lgb_trades.csv")

if __name__ == "__main__":
    main()
