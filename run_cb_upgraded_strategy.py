# -*- coding: utf-8 -*-

"""
升维可转债 LightGBM 策略主程序：包含置信度过滤、滚动窗口训练、分段行情评估及盈亏分析
Upgraded LightGBM Strategy Entry Point: Confidence Filter, Rolling Training, & Segment Evaluation
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def analyze_trades_detail(df_trades):
    """详细拉出平仓交易的盈亏比、最大盈亏及连续亏损统计"""
    if df_trades.empty or 'action' not in df_trades.columns:
        return {}
        
    sells = df_trades[df_trades['action'] != 'BUY'].copy()
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

    # 计算最大连续亏损次数
    sells['is_loss'] = (sells['pnl'] < 0).astype(int)
    consec_losses = 0
    max_consec_losses = 0
    for l in sells['is_loss']:
        if l == 1:
            consec_losses += 1
            max_consec_losses = max(max_consec_losses, consec_losses)
        else:
            consec_losses = 0

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
        'max_consec_losses': max_consec_losses
    }

def main():
    logging.info("=== 开始运行 升维可转债 LightGBM 置信度+动态止损策略 ===")
    
    # 1. 加载 5 分钟 K 线行情面板
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2024-01-01", end_date="2026-07-25", max_bonds=250)
    
    # 2. 计算特征工程与 60 分钟超额 Label
    logging.info("构建市场微观体征、ATR 真实波幅与二分类上行 Alpha Label...")
    df_ml_factors = CBMLFactorEngine.compute_ml_features(df_panel)
    
    # 3. 攻坚点2：划分训练集与样本外测试集 (滚动窗口重训)
    train_mask = (df_ml_factors['trade_time'] >= '2024-01-01') & (df_ml_factors['trade_time'] < '2025-01-01')
    test_mask = (df_ml_factors['trade_time'] >= '2025-01-01')
    
    df_train = df_ml_factors[train_mask]
    df_test = df_ml_factors[test_mask]
    
    logging.info(f"训练集样本数: {len(df_train):,} 行 | 样本外测试集样本数: {len(df_test):,} 行")
    
    # 4. 训练 LightGBM 二分类概率模型 (攻坚点1：置信度过滤 P >= 0.55)
    model_engine = CBMLModelEngine(n_estimators=150, learning_rate=0.03, max_depth=5, min_confidence=0.55)
    model_engine.train(df_train)
    
    logging.info("推断样本外二分类概率并应用置信度过滤 (P >= 0.55)...")
    df_predicted = model_engine.predict_ranks(df_test)
    
    # 5. 攻坚点3：动态 ATR 止损 + 时间平仓 + 波动率门槛 (>=0.8%)
    strategy_engine = CBMLStrategyEngine(
        initial_capital=1000000.0,
        top_n=8,                 # 持仓 8 只
        exit_rank_k=12,          # 掉出前 12 名软平仓
        atr_multiplier=1.5,      # 1.5 * ATR 动态止损
        max_hold_bars=24,        # 120 分钟时间平仓
        min_amp_threshold=0.008, # 振幅 >= 0.8% 覆盖交易成本
        cooldown_bars=6,         # 30 分钟冷却
        single_slippage=0.001,   # 单边 0.1% 滑点
        commission=0.00005       # 万0.5 佣金
    )
    
    df_equity, df_trades = strategy_engine.run_backtest(df_predicted)
    
    # 6. 计算交易明细盈亏比与绩效
    trade_stats = analyze_trades_detail(df_trades)
    
    if not df_equity.empty:
        nav_s = df_equity['nav']
        tot_ret = (nav_s.iloc[-1] - 1000000.0) / 1000000.0
        cummax = nav_s.cummax()
        mdd = ((nav_s - cummax) / cummax).min()
        
        num_bars = len(nav_s)
        num_days = num_bars / 48.0
        daily_trades = len(df_trades) / max(num_days, 1.0)
    else:
        tot_ret, mdd, daily_trades, num_days = 0.0, 0.0, 0.0, 0.0

    print("\n" + "="*65)
    print("      升维可转债 LightGBM 策略 (置信度+ATR动态止损) 绩效分析")
    print("="*65)
    print(f"测试时间段:         2025-01-01 至 2026-07-24 (共 {num_days:.1f} 个交易日)")
    print(f"初始资金:           1,000,000.00 元")
    print(f"期末净值:           {df_equity['nav'].iloc[-1] if not df_equity.empty else 0:,.2f} 元")
    print(f"累计收益率:         {tot_ret*100:.2f}%")
    print(f"最大回撤 (MaxDD):   {mdd*100:.2f}%")
    print(f"日均总交易笔数:     {daily_trades:.2f} 笔/天 [目标: 10 ~ 20 笔/天]")
    print(f"平仓胜率 (WinRate): {trade_stats.get('win_rate', 0.0)*100:.2f}%")
    print(f"盈亏比 (Profit/Loss):{trade_stats.get('profit_loss_ratio', 0.0):.2f}")
    print(f"平均单笔盈利:       {trade_stats.get('avg_win', 0.0):,.2f} 元")
    print(f"平均单笔亏损:       {trade_stats.get('avg_loss', 0.0):,.2f} 元")
    print(f"最大单笔盈利:       {trade_stats.get('max_win', 0.0):,.2f} 元")
    print(f"最大单笔亏损:       {trade_stats.get('max_loss', 0.0):,.2f} 元")
    print(f"最大连续亏损次数:   {trade_stats.get('max_consec_losses', 0)} 次")
    print("="*65 + "\n")
    
    # 7. 保存净值走势图
    if not df_equity.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(pd.to_datetime(df_equity['trade_time']), df_equity['nav'], label='Upgraded Strategy NAV (Confidence + ATR Stop)', color='#d62728', linewidth=1.8)
        plt.axhline(1000000.0, color='gray', linestyle='--', alpha=0.7, label='Initial Capital (1M)')
        plt.title("Upgraded Confidence-Filtered LightGBM Convertible Bond Strategy Equity Curve", fontsize=14)
        plt.xlabel("Trade Time", fontsize=12)
        plt.ylabel("Net Asset Value (RMB)", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left', fontsize=11)
        plt.tight_layout()
        
        output_img = "cb_upgraded_strategy_equity.png"
        plt.savefig(output_img, dpi=300)
        logging.info(f"升维策略净值曲线图已生成存至: {os.path.abspath(output_img)}")
        
    if not df_trades.empty:
        df_trades.to_csv("cb_upgraded_trades.csv", index=False, encoding='utf-8-sig')
        logging.info("升维策略交易日志已存至: cb_upgraded_trades.csv")

if __name__ == "__main__":
    main()
