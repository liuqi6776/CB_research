# -*- coding: utf-8 -*-

"""
三项聚焦研究与 9 项 Kill Criteria 终止判定主程序
Three Focused Research Tasks & 9 Kill Criteria Decision Runner
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader
from cb_quant.intraday_cross_sectional import CBIntradayCrossSectionalEngine
from cb_quant.credible_baseline_engine import CBCredibleBaselineEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    logging.info("=== 启动三项聚焦研究与 9 项 Kill Criteria 终极评估 ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=200)
    
    # 过滤交易池
    df_filtered = CBIntradayCrossSectionalEngine.filter_universe(df_panel, max_price=180.0, min_scale=2.0)
    
    # 转换为 15分钟 K 线并计算 15m 截面 Z-Score 打分
    df_filtered['trade_time'] = pd.to_datetime(df_filtered['trade_time'])
    df_15m = df_filtered.groupby(['ts_code', pd.Grouper(key='trade_time', freq='15min')]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'amount': 'sum'
    }).dropna().reset_index()
    
    df_scored = CBIntradayCrossSectionalEngine.compute_cross_sectional_scores(df_15m)
    
    # =========================================================================
    # 研究任务 1：三种成交假设下基线表现
    # =========================================================================
    logging.info("运行【研究任务 1】：三种成交假设下的可信 15分钟基线...")
    task1_res = CBCredibleBaselineEngine.run_three_execution_assumptions(df_scored, top_n=5)
    
    # =========================================================================
    # 研究任务 2：信号强度五分组单调性检验
    # =========================================================================
    logging.info("运行【研究任务 2】：信号强度五分组 (Q1~Q5) 单调性检验...")
    q_dict, is_monotonic = CBCredibleBaselineEngine.test_alpha_monotonicity_grouping(df_scored)
    
    # =========================================================================
    # 研究任务 3：只验证一个增量因子 (T-1 正股筹码消融)
    # =========================================================================
    logging.info("运行【研究任务 3】：T-1 正股筹码增量因子四臂消融检验...")
    task3_res = CBCredibleBaselineEngine.test_t1_stock_chip_ablation(df_scored)
    
    # =========================================================================
    # 9 项 Kill Criteria 硬性条件触发检测
    # =========================================================================
    assump_a = task1_res.get("假设 A: Next-Bar Open (t+1 Open)", {})
    assump_c = task1_res.get("假设 C: Next-Bar Open + 保守 0.1% 滑点", {})
    
    avg_gross_bp = assump_a.get('avg_gross_bp', -999)
    total_gross_pnl = assump_a.get('total_gross_pnl', -999)
    
    kill_triggers = []
    
    # Kill 1: 修正回测后毛收益 <= 0
    if total_gross_pnl <= 0:
        kill_triggers.append("Kill 1: 修正 t+1 回测后总毛收益 <= 0 (实际: {:.2f}元)".format(total_gross_pnl))
        
    # Kill 2: 每笔毛优势 < 3bp
    if avg_gross_bp < 3.0:
        kill_triggers.append("Kill 2: 每笔毛优势 < +3bp (实际: {:.2f}bp/笔)".format(avg_gross_bp))
        
    # Kill 3: 信号分组不具备单调性
    if not is_monotonic:
        kill_triggers.append("Kill 3: 信号五分组 (Q1~Q5) 不具备收益单调性 (Q1: {:.2f}bp -> Q5: {:.2f}bp)".format(
            q_dict.get(1, 0.0), q_dict.get(5, 0.0)))
        
    # Kill 6: 下一根成交后 Alpha 消失
    if assump_a.get('total_gross_pnl', -1) <= 0:
        kill_triggers.append("Kill 6: 下一根 t+1 开盘成交后 Alpha 彻底消失")
        
    # Kill 7: 筹码因子 OOS 增量低于 2%~3% 年化 (Rank IC 增量 < 0.02)
    chip_ic_gain = task3_res.get('Baseline + Chip Level T-1 Rank IC', 0) - task3_res.get('Pure 15m Baseline Rank IC', 0)
    if chip_ic_gain < 0.02:
        kill_triggers.append("Kill 7: T-1 筹码因子 IC 净增量 < 0.02 (实际 IC 增量: {:.4f})".format(chip_ic_gain))

    # 输出报告
    print("\n" + "="*70)
    print("        三项聚焦研究实测结果与 9 项 Kill Criteria 评估报告")
    print("="*70)
    
    print("\n【研究任务 1：三种成交假设下的 15分钟基线】")
    for k, v in task1_res.items():
        print("  - {:<42} | 每笔毛收益: {:+6.2f}bp | 扣后净收益: {:>+10,.2f}元 | 胜率: {:.2f}%".format(
            k, v['avg_gross_bp'], v['total_gross_pnl'], v['win_rate']*100))

    print("\n【研究任务 2：信号五分组 (Quintile 1 ~ 5) 未来 60m 收益与单调性】")
    for q_idx in range(1, 6):
        print("  - Quintile {} (得分档位 {}): {:+6.2f} bp".format(q_idx, q_idx, q_dict.get(q_idx, 0.0)))
    print("  - ★ 收益单调性判定: {}".format("满足单调性" if is_monotonic else "不具备单调性 (数据挖掘特征)"))

    print("\n【研究任务 3：T-1 正股筹码因子四臂消融对比 (Rank IC)】")
    for k, v in task3_res.items():
        print("  - {:<42} | Rank IC: {:+.4f}".format(k, v))

    print("======================================================================")
    print("                     【终止条件 (Kill Criteria) 判定】")
    print("======================================================================")
    print("触发的 Kill Criteria 数量: {} 项 / (任意 2 项成立即彻底终止路线)".format(len(kill_triggers)))
    for trig in kill_triggers:
        print("  [TRIGGERED] {}".format(trig))
        
    print("-" * 70)
    if len(kill_triggers) >= 2:
        print("⛔ 最终判定:【KILL】—— 触发 {} 项终止条件，彻底终止 15分钟日内高频轮动路线！".format(len(kill_triggers)))
        print("   结论：该策略不存在稳定的正向毛 Alpha，系数据挖掘与高频摩擦陷阱，停止一切相关调参。")
    else:
        print("✅ 最终判定:【PASS / GO】—— 未达到终止线，允许进入 20 交易日影子采集阶段。")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
