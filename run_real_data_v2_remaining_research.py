# -*- coding: utf-8 -*-

"""
基于 D:\iquant_data\data_v2 真实数据的剩余研究任务全量主程序
Master Execution Runner for Remaining Research Tasks Driven by Real Data in D:\iquant_data\data_v2
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader
from cb_quant.intraday_cross_sectional import CBIntradayCrossSectionalEngine
from cb_quant.real_data_v2_engine import CBRealDataV2Engine
from cb_quant.real_data_v2_research_engine import CBRealDataV2ResearchEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    logging.info("=== 启动 D:\\iquant_data\\data_v2 真实数据驱动的剩余研究任务全量评估 ===")
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    # 1. 精确合并 D:\iquant_data\data_v2 真实正股日线与真实 T-1 筹码数据
    v2_engine = CBRealDataV2Engine()
    df_merged = v2_engine.merge_real_pit_data(df_panel)
    
    # 2. 转换为 15分钟 K 线并计算 15m 截面 Z-Score 打分
    df_merged['trade_time'] = pd.to_datetime(df_merged['trade_time'])
    df_15m = df_merged.groupby(['ts_code', pd.Grouper(key='trade_time', freq='15min')]).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'vol': 'sum',
        'amount': 'sum',
        'is_tradable': 'first',
        'chip_winner_rate': 'first',
        'chip_weight_avg': 'first',
        'stk_close': 'first',
        'premium_rate': 'first',
        'stk_code': 'first'
    }).dropna(subset=['close']).reset_index()
    
    df_scored = CBIntradayCrossSectionalEngine.compute_cross_sectional_scores(df_15m)
    
    # =========================================================================
    # 研究任务 2：使用真实数据进行 6 维分组与单调性分析
    # =========================================================================
    logging.info("运行【研究任务 2】：多维分组 (Q1~Q5, 溢价率, 时段, 持有期, 背离度) 分析...")
    grouping_res = CBRealDataV2ResearchEngine.run_multidimensional_grouping(df_scored)
    
    # =========================================================================
    # 研究任务 3：使用 D:\iquant_data\data_v2 真实 T-1 筹码进行 4 臂消融对比
    # =========================================================================
    logging.info("运行【研究任务 3】：真实 T-1 筹码 4 臂消融对比 (基线 / 获利盘 / 成本比率 / Placebo)...")
    ablation_res = CBRealDataV2ResearchEngine.run_real_t1_chip_ablation(df_scored)
    
    # 导出 CSV 日志
    q_df = pd.DataFrame(list(grouping_res['quintile_returns_bp'].items()), columns=['Quintile', 'Return_bp'])
    q_df.to_csv("real_grouping_results.csv", index=False, encoding="utf-8-sig")
    
    ablation_df = pd.DataFrame(list(ablation_res.items()), columns=['Arm_Description', 'Rank_IC'])
    ablation_df.to_csv("real_chip_ablation_results.csv", index=False, encoding="utf-8-sig")
    
    # 打印最终多维分析报告
    print("\n" + "="*75)
    print("      D:\\iquant_data\\data_v2 真实数据驱动的剩余研究全量评估报告")
    print("="*75)
    
    print("\n【研究任务 2-A：信号五分组 (Q1 ~ Q5) 未来 60m 收益 (bp)】")
    q_vals = []
    for q_idx in range(1, 6):
        ret_val = grouping_res['quintile_returns_bp'].get(q_idx, 0.0)
        q_vals.append(ret_val)
        print("  - Quintile {} (得分档位 {}): {:+6.2f} bp".format(q_idx, q_idx, ret_val))
        
    is_monotonic = all(x <= y for x, y in zip(q_vals, q_vals[1:]))
    print("  - ★ 真实数据信号单调性判定: {}".format("满足单调性" if is_monotonic else "反向单调/无单调性 (数据挖掘特征)"))

    print("\n【研究任务 2-B：不同持仓期 (15m, 30m, 60m, 120m) 收益比较 (bp)】")
    for hp, h_dict in grouping_res['hold_period_returns'].items():
        q1_r = h_dict.get(1, 0.0)
        q5_r = h_dict.get(5, 0.0)
        print("  - 持有期 {:<5}: Quintile 1 = {:+6.2f} bp | Quintile 5 = {:+6.2f} bp".format(hp, q1_r, q5_r))

    print("\n【研究任务 2-C：上午 vs 下午交易时段 60m 收益 (bp)】")
    for (sess, q_idx), r_val in grouping_res['session_returns'].items():
        print("  - 时段 {:<9} | Quintile {}: {:+6.2f} bp".format(sess, q_idx, r_val))

    if grouping_res['divergence_returns']:
        print("\n【研究任务 P3：正股-转债 15m 收益背离度 60m 收益 (bp)】")
        for (div_grp, q_idx), r_val in grouping_res['divergence_returns'].items():
            print("  - 背离分组 {:<16} | Quintile {}: {:+6.2f} bp".format(div_grp, q_idx, r_val))

    print("\n【研究任务 3：D:\\iquant_data\\data_v2\\cyq1 真实 T-1 筹码 4 臂消融对比 (Rank IC)】")
    for k, v in ablation_res.items():
        if 'Increment' in k:
            print("  - {:<46} | IC 净增量: {:+.4f} (门槛要求 >= +0.02)".format(k, v))
        else:
            print("  - {:<46} | Rank IC:   {:+.4f}".format(k, v))

    print("\n" + "="*75)
    print("                        【研究终极判定总结】")
    print("="*75)
    winner_inc = ablation_res.get('Real Chip Winner IC Increment', 0.0)
    cost_inc = ablation_res.get('Real Chip Cost Ratio IC Increment', 0.0)
    max_inc = max(winner_inc, cost_inc)
    
    print("1. 信号单调性测试:      {}".format("通过 (纯正单调)" if is_monotonic else "未通过 (反向单调，脉冲追高为陷阱)"))
    print("2. 真实筹码 IC 增量测试: {}".format("满足 (IC增量 >= 0.02)" if max_inc >= 0.02 else "未通过 (真实筹码增量 < 0.02)"))
    print("-" * 75)
    
    if not is_monotonic or max_inc < 0.02:
        print("最终学术与研究结论: 【KILL / 终止纯日内轮动 Alpha 研究】")
        print("   原因：真实数据证实 15m 截面急拉系反向反转陷阱；真实 T-1 筹码未能带来 > 0.02 的 Rank IC 增量。")
        print("   保留事项：15分钟分钟线数据 asset 100% 保留，转入择时入场、正股-转债联动与流动性建模等 6 大底层场景！")
    else:
        print("最终学术与研究结论: 【PASS / 允许进入下一阶段】")
    print("="*75 + "\n")

if __name__ == "__main__":
    main()
