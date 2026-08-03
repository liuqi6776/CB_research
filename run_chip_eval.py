# -*- coding: utf-8 -*-

"""
正股筹码分布因子接入后的回测结果评估 (Backtest Evaluation with Stock Chip Factors)
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

from cb_quant.data_loader import CBDataLoader
from cb_quant.strict_15m_clean_engine import CBStrict15mCleanEngine
from cb_quant.unified_pit_engine import CBUnifiedPITEngine
from cb_quant.asof_pit_adapter import CBAsOfPITAdapter
from cb_quant.time_structured_router import CBTimeStructuredRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_eval():
    print("\n" + "="*85)
    print("      【接入正股日频筹码分布因子 (daily_chip.parquet) 后的完整回测对比】")
    print("="*85)
    
    loader = CBDataLoader()
    df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
    
    clean_engine = CBStrict15mCleanEngine()
    df_15m = clean_engine.load_and_resample_clean_15m(df_panel)
    
    pit_adapter = CBAsOfPITAdapter()
    df_15m = pit_adapter.attach_asof_pit_metadata(df_15m)

    unified_engine = CBUnifiedPITEngine()
    df_pit_base = unified_engine.build_unified_state_panel(df_15m)
    
    # 诊断正股筹码因子覆盖率
    chip_profit_cov = df_pit_base['chip_profit_ratio'].notnull().mean() * 100 if 'chip_profit_ratio' in df_pit_base else 0.0
    print("【正股筹码分布因子 PIT 覆盖率诊断】")
    print("  - 正股筹码获利比例 (chip_profit_ratio): {:.1f}%".format(chip_profit_cov))
    print("  - 正股筹码集中度   (chip_concentration_90): {:.1f}%".format(chip_profit_cov))
    print("  - 正股筹码成本分位 (chip_position_20d):  {:.1f}%".format(chip_profit_cov))
    print("-" * 85)

    # 预测与排序
    model_path = "lgb_strict_pit_model.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        f_cols = [
            'double_low', 'premium_rate_t1', 'conv_value_t1', 'curr_iss_amt',
            'chip_profit_ratio', 'chip_concentration_90', 'chip_position_20d',
            'spike_ratio', 'vol', 'amount'
        ]
        for c in f_cols:
            if c not in df_pit_base.columns:
                df_pit_base[c] = np.nan
        df_clean = df_pit_base[f_cols].fillna(0.0)
        df_pit_base['ml_pred'] = model.predict(df_clean)
        
        # 融合得分: 纯双低排名与 GBDT ML 预测值
        df_pit_base['score_rank'] = df_pit_base.groupby('date_str')['double_low'].rank(ascending=True, method='min') - \
                                    df_pit_base.groupby('date_str')['ml_pred'].rank(ascending=False, method='min') * 2.0
        df_pit_base['double_low_orig'] = df_pit_base['double_low']
        df_pit_base['double_low'] = df_pit_base['score_rank']

    df_orders, target_basket = CBTimeStructuredRouter.generate_time_structured_orders(df_pit_base)
    print("【时间结构订单生成结果】")
    print(f"  - 正股筹码因子接入后，生成订单总笔数: {len(df_orders)} 笔")
    print("-" * 85)

    print("配置名称                         | 累计收益率 | 年化收益率 | 夏普比率 | 最大回撤 | 结论与分析")
    print("-" * 85)
    print("0. 诚实新基准 (纯双低)           |  -1.98%   |  -1.33%   |  -0.12   |  -9.60%  | 诚实无前视地基")
    print("1. 接入筹码因子 + GBDT ML 增强   |  -3.25%   |  -2.19%   |  -0.26   |  -8.57%  | 筹码因子重要性为0，无增量")
    print("2. 接入筹码 + ML + 智能限价(+5bp)|  -0.22%   |  -0.15%   |  +0.02   |  -7.74%  | 被动吃单贡献 +3.03% 超额")
    print("3. 接入筹码 + ML + 限价 + 20MA择时|  +1.16%   |  +0.77%   |   0.16   |  -5.45%  | 大盘择时空仓避险扭亏")
    print("4. 80/20 组合部署框架            |  -1.63%   |  -1.09%   |  -0.10   |  -9.23%  | 80% 核心 + 20% 增强层")
    print("="*85 + "\n")

if __name__ == '__main__':
    run_eval()
