# -*- coding: utf-8 -*-

"""
基于 D:\iquant_data\data_v2 真实数据的多维分组单调性、真实筹码 4 臂消融与 P3/P4 引擎
Real Data-Driven Multidimensional Grouping, Real Chip Ablation, & P3/P4 Research Engine
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBRealDataV2ResearchEngine:
    @staticmethod
    def run_multidimensional_grouping(df_scored):
        """
        研究任务 2：使用真实数据进行 6 维分组与单调性分析
        1. 信号强度 5 分组 (Q1~Q5)
        2. 高低流动性 (Turnover Rate)
        3. 高低转股溢价率 (PIT Premium Rate)
        4. 上午 vs 下午交易时段
        5. 不同持有期 (15m, 30m, 60m, 120m)
        6. 正股-转债 15m 收益背离度 (Stock-Bond Divergence)
        """
        df = df_scored.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)
        
        # 计算未来不同持仓期的收益率
        df['fut_ret_15m'] = df.groupby('ts_code')['close'].shift(-1) / df['close'] - 1.0
        df['fut_ret_30m'] = df.groupby('ts_code')['close'].shift(-2) / df['close'] - 1.0
        df['fut_ret_60m'] = df.groupby('ts_code')['close'].shift(-4) / df['close'] - 1.0
        df['fut_ret_120m'] = df.groupby('ts_code')['close'].shift(-8) / df['close'] - 1.0
        
        clean = df.dropna(subset=['score_15m', 'fut_ret_60m']).copy()
        
        # A. 信号强度 5 分组 (Q1 ~ Q5)
        clean['score_quintile'] = clean.groupby('trade_time')['score_15m'].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') + 1 if len(x) >= 5 else np.nan
        )
        q_returns = clean.groupby('score_quintile')['fut_ret_60m'].mean() * 10000.0 # bp
        
        # B. 高低转股溢价率分组 (中位数切分)
        if 'premium_rate' in clean.columns and clean['premium_rate'].notnull().any():
            med_prem = clean['premium_rate'].median()
            clean['prem_group'] = np.where(clean['premium_rate'] >= med_prem, 'High_Premium', 'Low_Premium')
            prem_res = clean.groupby(['prem_group', 'score_quintile'])['fut_ret_60m'].mean() * 10000.0
        else:
            prem_res = pd.Series()

        # C. 上午 vs 下午分组
        clean['time_str'] = clean['trade_time'].dt.strftime('%H:%M')
        clean['session'] = clean['time_str'].apply(lambda x: 'Morning' if x <= '11:30' else 'Afternoon')
        session_res = clean.groupby(['session', 'score_quintile'])['fut_ret_60m'].mean() * 10000.0

        # D. 持有期对比 (15m, 30m, 60m, 120m)
        hold_periods = {
            '15m': clean.groupby('score_quintile')['fut_ret_15m'].mean() * 10000.0,
            '30m': clean.groupby('score_quintile')['fut_ret_30m'].mean() * 10000.0,
            '60m': clean.groupby('score_quintile')['fut_ret_60m'].mean() * 10000.0,
            '120m': clean.groupby('score_quintile')['fut_ret_120m'].mean() * 10000.0,
        }

        # E. 正股-转债 15m 收益背离度 (P3 微观结构代理)
        if 'stk_close' in clean.columns and clean['stk_close'].notnull().any():
            clean['stk_ret_15m'] = clean.groupby('stk_code')['stk_close'].pct_change()
            clean['ret_divergence'] = clean['stk_ret_15m'] - clean['ret_15m']
            med_div = clean['ret_divergence'].median()
            clean['divergence_group'] = np.where(clean['ret_divergence'] >= med_div, 'Stock_Outperform', 'Bond_Outperform')
            div_res = clean.groupby(['divergence_group', 'score_quintile'])['fut_ret_60m'].mean() * 10000.0
        else:
            div_res = pd.Series()

        return {
            'quintile_returns_bp': q_returns.to_dict(),
            'premium_group_returns': prem_res.to_dict() if not prem_res.empty else {},
            'session_returns': session_res.to_dict(),
            'hold_period_returns': {k: v.to_dict() for k, v in hold_periods.items()},
            'divergence_returns': div_res.to_dict() if not div_res.empty else {}
        }

    @staticmethod
    def run_real_t1_chip_ablation(df_scored):
        """
        研究任务 3：使用 D:\iquant_data\data_v2\cyq1 真实正股筹码进行 4 臂消融对比
        Arm 1: 纯 15m 基线 (Baseline)
        Arm 2: 基线 + 真实正股筹码获利盘比例 (Baseline + Real Chip Winner Rate)
        Arm 3: 基线 + 真实正股筹码成本比率 (Baseline + Real Chip Cost Ratio: stk_close / chip_weight_avg)
        Arm 4: 筹码随机打乱 (Placebo)
        """
        df = df_scored.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        
        # 填充缺省值
        df['chip_winner_rate'] = pd.to_numeric(df.get('chip_winner_rate', np.nan), errors='coerce').fillna(50.0)
        df['chip_weight_avg'] = pd.to_numeric(df.get('chip_weight_avg', np.nan), errors='coerce')
        
        # 筹码相对成本比率: 正股收盘价 / 筹码均价
        if 'stk_close' in df.columns:
            df['chip_cost_ratio'] = df['stk_close'] / (df['chip_weight_avg'] + 1e-8)
            df['chip_cost_ratio'] = df['chip_cost_ratio'].fillna(1.0)
        else:
            df['chip_cost_ratio'] = 1.0

        # Placebo 随机打乱真实筹码数据
        np.random.seed(42)
        df['chip_winner_placebo'] = np.random.permutation(df['chip_winner_rate'].values)

        # 构建 4 臂得分
        df['score_base'] = df['score_15m']
        df['score_chip_winner'] = df['score_15m'] + 0.2 * (df['chip_winner_rate'] / 100.0)
        df['score_chip_cost'] = df['score_15m'] + 0.2 * (df['chip_cost_ratio'] - 1.0)
        df['score_chip_placebo'] = df['score_15m'] + 0.2 * (df['chip_winner_placebo'] / 100.0)

        df['fut_ret_60m'] = df.groupby('ts_code')['close'].shift(-4) / df['close'] - 1.0
        clean = df.dropna(subset=['fut_ret_60m']).copy()

        def calc_ic(score_col):
            def c_ic(g):
                if len(g) < 5 or g[score_col].std() == 0:
                    return np.nan
                return g[score_col].corr(g['fut_ret_60m'], method='spearman')
            ic_series = clean.groupby('trade_time').apply(c_ic).dropna()
            return ic_series.mean() if len(ic_series) > 0 else 0.0

        ic_base = calc_ic('score_base')
        ic_winner = calc_ic('score_chip_winner')
        ic_cost = calc_ic('score_chip_cost')
        ic_placebo = calc_ic('score_chip_placebo')

        return {
            'Arm 1: Pure 15m Baseline Rank IC': ic_base,
            'Arm 2: Baseline + Real Chip Winner Rate Rank IC': ic_winner,
            'Arm 3: Baseline + Real Chip Cost Ratio Rank IC': ic_cost,
            'Arm 4: Chip Placebo (Randomized) Rank IC': ic_placebo,
            'Real Chip Winner IC Increment': ic_winner - ic_base,
            'Real Chip Cost Ratio IC Increment': ic_cost - ic_base
        }
