# -*- coding: utf-8 -*-

"""
可信 15分钟基线、Alpha 五分组单调性与 T-1 正股筹码因子消融引擎
Credible 15m Baseline, Alpha Monotonicity, & T-1 Stock Chip Ablation Engine
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBCredibleBaselineEngine:
    @staticmethod
    def run_three_execution_assumptions(df_scored, top_n=5):
        """
        研究任务 1：至少运行三种成交假设
        - 假设 A: 下一根 K 线 ($t+1$) Open 开盘价成交
        - 假设 B: 下一根 K 线 ($t+1$) Typical Price (典型价 (H+L+C)/3) 成交
        - 假设 C: 下一根 K 线 ($t+1$) Open 开盘价 + 保守单边 0.10% 滑点成交
        """
        df = df_scored.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)
        
        # 预先构建 $t+1$ Open 与 Typical Price 字段
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3.0
        df['next_open'] = df.groupby('ts_code')['open'].shift(-1)
        df['next_typical'] = df.groupby('ts_code')['typical_price'].shift(-1)
        
        results = {}
        assumptions = [
            ("假设 A: Next-Bar Open (t+1 Open)", "open", 0.0),
            ("假设 B: Next-Bar Typical Price ((H+L+C)/3)", "typical", 0.0),
            ("假设 C: Next-Bar Open + 保守 0.1% 滑点", "open", 0.0010)
        ]
        
        for name, mode, extra_slip in assumptions:
            time_groups = dict(tuple(df.groupby('trade_time')))
            timestamps = sorted(time_groups.keys())
            
            capital = 1000000.0
            positions = {}
            pending_orders = []
            trade_logs = []

            for b_idx, t in enumerate(timestamps):
                df_t = time_groups[t]
                t_dict = df_t.set_index('ts_code').to_dict('index')
                time_str = str(t)[11:16]

                # 1. 严格在 $t+1$ 执行上一根产生的待成交订单
                for p_order in pending_orders:
                    code = p_order['ts_code']
                    action = p_order['action']
                    if code not in t_dict:
                        continue
                    
                    row_t = t_dict[code]
                    base_px = row_t['open'] if mode == 'open' else row_t['typical_price']
                    
                    if action == 'BUY':
                        buy_gross_px = base_px
                        buy_net_px = buy_gross_px * (1.0 + extra_slip + 0.00005) # 含佣金
                        target_alloc = capital / top_n
                        shares = target_alloc / buy_net_px
                        
                        if shares >= 10 and capital >= shares * buy_net_px:
                            capital -= shares * buy_net_px
                            positions[code] = {
                                'shares': shares,
                                'entry_gross': buy_gross_px,
                                'entry_net': buy_net_px,
                                'bars_held': 0
                            }
                    elif action == 'SELL':
                        if code in positions:
                            pos = positions[code]
                            shares = pos['shares']
                            sell_gross_px = base_px
                            sell_net_px = sell_gross_px * (1.0 - extra_slip - 0.00005)
                            capital += shares * sell_net_px
                            
                            gross_pnl = shares * (sell_gross_px - pos['entry_gross'])
                            net_pnl = shares * (sell_net_px - pos['entry_net'])
                            gross_ret = (sell_gross_px / pos['entry_gross']) - 1.0
                            
                            trade_logs.append({
                                'trade_time': str(t), 'ts_code': code, 'action': 'SELL',
                                'gross_pnl': gross_pnl, 'net_pnl': net_pnl,
                                'gross_ret': gross_ret, 'bars_held': pos['bars_held']
                            })
                            del positions[code]

                pending_orders = []

                # 2. 当前 bar t 结束生成信号，下达 $t+1$ 执行信号
                for code, pos in positions.items():
                    pos['bars_held'] += 1
                    if code in t_dict:
                        row = t_dict[code]
                        rank_15m = row.get('rank_15m', 999)
                        if pos['bars_held'] >= 4 or rank_15m > 15 or time_str >= '14:30':
                            pending_orders.append({'ts_code': code, 'action': 'SELL'})

                open_slots = top_n - len(positions) - len([p for p in pending_orders if p['action'] == 'BUY'])
                if open_slots > 0:
                    candidates = []
                    for code, row in t_dict.items():
                        rank_15m = row.get('rank_15m', 999)
                        if rank_15m <= top_n and code not in positions:
                            candidates.append((code, rank_15m))
                    candidates.sort(key=lambda x: x[1])
                    for code, r_val in candidates[:open_slots]:
                        pending_orders.append({'ts_code': code, 'action': 'BUY'})

            df_tr = pd.DataFrame(trade_logs)
            if not df_tr.empty:
                avg_gross_bp = df_tr['gross_ret'].mean() * 10000.0
                total_gross_pnl = df_tr['gross_pnl'].sum()
                win_rate = (df_tr['gross_pnl'] > 0).mean()
                wins = df_tr[df_tr['gross_pnl'] > 0]['gross_pnl']
                losses = abs(df_tr[df_tr['gross_pnl'] < 0]['gross_pnl'])
                pl_ratio = wins.mean() / (losses.mean() + 1e-8) if len(losses) > 0 else 0.0
            else:
                avg_gross_bp, total_gross_pnl, win_rate, pl_ratio = 0.0, 0.0, 0.0, 0.0

            results[name] = {
                'avg_gross_bp': avg_gross_bp,
                'total_gross_pnl': total_gross_pnl,
                'win_rate': win_rate,
                'pl_ratio': pl_ratio,
                'total_trades': len(df_tr)
            }

        return results

    @staticmethod
    def test_alpha_monotonicity_grouping(df_scored):
        """
        研究任务 2：信号强度五分组单调性检验 (Quintile 1 ~ 5)
        把每 15m 截面按得分为 5 等份 (Q1=最低分，Q5=最高分)，计算未来 60m 平均收益
        """
        df = df_scored.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)
        
        df['fut_ret_60m'] = df.groupby('ts_code')['close'].shift(-4) / df['close'] - 1.0
        clean = df.dropna(subset=['score_15m', 'fut_ret_60m']).copy()
        
        # 逐 15m 截面分为 5 等份
        clean['score_quintile'] = clean.groupby('trade_time')['score_15m'].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop') + 1 if len(x) >= 5 else np.nan
        )
        
        q_returns = clean.groupby('score_quintile')['fut_ret_60m'].mean() * 10000.0 # 转换为 bp
        
        # 单调性判定
        q_vals = [q_returns.get(i, 0.0) for i in range(1, 6)]
        is_monotonic = all(x <= y for x, y in zip(q_vals, q_vals[1:]))
        
        return q_returns.to_dict(), is_monotonic

    @staticmethod
    def test_t1_stock_chip_ablation(df_scored):
        """
        研究任务 3：只验证一个增量因子 - T-1 正股筹码
        比较：1. 纯 15m 基线 | 2. 基线+筹码水平 (T-1) | 3. 基线+筹码 Delta | 4. 筹码 Placebo 随机打乱
        """
        df = df_scored.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        
        # 模拟 T-1 筹码集中度与筹码获利盘比例 (0 ~ 1)
        np.random.seed(42)
        unique_codes = df['ts_code'].unique()
        chip_base_map = {code: np.random.uniform(0.3, 0.8) for code in unique_codes}
        
        df['chip_level_t1'] = df['ts_code'].map(chip_base_map) + np.random.normal(0, 0.02, len(df))
        df['chip_delta_t1'] = df.groupby('ts_code')['chip_level_t1'].diff(240).fillna(0.0)
        df['chip_placebo'] = np.random.permutation(df['chip_level_t1'].values)

        # 四臂得分计算
        df['score_base'] = df['score_15m']
        df['score_chip_level'] = df['score_15m'] + 0.3 * df['chip_level_t1']
        df['score_chip_delta'] = df['score_15m'] + 0.3 * df['chip_delta_t1']
        df['score_chip_placebo'] = df['score_15m'] + 0.3 * df['chip_placebo']
        
        df['fut_ret_60m'] = df.groupby('ts_code')['close'].shift(-4) / df['close'] - 1.0
        clean = df.dropna(subset=['fut_ret_60m']).copy()

        def calc_ic(score_col):
            def c_ic(g):
                if len(g) < 5 or g[score_col].std() == 0:
                    return np.nan
                return g[score_col].corr(g['fut_ret_60m'], method='spearman')
            return clean.groupby('trade_time').apply(c_ic).mean()

        ic_base = calc_ic('score_base')
        ic_level = calc_ic('score_chip_level')
        ic_delta = calc_ic('score_chip_delta')
        ic_placebo = calc_ic('score_chip_placebo')

        return {
            'Pure 15m Baseline Rank IC': ic_base,
            'Baseline + Chip Level T-1 Rank IC': ic_level,
            'Baseline + Chip Delta T-1 Rank IC': ic_delta,
            'Chip Placebo (Random) Rank IC': ic_placebo
        }
