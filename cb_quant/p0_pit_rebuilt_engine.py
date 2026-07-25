# -*- coding: utf-8 -*-

"""
机构级重构 P0 级 PIT 审核引擎 (修复元数据字段映射、无默认放行、真实整数张数与完整成本对账)
Rebuilt Institutional P0 PIT Audit Engine (Fixed Metadata Mapping, No Default Pass, Integer Lots & Cost Reconciliation)
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CBP0PITRebuiltEngine:
    def __init__(self, initial_capital=1000000.0, top_n=5,
                 single_slippage=0.0005, single_impact=0.0005, commission=0.00005,
                 max_volume_ratio=0.01):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.single_slippage = single_slippage
        self.single_impact = single_impact
        self.commission = commission
        self.max_volume_ratio = max_volume_ratio
        self.total_one_way_cost = single_slippage + single_impact + commission

    @staticmethod
    def load_strict_pit_metadata(df_panel, data_dir=r"D:\CB_mins_data"):
        """
        1. 修复转股价值与债池过滤：
        - 准确映射 issue_size 为发行/剩余规模 (亿元)；
        - 按 delist_date 判断退市/强赎状态 (已退市不可交易)；
        - 元数据缺失时标记为不可交易 (is_tradable = False)，绝对禁止默认放行！
        """
        df = df_panel.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'], errors='coerce', format='mixed')
        df['date_int'] = df['trade_time'].dt.strftime('%Y%m%d').astype(int)
        
        basic_info_path = os.path.join(data_dir, "cb_basic_info.csv")
        
        if os.path.exists(basic_info_path):
            try:
                basic_info = pd.read_csv(basic_info_path)
                cols_to_merge = ['ts_code']
                if 'issue_size' in basic_info.columns:
                    cols_to_merge.append('issue_size')
                if 'delist_date_clean' in basic_info.columns:
                    cols_to_merge.append('delist_date_clean')
                if 'list_date_clean' in basic_info.columns:
                    cols_to_merge.append('list_date_clean')
                    
                df = df.merge(basic_info[cols_to_merge], on='ts_code', how='left')
            except Exception as e:
                logger.warning(f"读取 PIT 基础元数据文件失败: {e}")

        # 映射字段名与数值类型
        if 'issue_size' in df.columns:
            df['curr_iss_amt'] = pd.to_numeric(df['issue_size'], errors='coerce')
        else:
            df['curr_iss_amt'] = np.nan

        if 'delist_date_clean' in df.columns:
            df['delist_date_clean'] = pd.to_numeric(df['delist_date_clean'], errors='coerce').fillna(20991231)
            # 若当前交易日期 >= 退市/赎回日期，标记为已强赎/退市
            df['is_redeemed'] = df['date_int'] >= df['delist_date_clean']
        else:
            df['is_redeemed'] = False

        if 'list_date_clean' in df.columns:
            df['list_date_clean'] = pd.to_numeric(df['list_date_clean'], errors='coerce').fillna(20000101)
            # 上市不满 30 天禁止交易
            df['is_listed_30d'] = df['date_int'] >= (df['list_date_clean'] + 100) # 简化约30交易日
        else:
            df['is_listed_30d'] = True

        # 2. 严谨交易池判断：必须有完整规模元数据 + 价格<=180 + 规模>=2.0亿 + 非退市/强赎 + 上市>30天
        has_metadata = df['curr_iss_amt'].notnull() & (df['curr_iss_amt'] > 0)
        
        df['is_tradable'] = (
            has_metadata &
            (df['close'] <= 180.0) &
            (df['curr_iss_amt'] >= 2.0) &
            (df['is_redeemed'] == False) &
            (df['is_listed_30d'] == True)
        )
        
        return df

    def run_rebuilt_backtest(self, df_scored):
        """
        3. 纯 $t+1$ 下一根 K 线成交模拟器：
        - 10 张整数倍成手；
        - 成交量 1% 限制参与率；
        - 停牌/零成交/缺失元数据禁止成交；
        - 显式输出买卖双边成本对账单。
        """
        df = df_scored.copy()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.sort_values(by=['ts_code', 'trade_time']).reset_index(drop=True)
        
        time_groups = dict(tuple(df.groupby('trade_time')))
        timestamps = sorted(time_groups.keys())
        
        capital = self.initial_capital
        positions = {}
        pending_orders = []
        
        trade_logs = []
        equity_curve = []
        slot_target_value = capital / self.top_n

        for b_idx, t in enumerate(timestamps):
            df_t = time_groups[t]
            t_dict = df_t.set_index('ts_code').to_dict('index')
            time_str = str(t)[11:16]

            # --- A. 在 $t+1$ Open 执行上一根 bar 下达的委托 ---
            for p_order in pending_orders:
                code = p_order['ts_code']
                action = p_order['action']
                if code not in t_dict:
                    continue
                
                row_t = t_dict[code]
                open_px = row_t['open']
                bar_vol = row_t['vol']
                is_tradable = row_t.get('is_tradable', False)

                # 停牌、零成交或不可交易元数据 -> 绝对禁止成交！
                if open_px <= 0 or bar_vol <= 0 or not is_tradable:
                    continue
                
                # 成交量参与率限制 (最大为当前 15m 成交量的 1%)
                max_allowable_shares = int((bar_vol * self.max_volume_ratio) // 10) * 10
                if max_allowable_shares < 10:
                    continue

                if action == 'BUY':
                    buy_gross_px = open_px
                    buy_spread_cost = buy_gross_px * self.single_slippage
                    buy_impact_cost = buy_gross_px * self.single_impact
                    buy_commission_cost = buy_gross_px * self.commission
                    buy_net_px = buy_gross_px + buy_spread_cost + buy_impact_cost + buy_commission_cost
                    
                    target_alloc = min(slot_target_value, capital * 0.95)
                    calc_shares = target_alloc / buy_net_px
                    
                    # 取 10 张整数倍成手
                    shares = int(calc_shares // 10) * 10
                    shares = min(shares, max_allowable_shares)
                    
                    if shares >= 10 and capital >= shares * buy_net_px:
                        capital -= shares * buy_net_px
                        positions[code] = {
                            'shares': shares,
                            'entry_gross_px': buy_gross_px,
                            'entry_net_px': buy_net_px,
                            'buy_spread': shares * buy_spread_cost,
                            'buy_impact': shares * buy_impact_cost,
                            'buy_commission': shares * buy_commission_cost,
                            'entry_time': str(t),
                            'bars_held': 0
                        }
                        trade_logs.append({
                            'trade_time': str(t), 'ts_code': code, 'side': 'BUY', 'action': 'BUY_NEXT_OPEN',
                            'gross_price': buy_gross_px, 'net_price': buy_net_px, 'shares': shares,
                            'gross_pnl': 0.0, 'net_pnl': 0.0,
                            'spread_cost': shares * buy_spread_cost,
                            'impact_cost': shares * buy_impact_cost,
                            'commission_cost': shares * buy_commission_cost
                        })
                        
                elif action == 'SELL':
                    if code in positions:
                        pos = positions[code]
                        shares = pos['shares']
                        shares = min(shares, max_allowable_shares)
                        
                        sell_gross_px = open_px
                        sell_spread_cost = sell_gross_px * self.single_slippage
                        sell_impact_cost = sell_gross_px * self.single_impact # 修复变量名
                        sell_commission_cost = sell_gross_px * self.commission
                        sell_net_px = sell_gross_px - sell_spread_cost - sell_impact_cost - sell_commission_cost
                        
                        capital += shares * sell_net_px
                        
                        gross_pnl = shares * (sell_gross_px - pos['entry_gross_px'])
                        net_pnl = (shares * sell_net_px) - (shares * pos['entry_net_px'])
                        
                        trade_logs.append({
                            'trade_time': str(t), 'ts_code': code, 'side': 'SELL', 'action': p_order['reason'],
                            'gross_price': sell_gross_px, 'net_price': sell_net_px, 'shares': shares,
                            'gross_pnl': gross_pnl, 'net_pnl': net_pnl,
                            'spread_cost': shares * sell_spread_cost + pos['buy_spread'],
                            'impact_cost': shares * sell_impact_cost + pos['buy_impact'],
                            'commission_cost': shares * sell_commission_cost + pos['buy_commission']
                        })
                        del positions[code]

            pending_orders = []

            # --- B. 评估当前时刻 NAV ---
            total_pos_val = 0.0
            for code, pos in positions.items():
                pos['bars_held'] += 1
                if code in t_dict:
                    total_pos_val += pos['shares'] * t_dict[code]['close']
                else:
                    total_pos_val += pos['shares'] * pos['entry_gross_px']

            nav = capital + total_pos_val
            equity_curve.append({
                'trade_time': str(t),
                'nav': nav,
                'cash': capital,
                'position_val': total_pos_val,
                'num_positions': len(positions)
            })
            slot_target_value = nav / self.top_n

            # --- C. 当前 bar t 结束生成信号 ---
            for code, pos in positions.items():
                if code not in t_dict:
                    continue
                row = t_dict[code]
                rank_15m = row.get('rank_15m', 999)
                curr_ret = (row['close'] - pos['entry_gross_px']) / pos['entry_gross_px']

                if curr_ret <= -0.010:
                    pending_orders.append({'ts_code': code, 'action': 'SELL', 'reason': 'REBUILT_STOP_LOSS_-1%'})
                elif rank_15m > 15:
                    pending_orders.append({'ts_code': code, 'action': 'SELL', 'reason': 'REBUILT_RANK_EXIT'})
                elif pos['bars_held'] >= 6 or time_str >= '14:30':
                    pending_orders.append({'ts_code': code, 'action': 'SELL', 'reason': 'REBUILT_TIME_EXIT'})

            open_slots = self.top_n - len(positions) - len([p for p in pending_orders if p['action'] == 'BUY'])
            if open_slots > 0 and capital > 10000.0:
                candidates = []
                for code, row in t_dict.items():
                    rank_15m = row.get('rank_15m', 999)
                    if row.get('is_tradable', False) and rank_15m <= self.top_n and code not in positions:
                        candidates.append((code, rank_15m))

                candidates.sort(key=lambda x: x[1])
                for code, r_val in candidates[:open_slots]:
                    pending_orders.append({'ts_code': code, 'action': 'BUY', 'reason': 'REBUILT_BUY_SIGNAL'})

        return pd.DataFrame(equity_curve), pd.DataFrame(trade_logs)
