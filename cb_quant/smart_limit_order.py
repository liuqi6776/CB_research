# -*- coding: utf-8 -*-

"""
智能限价挂单算法模块 (Smart Limit Order Execution Module for QMT)
核心动作：被动挂单吃价差，降低买卖滑点与交易成本，超时自动撤单改市价扫货。
"""

import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

class SmartLimitOrderManager:
    """
    QMT 智能限价单执行管理器
    """
    def __init__(self, account_id="MOCK_ACCOUNT", is_simulation=True):
        self.account_id = account_id
        self.is_simulation = is_simulation
        self.orders = {}
        self.order_counter = 1000

    def get_full_tick(self, security):
        """
        获取实时盘口五档 (QMT 原生接口适配或模拟盘口)
        """
        try:
            import xtquant.xtdata as xtdata
            tick = xtdata.get_full_tick([security]).get(security, {})
            if tick:
                return {
                    'bid1': tick.get('bidPrice', [0])[0],
                    'ask1': tick.get('askPrice', [0])[0],
                    'bid_vol1': tick.get('bidVol', [0])[0],
                    'ask_vol1': tick.get('askVol', [0])[0],
                    'last_price': tick.get('lastPrice', 0)
                }
        except Exception:
            pass

        return {
            'bid1': 100.00,
            'ask1': 100.02,
            'bid_vol1': 500,
            'ask_vol1': 500,
            'last_price': 100.01
        }

    def passorder(self, action, order_type, account, security, price, volume, remark="", quick_type=2):
        """
        QMT passorder 下单原生 API 适配
        action: 23=买入, 24=卖出
        order_type: 1101=单账号限价
        """
        if not self.is_simulation:
            try:
                from xtquant.xttrader import passorder
                return passorder(action, order_type, account, security, price, volume, remark, quick_type)
            except Exception as e:
                logging.error(f"[QMT Passorder Error] {security} 下单失败: {e}")
                return None
        
        self.order_counter += 1
        order_id = f"ORD_{self.order_counter}"
        self.orders[order_id] = {
            'security': security,
            'action': action,
            'price': price,
            'target_volume': volume,
            'filled_volume': volume,
            'status': 'FILLED',
            'create_time': time.time()
        }
        logging.info(f"[SmartLimitOrder] 模拟下单 {order_id}: {security} {'BUY' if action==23 else 'SELL'} {volume}张 @ {price:.3f}")
        return order_id

    def get_order_detail(self, order_id):
        if order_id in self.orders:
            return self.orders[order_id]
        return {'filled_volume': 0, 'status': 'UNKNOWN'}

    def cancel_order(self, order_id):
        if not self.is_simulation:
            try:
                from xtquant.xttrader import cancel_order
                cancel_order(order_id)
            except Exception as e:
                logging.error(f"[Cancel Error] {order_id} 撤单失败: {e}")
        else:
            if order_id in self.orders:
                self.orders[order_id]['status'] = 'CANCELLED'
                logging.info(f"[SmartLimitOrder] 模拟撤单成功: {order_id}")

    def execute_smart_limit_order(self, security, direction, target_volume, timeout=30, tick_precision=0.001):
        """
        智能限价单核心算法逻辑:
        1. 根据方向计算买一价 + 0.001 (买) 或 卖一价 - 0.001 (卖) 的被动挂单价；
        2. 挂单后进入 30 秒超时监控循环；
        3. 若超时仍未全部成交，撤销剩余订单并用市价/第一档吃单扫尾。
        """
        target_volume = int(target_volume)
        if target_volume <= 0:
            return None

        tick = self.get_full_tick(security)
        bid1 = tick['bid1']
        ask1 = tick['ask1']

        if direction.lower() == 'buy':
            limit_price = bid1 + tick_precision if bid1 > 0 else tick['last_price']
            action = 23
        else:
            limit_price = ask1 - tick_precision if ask1 > 0 else tick['last_price']
            action = 24

        logging.info(f"[SmartLimitOrder] 启动挂单 {security} | 方向: {direction.upper()} | 目标量: {target_volume}张 | 被动限价: {limit_price:.3f}")
        
        order_id = self.passorder(action, 1101, self.account_id, security, limit_price, target_volume, 'SMART_LIMIT', 2)
        if not order_id:
            return None

        start_time = time.time()
        filled_vol = 0
        
        while time.time() - start_time < timeout:
            order_status = self.get_order_detail(order_id)
            filled_vol = order_status.get('filled_volume', 0)
            
            if filled_vol >= target_volume:
                logging.info(f"[SmartLimitOrder] 完美限价成交! {security} 全部成交 {filled_vol}/{target_volume}张 @ {limit_price:.3f}")
                return {
                    'security': security, 'direction': direction,
                    'filled_volume': filled_vol, 'avg_price': limit_price,
                    'execution_type': 'PASSIVE_LIMIT'
                }
            
            time.sleep(0.5)

        unfilled_vol = target_volume - filled_vol
        logging.warning(f"[SmartLimitOrder] 超时 {timeout}s! {security} 已成交 {filled_vol}/{target_volume}张，撤单并扫尾剩 {unfilled_vol}张")
        
        self.cancel_order(order_id)
        
        latest_tick = self.get_full_tick(security)
        if direction.lower() == 'buy':
            sweep_price = latest_tick['ask1'] if latest_tick['ask1'] > 0 else limit_price
        else:
            sweep_price = latest_tick['bid1'] if latest_tick['bid1'] > 0 else limit_price
            
        sweep_order_id = self.passorder(action, 1101, self.account_id, security, sweep_price, unfilled_vol, 'SWEEP_MARKET', 2)
        
        avg_price = (filled_vol * limit_price + unfilled_vol * sweep_price) / target_volume
        return {
            'security': security, 'direction': direction,
            'filled_volume': target_volume, 'avg_price': avg_price,
            'execution_type': 'SWEEP_FALLBACK'
        }
