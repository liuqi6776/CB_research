# -*- coding: utf-8 -*-

"""
QMT 实盘/模拟盘全自动运行主脚本 (Production QMT Live Runner)
功能：
1. 盘前自动调用 daily_chip_factor.py 更新正股筹码分布；
2. 盘中 15m 触发信号计算与三级统一资格状态机过滤；
3. 调用 SmartLimitOrderManager 被动挂单吃价差 (降低滑点与交易成本)；
4. 包含完整的交易日志、撤单追单与风险隔离系统。
"""

import os
import sys
import time
import logging
import datetime
import pandas as pd
import numpy as np

from cb_quant.data_loader import CBDataLoader
from cb_quant.strict_15m_clean_engine import CBStrict15mCleanEngine
from cb_quant.unified_pit_engine import CBUnifiedPITEngine
from cb_quant.time_structured_router import CBTimeStructuredRouter
from cb_quant.smart_limit_order import SmartLimitOrderManager
from cb_quant.daily_chip_factor import DailyChipFactorEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("qmt_live_runner.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

class QMTLiveRunner:
    """
    QMT 实盘/模拟盘策略总控引擎
    """
    def __init__(self, account_id="MOCK_ACCOUNT_8888", is_simulation=True, capital=2000000.0):
        self.account_id = account_id
        self.is_simulation = is_simulation
        self.capital = capital
        self.smart_manager = SmartLimitOrderManager(account_id=account_id, is_simulation=is_simulation)
        self.positions = {}

    def run_pre_market_job(self):
        """
        盘前 09:10 任务：更新正股筹码分布因子 (daily_chip.parquet)
        """
        logging.info("=== 触发盘前定时任务：生成正股筹码与成本分布因子 ===")
        try:
            chip_engine = DailyChipFactorEngine()
            chip_engine.generate_daily_chip_factors(output_path="daily_chip.parquet")
            logging.info("盘前筹码因子更新完成！")
        except Exception as e:
            logging.error(f"盘前筹码因子生成失败: {e}")

    def run_intraday_tick(self):
        """
        盘中 15m 触发主循环
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"=== 触发盘中 15m 信号引擎扫描 [{now_str}] ===")

        loader = CBDataLoader()
        df_panel = loader.load_minute_panel(start_date="2025-01-01", end_date="2026-07-25", max_bonds=250)
        
        if df_panel.empty:
            logging.warning("当前无有效 15m 行情数据输入！")
            return

        clean_engine = CBStrict15mCleanEngine()
        df_15m = clean_engine.load_and_resample_clean_15m(df_panel)

        unified_engine = CBUnifiedPITEngine()
        df_pit = unified_engine.build_unified_state_panel(df_15m)

        df_orders, target_basket = CBTimeStructuredRouter.generate_time_structured_orders(df_pit)

        if df_orders.empty:
            logging.info("当前 15m 无符合触发条件的交易指令。")
            return

        latest_date = df_orders['trade_date'].iloc[-1]
        today_orders = df_orders[df_orders['trade_date'] == latest_date]
        
        logging.info(f"扫描到 [{latest_date}] 今日目标订单共 {len(today_orders)} 笔，准备执行智能限价挂单算法...")

        for _, ord_row in today_orders.iterrows():
            code = ord_row['ts_code']
            if code not in self.positions and len(self.positions) < 10:
                signal_price = ord_row['execution_price']
                target_volume = 100
                
                exec_result = self.smart_manager.execute_smart_limit_order(
                    security=code,
                    direction='buy',
                    target_volume=target_volume,
                    timeout=30,
                    tick_precision=0.001
                )
                
                if exec_result:
                    avg_px = exec_result['avg_price']
                    self.positions[code] = {
                        'shares': target_volume,
                        'entry_price': avg_px,
                        'entry_time': now_str
                    }
                    logging.info(f"[实盘买入成功] {code} | 成交量: {target_volume}张 | 均价: {avg_px:.3f} | 执行模式: {exec_result['execution_type']}")

    def start_runner_loop(self):
        logging.info(f"=== QMT 实盘/模拟盘全自动策略引擎启动 [{self.account_id}] (模拟模式={self.is_simulation}) ===")
        self.run_pre_market_job()
        self.run_intraday_tick()
        logging.info("=== 本轮实盘策略引擎扫描执行完毕！日志已写入 qmt_live_runner.log ===")

if __name__ == '__main__':
    runner = QMTLiveRunner(account_id="MOCK_ACCOUNT_200W", is_simulation=True)
    runner.start_runner_loop()
