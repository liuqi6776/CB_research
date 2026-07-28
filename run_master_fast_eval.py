# -*- coding: utf-8 -*-

"""
全量多因子 GBDT 融合策略与 3 档择时控仓评估报告 (纯真实动态计算版)
"""

import sys
import logging
from run_master_multifactor_backtest import run_empirical_backtest

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def main():
    print("\n[EMPIRICAL EVALUATION RUNNER] 正在从行情面板加载数据并实时计算全量样本外 (OOS) 真实回测指标...\n")
    run_empirical_backtest()

if __name__ == '__main__':
    main()
