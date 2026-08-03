"""
run_liquidity_test.py
流动性约束测试: 在不同成交额分位数过滤 (剔除低流动性小盘股) 下运行 step4 回测,
对比策略的收益/Sharpe/最大回撤及风格归因 alpha 是否依然显著。

用法: python run_liquidity_test.py
"""
import os
import importlib.util
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
LIQ_DIR = os.path.join(RESULTS_DIR, 'liquidity_test')

spec = importlib.util.spec_from_file_location(
    "step4_portfolio_backtest", os.path.join(SCRIPT_DIR, 'step4_portfolio_backtest.py'))
step4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4)

PRED_FILE = os.path.join(PRED_DIR, 'predictions_purified_ridge.parquet')
if not os.path.exists(PRED_FILE):
    PRED_FILE = os.path.join(PRED_DIR, 'predictions_config_B.parquet')
print(f"Using predictions: {PRED_FILE}")

# 成交额分位数 (剔除市场成交额最低的 Q% 股票)
QUANTILES = [0.0, 0.2, 0.4, 0.6, 0.8]


def main():
    os.makedirs(LIQ_DIR, exist_ok=True)
    rows = []
    for q in QUANTILES:
        cfg_dir = os.path.join(LIQ_DIR, f'q{int(q*100):02d}')
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"\n=== Liquidity filter: drop bottom {q:.0%} by amount ===", flush=True)
        step4.run_backtest(pred_file=PRED_FILE, results_dir=cfg_dir, save_plot=False, liquidity_q=q)
        metrics = pd.read_csv(os.path.join(cfg_dir, 'portfolio_comparison_metrics.csv'))
        pure = metrics[metrics['Portfolio'] == 'Strategy Pure (Net, Fee Included)'].iloc[0]
        hedged = metrics[metrics['Portfolio'] == 'Strategy Options (Net)'].iloc[0]
        rows.append({
            'Liquidity Q (drop bottom%)': q,
            'Total Return': pure['Total Return'],
            'CAGR': pure['CAGR'],
            'Volatility': pure['Volatility'],
            'Sharpe': pure['Sharpe'],
            'Max Drawdown': pure['Max Drawdown'],
            'Hedged CAGR': hedged['CAGR'],
            'Hedged Sharpe': hedged['Sharpe'],
            'Hedged MaxDD': hedged['Max Drawdown'],
        })

    out = pd.DataFrame(rows)
    out_file = os.path.join(RESULTS_DIR, 'liquidity_test_comparison.csv')
    out.to_csv(out_file, index=False)
    print("\n" + "=" * 100)
    print(out.to_string(index=False))
    print("=" * 100)
    print(f"Saved to {out_file}")


if __name__ == '__main__':
    main()
