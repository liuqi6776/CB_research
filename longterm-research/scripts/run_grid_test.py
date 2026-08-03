"""
run_grid_test.py
参数网格验证:
1. 调仓周期: HOLDING_DAYS ∈ {10, 20, 40}
2. QVIX 恐慌阈值: QVIX_PANIC_THRESHOLD ∈ {1.5, 2.0, 2.5}

输出: results/grid_test_comparison.csv
用法: python run_grid_test.py
"""
import os
import importlib.util
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
GRID_DIR = os.path.join(RESULTS_DIR, 'grid_test')

spec = importlib.util.spec_from_file_location(
    "step4_portfolio_backtest", os.path.join(SCRIPT_DIR, 'step4_portfolio_backtest.py'))
step4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4)

PRED_FILE = os.path.join(PRED_DIR, 'predictions_purified_ridge.parquet')
if not os.path.exists(PRED_FILE):
    PRED_FILE = os.path.join(PRED_DIR, 'predictions_config_B.parquet')
print(f"Using predictions: {PRED_FILE}")

HOLDINGS = [10, 20, 40]      # 调仓周期 (交易日)
QVIX_THRESHOLDS = [1.5, 2.0, 2.5]  # QVIX Z-Score 恐慌阈值


def main():
    os.makedirs(GRID_DIR, exist_ok=True)
    rows = []

    # 1. 调仓周期网格
    print("\n" + "=" * 80)
    print("PART 1: Rebalance Holding Days Grid")
    print("=" * 80)
    for h in HOLDINGS:
        cfg_dir = os.path.join(GRID_DIR, f'hold{h:02d}')
        os.makedirs(cfg_dir, exist_ok=True)
        step4.HOLDING_DAYS = h
        print(f"\n=== HOLDING_DAYS = {h} ===", flush=True)
        step4.run_backtest(pred_file=PRED_FILE, results_dir=cfg_dir, save_plot=False)
        metrics = pd.read_csv(os.path.join(cfg_dir, 'portfolio_comparison_metrics.csv'))
        pure = metrics[metrics['Portfolio'] == 'Strategy Pure (Net, Fee Included)'].iloc[0]
        hedged = metrics[metrics['Portfolio'] == 'Strategy Options (Net)'].iloc[0]
        rows.append({
            'Param': f'HoldingDays={h}',
            'Total Return': pure['Total Return'],
            'CAGR': pure['CAGR'],
            'Volatility': pure['Volatility'],
            'Sharpe': pure['Sharpe'],
            'Max Drawdown': pure['Max Drawdown'],
            'Hedged CAGR': hedged['CAGR'],
            'Hedged Sharpe': hedged['Sharpe'],
            'Hedged MaxDD': hedged['Max Drawdown'],
        })

    # 2. QVIX 阈值网格
    print("\n" + "=" * 80)
    print("PART 2: QVIX Panic Threshold Grid")
    print("=" * 80)
    step4.HOLDING_DAYS = 20
    for t in QVIX_THRESHOLDS:
        cfg_dir = os.path.join(GRID_DIR, f'qvix{int(t*10):02d}')
        os.makedirs(cfg_dir, exist_ok=True)
        step4.QVIX_PANIC_THRESHOLD = t
        print(f"\n=== QVIX_PANIC_THRESHOLD = {t} ===", flush=True)
        step4.run_backtest(pred_file=PRED_FILE, results_dir=cfg_dir, save_plot=False)
        metrics = pd.read_csv(os.path.join(cfg_dir, 'portfolio_comparison_metrics.csv'))
        pure = metrics[metrics['Portfolio'] == 'Strategy Pure (Net, Fee Included)'].iloc[0]
        hedged = metrics[metrics['Portfolio'] == 'Strategy Options (Net)'].iloc[0]
        rows.append({
            'Param': f'QVIX_Threshold={t}',
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
    out_file = os.path.join(RESULTS_DIR, 'grid_test_comparison.csv')
    out.to_csv(out_file, index=False)
    print("\n" + "=" * 100)
    print(out.to_string(index=False))
    print("=" * 100)
    print(f"Saved to {out_file}")


if __name__ == '__main__':
    main()
