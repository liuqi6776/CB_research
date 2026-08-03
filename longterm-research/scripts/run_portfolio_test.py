"""
run_portfolio_test.py
组合构建升级对比:
- equal:   等权 Top50 (当前基线)
- inv_vol: 波动率倒数加权
- ind_rp:  行业风险平价 (行业等权 + 行业内波动率倒数)

输出: results/portfolio_construction_comparison.csv
用法: python run_portfolio_test.py
"""
import os
import importlib.util
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
PF_DIR = os.path.join(RESULTS_DIR, 'portfolio_construction')

spec = importlib.util.spec_from_file_location(
    "step4_portfolio_backtest", os.path.join(SCRIPT_DIR, 'step4_portfolio_backtest.py'))
step4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4)

PRED_FILE = os.path.join(PRED_DIR, 'predictions_purified_ridge.parquet')
if not os.path.exists(PRED_FILE):
    PRED_FILE = os.path.join(PRED_DIR, 'predictions_config_B.parquet')
print(f"Using predictions: {PRED_FILE}")

WEIGHTINGS = ['equal', 'inv_vol', 'ind_rp']


def main():
    os.makedirs(PF_DIR, exist_ok=True)
    rows = []
    for w in WEIGHTINGS:
        cfg_dir = os.path.join(PF_DIR, w)
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"\n=== Weighting: {w} ===", flush=True)
        step4.run_backtest(pred_file=PRED_FILE, results_dir=cfg_dir, save_plot=False, weighting=w)
        metrics = pd.read_csv(os.path.join(cfg_dir, 'portfolio_comparison_metrics.csv'))
        pure = metrics[metrics['Portfolio'] == 'Strategy Pure (Net, Fee Included)'].iloc[0]
        hedged = metrics[metrics['Portfolio'] == 'Strategy Options (Net)'].iloc[0]
        rows.append({
            'Weighting': w,
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
    out_file = os.path.join(RESULTS_DIR, 'portfolio_construction_comparison.csv')
    out.to_csv(out_file, index=False)
    print("\n" + "=" * 100)
    print(out.to_string(index=False))
    print("=" * 100)
    print(f"Saved to {out_file}")


if __name__ == '__main__':
    main()
