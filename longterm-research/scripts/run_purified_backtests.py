"""
run_purified_backtests.py
对每个提纯配置的预测文件运行 step4 回测，汇总绩效指标，并对每个配置做
紧凑版风格归因 (Model 2: Market + SMB + Industry Excess, HAC)。

用法: python run_purified_backtests.py
"""
import os
import sys
import importlib.util
import pandas as pd
import numpy as np
import statsmodels.api as sm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
PURIFIED_DIR = os.path.join(RESULTS_DIR, 'purified')
FEATURES_FILE = os.path.join(PROJECT_DIR, 'data', 'features_longterm.parquet')

# 加载 step4 模块
spec = importlib.util.spec_from_file_location(
    "step4_portfolio_backtest", os.path.join(SCRIPT_DIR, 'step4_portfolio_backtest.py'))
step4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4)

CONFIGS = [
    ('Baseline (Ridge)', 'pred_base_ridge.parquet'),
    ('Ridge FullPurify', 'predictions_purified_ridge.parquet'),
    ('XGB FullPurify', 'predictions_purified_xgb.parquet'),
    ('Ridge SizeNeutral', 'predictions_purified_sizeneutral.parquet'),
    ('Ridge NewsSurprise', 'predictions_purified_news.parquet'),
]


def style_attribution(nav_csv, start_date, end_date):
    """紧凑版 Model2 归因: alpha / t / p / beta_s / R2"""
    df_nav = pd.read_csv(nav_csv, index_col=0)
    df_nav.index = pd.to_datetime(df_nav.index)
    df_nav = df_nav.sort_index()
    strat_ret = df_nav['Strategy_Pure'].pct_change().fillna(0.0)
    df_nav['Strategy_Ret'] = strat_ret
    df_nav['td'] = df_nav.index.strftime('%Y%m%d')
    dates = df_nav['td'].tolist()

    feat = pd.read_parquet(FEATURES_FILE, columns=['trade_date', 'ts_code', 'pct_chg', 'circ_mv', 'industry'])
    feat['trade_date'] = feat['trade_date'].astype(str)
    feat = feat[(feat['trade_date'] >= start_date) & (feat['trade_date'] <= end_date)].copy()
    feat['pct_chg'] = feat['pct_chg'].fillna(0.0)
    feat['circ_mv'] = pd.to_numeric(feat['circ_mv'], errors='coerce').fillna(0.0)

    mkt = feat.groupby('trade_date')['pct_chg'].mean().rename('R_m')

    def smb(g):
        g = g[g['circ_mv'] > 0]
        if len(g) < 10:
            return np.nan
        s = g.sort_values('circ_mv')
        n = max(1, int(len(s) * 0.3))
        return s.iloc[:n]['pct_chg'].mean() - s.iloc[-n:]['pct_chg'].mean()
    smb_ser = feat.groupby('trade_date').apply(smb).rename('SMB').dropna()

    ind = feat.groupby(['trade_date', 'industry'])['pct_chg'].mean().unstack(fill_value=0.0)
    ind_cols = [c for c in ind.columns if c != 'Unknown']

    reg = pd.DataFrame({'td': dates, 'Strategy_Ret': df_nav['Strategy_Ret'].values})
    reg = reg.merge(mkt.rename('R_m'), left_on='td', right_index=True, how='inner')
    reg = reg.merge(smb_ser, left_on='td', right_index=True, how='inner')
    reg = reg.merge(ind, left_on='td', right_index=True, how='inner')
    for c in ind_cols:
        reg[c] = reg[c] - reg['R_m']
    reg = reg.dropna(subset=['R_m', 'SMB'])

    y = reg['Strategy_Ret']
    X = sm.add_constant(reg[['R_m', 'SMB'] + ind_cols])
    m = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 20})
    return {
        'alpha_ann': m.params['const'] * 252,
        'alpha_t': m.tvalues['const'],
        'alpha_p': m.pvalues['const'],
        'beta_s': m.params['SMB'],
        'beta_s_t': m.tvalues['SMB'],
        'r2': m.rsquared,
        'n_days': len(reg),
    }


def main():
    os.makedirs(PURIFIED_DIR, exist_ok=True)
    all_rows = []
    for name, fname in CONFIGS:
        pred_file = os.path.join(PRED_DIR, fname)
        if not os.path.exists(pred_file):
            print(f"[SKIP] {name}: {fname} not found")
            continue
        cfg_dir = os.path.join(PURIFIED_DIR, name.replace(' ', '_'))
        os.makedirs(cfg_dir, exist_ok=True)
        print(f"\n=== Backtest: {name} ({fname}) ===", flush=True)
        step4.run_backtest(pred_file=pred_file, results_dir=cfg_dir, save_plot=False)

        metrics = pd.read_csv(os.path.join(cfg_dir, 'portfolio_comparison_metrics.csv'))
        pure = metrics[metrics['Portfolio'] == 'Strategy Pure (Net, Fee Included)'].iloc[0]
        hedged = metrics[metrics['Portfolio'] == 'Strategy Options (Net)'].iloc[0]
        csi = metrics[metrics['Portfolio'] == 'Benchmark (CSI 1000 Index)'].iloc[0]

        nav_csv = os.path.join(cfg_dir, 'portfolio_comparison_nav.csv')
        nav = pd.read_csv(nav_csv, index_col=0)
        start_date = nav.index[0].replace('-', '')
        end_date = nav.index[-1].replace('-', '')
        attr = style_attribution(nav_csv, start_date, end_date)

        all_rows.append({
            'Config': name,
            'Total Return': pure['Total Return'],
            'CAGR': pure['CAGR'],
            'Volatility': pure['Volatility'],
            'Sharpe': pure['Sharpe'],
            'Max Drawdown': pure['Max Drawdown'],
            'Hedged CAGR': hedged['CAGR'],
            'Hedged Sharpe': hedged['Sharpe'],
            'CSI1000 CAGR': csi['CAGR'],
            'Alpha_ann': f"{attr['alpha_ann']:.2%}",
            'Alpha_t': f"{attr['alpha_t']:.2f}",
            'Alpha_p': f"{attr['alpha_p']:.4f}",
            'Beta_S': f"{attr['beta_s']:.3f}",
            'Beta_S_t': f"{attr['beta_s_t']:.2f}",
            'R2': f"{attr['r2']:.2%}",
        })
        print(f"  Alpha_ann={attr['alpha_ann']:.2%} t={attr['alpha_t']:.2f} p={attr['alpha_p']:.4f} "
              f"beta_s={attr['beta_s']:.3f} R2={attr['r2']:.2%}", flush=True)

    out = pd.DataFrame(all_rows)
    out_file = os.path.join(RESULTS_DIR, 'purified_backtest_comparison.csv')
    out.to_csv(out_file, index=False)
    print("\n" + "=" * 100)
    print(out.to_string(index=False))
    print("=" * 100)
    print(f"Saved to {out_file}")


if __name__ == '__main__':
    main()
