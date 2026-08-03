"""
compare_purified_ic.py
统一口径对比各配置的 OOS 月度 Rank IC / ICIR / IC>0 比例。
用法: python compare_purified_ic.py
"""
import os
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')

CONFIGS = {
    'Baseline Ridge (Config B)': 'predictions_config_B.parquet',
    'Baseline Ridge (base)': 'pred_base_ridge.parquet',
    'Ridge + SizeNeutral': 'predictions_purified_sizeneutral.parquet',
    'Ridge + NewsSurprise': 'predictions_purified_news.parquet',
    'Ridge + Full Purify': 'predictions_purified_ridge.parquet',
    'XGB + Full Purify': 'predictions_purified_xgb.parquet',
}

def monthly_ic_stats(df):
    df = df[df['mkt_excess_ret_20d'].notna()].copy()
    df['month'] = df['trade_date'].astype(str).str[:6]
    ics = df.groupby('month').apply(
        lambda g: g['pred_score'].corr(g['mkt_excess_ret_20d'], method='spearman')).dropna()
    return {
        'OOS Avg Rank IC': ics.mean(),
        'ICIR': ics.mean() / ics.std() if len(ics) > 1 else np.nan,
        'IC>0 Ratio': (ics > 0).mean(),
        'Months': len(ics),
    }

rows = []
for name, fname in CONFIGS.items():
    path = os.path.join(PRED_DIR, fname)
    if not os.path.exists(path):
        print(f"[SKIP] {name}: {fname} not found")
        continue
    df = pd.read_parquet(path)
    stats = monthly_ic_stats(df)
    stats['Config'] = name
    rows.append(stats)
    print(f"{name:26s} | IC {stats['OOS Avg Rank IC']:+.4f} | "
          f"ICIR {stats['ICIR']:.3f} | IC>0 {stats['IC>0 Ratio']:.2%} | months {stats['Months']}")

out = pd.DataFrame(rows)
out_file = os.path.join(PROJECT_DIR, 'results', 'purified_ic_comparison.csv')
os.makedirs(os.path.dirname(out_file), exist_ok=True)
out.to_csv(out_file, index=False)
print(f"\nSaved to {out_file}")
