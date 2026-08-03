"""
step3_enhanced.py —— 多因子提纯 Walk-Forward 训练（提纯版）
============================================================

在 step3_train_ranking_model.py 基础上的增强：
1. 市值中性化（Size Neutralization）：横截面预处理增加"按 log_circ_mv 分10桶减去桶内均值"，
   削减 SMB beta 暴露（README 显示 beta_s=0.11~0.20 是主要风险）。
2. 新闻 surprise 特征：news_diff(当日-20日均值) 等聚合新闻因子（诊断显示 IC=+0.018, ICIR=0.67,
   优于原始 news_stock_impact 的 IC≈0），替换/补充原始单日新闻列。
3. PCA 正交化：消除 alpha101/gtja191 高度共线性，每期在训练集上 fit PCA（防泄漏）。
4. 模型可选：Ridge (线性) vs XGBoost (非线性)。

用法:
    python step3_enhanced.py --model ridge  --size_neutral --news_surprise --pca --out predictions_purified_ridge.parquet
    python step3_enhanced.py --model xgb    --size_neutral --news_surprise --pca --out predictions_purified_xgb.parquet
"""
import os
import sys
import time
import gc
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PRED_DIR = os.path.join(PROJECT_DIR, 'predictions')
os.makedirs(PRED_DIR, exist_ok=True)

FEATURES_FILE = os.path.join(DATA_DIR, 'features_longterm.parquet')
OUTPUT_FILE = os.path.join(PRED_DIR, 'predictions_purified_ridge.parquet')

TARGET_COL = 'mkt_excess_ret_20d'
HOLDING_DAYS = 20
RIDGE_ALPHA = 100.0
PCA_VAR_RATIO = 0.95          # PCA 保留 95% 方差
XGB_MAX_ROWS = 700_000        # XGB 每期训练采样上限

# 与 step3 一致的排除列
EXCLUDE_COLS = {
    'ts_code', 'trade_date', 'ds', 'industry',
    'open', 'high', 'low', 'close', 'pre_close',
    'change', 'pct_chg', 'vol', 'amount', 'amplitude',
    'entry_price', 'next_open',
    'ths_hot', 'ths_hot_rank',
    'exit_price_1d', 'return_1d', 'return_1d_open',
    'exit_price_5d', 'return_5d', 'return_5d_open',
    'exit_price_28d', 'return_28d', 'return_28d_open',
    'exit_28d_close',
    'calc_ret5d', 'return_5d_from_open', 'return_28d_from_open',
    'entry_vs_close',
    'return_1d_open_old', 'actual_return',
    't1_intraday_return', 'target_crash_bin', 'target_up_bin',
    'index_ma20_bias',
    'close_T5', 'close_T10', 'close_T20',
    'ret_5d', 'ret_10d', 'ret_20d',
    'mkt_excess_ret_5d', 'mkt_excess_ret_10d', 'mkt_excess_ret_20d',
    'ind_excess_ret_5d', 'ind_excess_ret_10d', 'ind_excess_ret_20d'
}

NEWS_DERIVED = ['news_stock_impact_ma20', 'news_stock_impact_surprise', 'news_mention_ma5']


def get_feature_cols(df, extra_excludes=None):
    exclude = set(EXCLUDE_COLS)
    if extra_excludes:
        exclude.update(extra_excludes)
    return [c for c in df.columns
            if c not in exclude
            and not c.startswith('hist_')
            and df[c].dtype in ('float64', 'float32', 'int64', 'int32')]


def add_news_surprise(df):
    """按股票滚动构造新闻 surprise 特征 (当日新闻 - 20日均值)"""
    if 'news_stock_impact' not in df.columns:
        return df
    print("构造新闻 surprise 特征 (当日 - 20日均值)...", flush=True)
    g = df.groupby('ts_code')
    df['news_stock_impact_ma20'] = g['news_stock_impact'].transform(
        lambda x: x.rolling(20, min_periods=1).mean())
    df['news_stock_impact_surprise'] = df['news_stock_impact'] - df['news_stock_impact_ma20']
    if 'news_has_mention' in df.columns:
        df['news_mention_ma5'] = g['news_has_mention'].transform(
            lambda x: x.rolling(5, min_periods=1).mean())
    return df


def preprocess_factors(df, feature_cols, size_neutral=False):
    """1) Winsorize 1%-99%  2) 行业中性  3) 市值中性(可选)  4) Z-score
    全程 float32 + 原位操作, 控制内存。"""
    print("预处理: Winsorize -> 行业中性 -> {} -> Z-score".format("市值中性" if size_neutral else "无市值中性"), flush=True)

    df_td = df['trade_date'].values

    # 1. Winsorize
    q01 = df.groupby('trade_date')[feature_cols].quantile(0.01)
    q99 = df.groupby('trade_date')[feature_cols].quantile(0.99)
    q01_aligned = q01.loc[df_td].values
    q99_aligned = q99.loc[df_td].values
    df[feature_cols] = np.clip(df[feature_cols].values, q01_aligned, q99_aligned)

    # 2. 行业中性
    ind_means = df.groupby(['trade_date', 'industry'])[feature_cols].transform('mean')
    df[feature_cols] = df[feature_cols] - ind_means

    # 3. 市值中性: 按当日 log_circ_mv 分10桶, 减去桶内均值 (削减 SMB beta)
    if size_neutral:
        if 'log_circ_mv' in df.columns:
            mv = df['log_circ_mv'].fillna(df['log_circ_mv'].median())
            bins = pd.qcut(mv.rank(method='first'), 10, labels=False)
            tmp = df[['trade_date'] + feature_cols].copy()
            tmp['_mv_bin'] = bins
            bucket_means = tmp.groupby(['trade_date', '_mv_bin'])[feature_cols].transform('mean')
            df[feature_cols] = df[feature_cols] - bucket_means
            del tmp, bins, mv
            gc.collect()
        else:
            print("⚠️ log_circ_mv 缺失, 跳过市值中性化", flush=True)

    # 4. Z-score
    date_means = df.groupby('trade_date')[feature_cols].transform('mean')
    date_stds = df.groupby('trade_date')[feature_cols].transform('std')
    date_stds = date_stds.replace(0, 1).fillna(1)
    df[feature_cols] = (df[feature_cols] - date_means) / date_stds

    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def train_and_predict(model='ridge', size_neutral=False, news_surprise=False,
                      pca=False, start_month='202201', output_file=OUTPUT_FILE):
    t0 = time.time()
    print(f"=== 提纯版 WFO: model={model} size_neutral={size_neutral} "
          f"news_surprise={news_surprise} pca={pca} ===", flush=True)
    df = pd.read_parquet(FEATURES_FILE)
    df['ds'] = df['trade_date'].astype(str)

    # 数值特征转 float32 控制内存 (463万行 x 96特征)
    for c in df.columns:
        if df[c].dtype in ('float64',):
            df[c] = df[c].astype('float32')

    # ths_hot_rank → ths_hot_score (与 step3 一致)
    if 'ths_hot_rank' in df.columns:
        df['ths_hot_score'] = np.where(
            (df['ths_hot_rank'].notna()) & (df['ths_hot_rank'] <= 100.0),
            101.0 - df['ths_hot_rank'], 0.0)

    # 新闻 surprise 特征
    if news_surprise:
        df = add_news_surprise(df)

    # 排除原始新闻列(用 surprise 替代) 以控制维度
    exclude_news = {'news_stock_impact', 'news_market_impact', 'news_has_mention'} if news_surprise else set()
    feature_cols = get_feature_cols(df, extra_excludes=exclude_news)
    print(f"特征数量: {len(feature_cols)}", flush=True)

    df = preprocess_factors(df, feature_cols, size_neutral=size_neutral)

    months = sorted(df['ds'].str[:6].unique())
    pred_months = [m for m in months if m >= start_month]
    print(f"预测区间: {pred_months[0]} ~ {pred_months[-1]} ({len(pred_months)} 月)", flush=True)

    trade_dates = sorted(df['trade_date'].unique())
    date_to_idx = {d: i for i, d in enumerate(trade_dates)}

    all_preds = []
    ics = []

    for month in pred_months:
        train_end_month = str(int(month) - 1)
        if train_end_month.endswith('00'):
            train_end_month = f"{int(train_end_month[:4])-1}12"
        year = int(train_end_month[:4]); mv = int(train_end_month[4:6])
        sy = year - 1; sm = mv + 1
        if sm > 12:
            sm -= 12; sy += 1
        rolling_start = f"{sy}{sm:02d}"

        train_dates_raw = [d for d in trade_dates if rolling_start <= d[:6] <= train_end_month]
        if len(train_dates_raw) < 20:
            continue
        # Purging
        purged_last = trade_dates[max(0, date_to_idx[train_dates_raw[-1]] - HOLDING_DAYS)]
        train_dates_purged = [d for d in train_dates_raw if d <= purged_last]

        test_dates = [d for d in trade_dates if d[:6] == month]

        train_mask = df['trade_date'].isin(train_dates_purged) & df[TARGET_COL].notna()
        test_mask = df['trade_date'].isin(test_dates)
        train_df = df.loc[train_mask, feature_cols + [TARGET_COL]].copy()
        test_df = df.loc[test_mask, feature_cols + ['trade_date', 'ts_code', 'next_open',
                                                    'close', 'pct_chg', 'industry',
                                                    'ret_20d', 'mkt_excess_ret_20d']].copy()
        if len(train_df) < 5000 or len(test_df) == 0:
            continue

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df[TARGET_COL]
        X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # PCA 正交化 (仅训练集 fit, 防泄漏)
        if pca:
            pca_m = PCA(n_components=PCA_VAR_RATIO, random_state=42)
            X_train = pca_m.fit_transform(X_train)
            X_test = pca_m.transform(X_test)

        if model == 'ridge':
            mdl = Ridge(alpha=RIDGE_ALPHA)
        elif model == 'xgb':
            # XGB 训练采样上限(70万行, 提速); PCA 后 X_train 是 ndarray
            if len(X_train) > XGB_MAX_ROWS:
                idx = np.random.RandomState(42).choice(len(X_train), XGB_MAX_ROWS, replace=False)
                if isinstance(X_train, np.ndarray):
                    X_train = X_train[idx]
                    y_train = y_train.iloc[idx]
                else:
                    X_train = X_train.iloc[idx]
                    y_train = y_train.iloc[idx]
            mdl = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        else:
            raise ValueError(model)
        mdl.fit(X_train, y_train)
        test_df['pred_score'] = mdl.predict(X_test)

        valid = test_df[test_df[TARGET_COL].notna()]
        ic = valid['pred_score'].corr(valid[TARGET_COL], method='spearman') if len(valid) else np.nan
        ics.append(ic)
        print(f"Month {month} | train {len(train_df)} | test {len(test_df)} | OOS Rank IC {ic:+.4f}", flush=True)
        all_preds.append(test_df[['trade_date', 'ts_code', 'next_open', 'close', 'pct_chg',
                                  'industry', 'ret_20d', 'mkt_excess_ret_20d', 'pred_score']])
        del train_df, test_df, X_train, X_test
        gc.collect()

    pred_df = pd.concat(all_preds, ignore_index=True)
    ics = pd.Series(ics).dropna()
    print(f"\nOOS 平均 Rank IC: {ics.mean():.4f} | ICIR: {ics.mean()/ics.std():.3f} | IC>0比例: {(ics>0).mean():.2%}")
    print(f"保存到 {output_file}")
    pred_df.to_parquet(output_file, index=False)
    print(f"总耗时 {time.time()-t0:.1f}s")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['ridge', 'xgb'], default='ridge')
    ap.add_argument('--size_neutral', action='store_true')
    ap.add_argument('--news_surprise', action='store_true')
    ap.add_argument('--pca', action='store_true')
    ap.add_argument('--out', default=OUTPUT_FILE)
    ap.add_argument('--start_month', default='202201')
    args = ap.parse_args()
    train_and_predict(model=args.model, size_neutral=args.size_neutral,
                      news_surprise=args.news_surprise, pca=args.pca,
                      start_month=args.start_month, output_file=args.out)
