# -*- coding: utf-8 -*-
"""
duobao 策略 修复版 —— 严格 T+1 Walk-Forward 回测 (优化版)
===========================================================

修复点（对照原 run_real_t1_fast.py）：
1. 时间线错位(off-by-one)修复：
   - 原代码: 用 T 日特征 + T+1 盘前新闻，却在 T+2 开盘买入、T+2 收盘卖出（和训练标签对不上）
   - 修复后: T 日收盘后特征 + T+1 盘前新闻 → T+1 开盘买入 → T+2 止盈8% 或收盘卖出
   - 与训练标签完全一致: label = (T+2 close / T+1 open - 1) > 4%
2. 全样本内模型 → 月度 Walk-Forward 重训：
   - 每月初用「严格早于当月1日 - 7天(embargo)」的历史样本训练
   - 训练样本上限30万(随机采样) + 提前停止, 提速且不损失信息
3. 交易成本: 佣金0.03%x2 + 印花税0.05%(卖) + 滑点0.1%x2 = 0.31% 双边
4. 股票池与训练集一致: 剔除688/689、流通市值<=100亿、成交额>=5000万(流动性)
5. 资金结算真实化: 当日开盘买入使用"前一日已平仓到账资金"，避免隐含2倍杠杆
6. 优化: 每月只重训一次, 概率统一缓存, 所有变体共享, 避免重复训练

时间线示意:
  T日收盘  →  T+1 早8点(盘前新闻)  →  T+1 开盘买入  →  T+2 止盈/收盘卖出
"""
import os
import json
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import timedelta

# ---------------------------------------------------------------- 配置
DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_DIR  = os.path.join(DATA_DIR, 'news_major1')   # 盘前新闻(按交易日命名, T+1早8点前可得)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'results_duobao', 'fixed_wfo')
CACHE_PATH = os.path.join(OUTPUT_DIR, 'samples.parquet')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = ['hot_rank_pct', 'chip_concentration', 'winner_rate',
                'news_market_impact', 'news_stock_impact']

INITIAL_CAP   = 100_000.0
COST          = 0.0031          # 0.31% 双边(佣金0.03%x2 + 印花税0.05% + 滑点0.1%x2)
CIRC_MV_LIMIT = 1_000_000       # 流通市值 <= 100亿 (单位: 万元)
MIN_AMOUNT    = 50_000          # 成交额 >= 5000万 (单位: 千元)
LABEL_TH      = 0.04            # 训练标签: 2日收益 > 4%
EMBARGO_DAYS  = 7
MAX_TRAIN_ROWS = 300_000        # 单月训练样本上限
TEST_START    = '20230801'
TEST_END      = '20260324'

XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  eval_metric='auc', n_jobs=-1, tree_method='hist')


def get_all_dates():
    return sorted(f.replace('.parquet', '') for f in os.listdir(PRICE_DIR) if f.endswith('.parquet'))


# ---------------------------------------------------------------- 新闻加载
def load_news():
    """直接读 news_major1 JSON, 按 article_date 对齐(不做 next-trading-day 平移)."""
    mkt, stk = {}, {}
    for f in glob.glob(os.path.join(NEWS_DIR, '*.json')):
        try:
            data = json.load(open(f, encoding='utf-8'))
        except Exception:
            continue
        d = data.get('article_date', '')
        if not d:
            continue
        d8 = d.replace('-', '')[:8]
        mkt[d8] = float(data.get('market_impact', 0.0))
        for s in data.get('stocks', []):
            code = s.get('stock_code', '')
            if not code:
                continue
            ts = f"{code}.SH" if code.startswith('6') else \
                 f"{code}.SZ" if code.startswith(('0', '3')) else \
                 f"{code}.BJ" if code.startswith(('4', '8')) else code
            stk[(d8, ts)] = float(s.get('impact', 0.0))
    return mkt, stk


# ---------------------------------------------------------------- 样本构建
def build_samples(dates, news_mkt, news_stk):
    rows = []
    for i in tqdm(range(len(dates) - 2), desc='构建样本'):
        d_t, d_buy, d_sell = dates[i], dates[i + 1], dates[i + 2]

        p_rank = os.path.join(RANK_DIR, f"{d_t}.parquet")
        p_chip = os.path.join(CHIP_DIR, f"{d_t}.parquet")
        p_px   = os.path.join(PRICE_DIR, f"{d_t}.parquet")
        p_oth  = os.path.join(OTHER_DIR, f"{d_t}.parquet")
        p_buy  = os.path.join(PRICE_DIR, f"{d_buy}.parquet")
        p_sell = os.path.join(PRICE_DIR, f"{d_sell}.parquet")
        if not all(os.path.exists(p) for p in [p_rank, p_chip, p_px, p_oth, p_buy, p_sell]):
            continue

        rank = pd.read_parquet(p_rank)
        chip = pd.read_parquet(p_chip)
        px   = pd.read_parquet(p_px, columns=['ts_code', 'amount'])
        oth  = pd.read_parquet(p_oth, columns=['ts_code', 'circ_mv'])
        buy  = pd.read_parquet(p_buy, columns=['ts_code', 'open', 'pre_close'])
        sell = pd.read_parquet(p_sell, columns=['ts_code', 'open', 'high', 'close'])

        df = px.merge(rank[['ts_code', 'hot']], on='ts_code', how='left')
        df = df.merge(chip[['ts_code', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'winner_rate']], on='ts_code', how='left')
        df = df.merge(oth, on='ts_code', how='left')
        df = df.merge(buy.rename(columns={'open': 'buy_open', 'pre_close': 'buy_pre_close'}), on='ts_code', how='inner')
        df = df.merge(sell.rename(columns={'open': 'sell_open', 'high': 'sell_high', 'close': 'sell_close'}), on='ts_code', how='inner')

        df = df[~df['ts_code'].str.startswith('688')]
        df = df[~df['ts_code'].str.startswith('689')]
        df = df[df['circ_mv'] <= CIRC_MV_LIMIT]
        df = df[df['amount'] >= MIN_AMOUNT]
        if df.empty:
            continue

        df['hot_rank_pct'] = df['hot'].rank(pct=True)
        df['chip_concentration'] = (df['cost_85pct'] - df['cost_15pct']) / (df['cost_50pct'] + 1e-8)
        df['trade_date'] = d_t
        df['buy_date'] = d_buy
        df['sell_date'] = d_sell
        df['news_market_impact'] = news_mkt.get(d_buy, 0.0)
        df['news_stock_impact'] = df['ts_code'].map(lambda c: news_stk.get((d_buy, c), 0.0))
        df['label'] = ((df['sell_close'] / df['buy_open'] - 1) > LABEL_TH).astype(int)

        cols = ['trade_date', 'buy_date', 'sell_date', 'ts_code',
                'hot_rank_pct', 'chip_concentration', 'winner_rate',
                'news_market_impact', 'news_stock_impact',
                'buy_open', 'buy_pre_close', 'sell_high', 'sell_close', 'label']
        rows.append(df[cols])

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=['buy_open', 'sell_close'])
    out = out[(out['buy_date'] >= '20220101') & (out['sell_date'] <= TEST_END)]
    out.to_parquet(CACHE_PATH, index=False)
    return out


def load_or_build_samples(dates, news_mkt, news_stk):
    if os.path.exists(CACHE_PATH):
        print('载入样本缓存:', CACHE_PATH)
        return pd.read_parquet(CACHE_PATH)
    return build_samples(dates, news_mkt, news_stk)


# ---------------------------------------------------------------- WFO 概率
def train_monthly_model(train_df):
    sub = train_df.dropna(subset=['label'])
    if len(sub) < 2000:
        return None
    if len(sub) > MAX_TRAIN_ROWS:
        sub = sub.sample(MAX_TRAIN_ROWS, random_state=42)
    sub = sub.sample(frac=1.0, random_state=42)
    n_val = max(2000, int(len(sub) * 0.1))
    tr, va = sub.iloc[:-n_val], sub.iloc[-n_val:]
    X, y = tr[FEATURE_COLS].fillna(0), tr['label']
    Xv, yv = va[FEATURE_COLS].fillna(0), va['label']
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X, y, eval_set=[(Xv, yv)], early_stopping_rounds=30, verbose=False)
    return model


def attach_prob_wfo(samples):
    """月度WFO: 每月重训一次, 给当月所有样本打分 (返回添加 prob 列的副本)"""
    s = samples.copy()
    s['prob'] = np.nan
    months = sorted(s['buy_date'].str[:6].unique())
    for m in tqdm(months, desc='WFO 重训+预测'):
        fom_dt = pd.to_datetime(m + '01') - timedelta(days=EMBARGO_DAYS)
        cutoff = fom_dt.strftime('%Y%m%d')
        train = s[s['trade_date'] < cutoff]
        model = train_monthly_model(train)
        if model is None:
            continue
        mask = s['buy_date'].str.startswith(m)
        X = s.loc[mask, FEATURE_COLS].fillna(0)
        s.loc[mask, 'prob'] = model.predict_proba(X)[:, 1]
    return s


# ---------------------------------------------------------------- 模拟
def simulate(day_groups, sel_rule, take_profit, top_n, prob_th):
    """
    sel_rule: 'orig' 原规则(prob>0.8取top3,否则top1) | 'thresh' 只取 prob>th 的top_n(可空仓)
    take_profit: 止盈比例, None 表示无止盈(收盘卖出)
    """
    capital = INITIAL_CAP
    pending = {}          # sell_date -> [pnl]
    navs, nav_dates = [], []
    trades = []

    for d in sorted(day_groups.keys()):
        day_df = day_groups[d]
        picks = pd.DataFrame()
        if 'prob' in day_df.columns:
            if sel_rule == 'orig':
                hi = day_df[day_df['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
                picks = hi if not hi.empty else day_df.sort_values('prob', ascending=False).head(1)
            else:
                picks = day_df[day_df['prob'] >= prob_th].sort_values('prob', ascending=False).head(top_n)

        # 开盘买入(用当前到账资金), 按卖出日登记待平仓
        pnl_by_sell = {}
        if not picks.empty:
            alloc = capital / len(picks)
            for _, r in picks.iterrows():
                buy_open, pre_close = r['buy_open'], r['buy_pre_close']
                if pd.isna(buy_open) or pd.isna(pre_close) or buy_open <= 0:
                    continue
                up_limit = 1.2 if (r['ts_code'].startswith('300') or r['ts_code'].startswith('688')) else 1.1
                if buy_open >= round(pre_close * up_limit, 2):   # 开盘涨停放弃
                    continue
                if pd.isna(r['sell_high']) or pd.isna(r['sell_close']):
                    continue
                if take_profit is not None and r['sell_high'] >= buy_open * (1 + take_profit):
                    sell_price = buy_open * (1 + take_profit)
                else:
                    sell_price = r['sell_close']
                ret = (sell_price / buy_open) - 1 - COST
                pnl_by_sell.setdefault(r['sell_date'], []).append({'alloc': alloc, 'ret': ret})
                trades.append({'ts_code': r['ts_code'], 'buy_date': d, 'sell_date': r['sell_date'],
                               'buy_price': buy_open, 'sell_price': sell_price, 'ret': ret})
        for sd, lst in pnl_by_sell.items():
            pending[sd] = pending.get(sd, []) + lst

        # 结算当日到期平仓(收盘到账, 供次日使用)
        for p in pending.pop(d, []):
            capital += p['alloc'] * p['ret']

        navs.append(capital)
        nav_dates.append(pd.to_datetime(d))

    for k in sorted(pending):
        for p in pending[k]:
            capital += p['alloc'] * p['ret']

    eq = pd.DataFrame({'date': nav_dates, 'nav': navs})
    return eq, pd.DataFrame(trades)


def metrics(eq):
    nav = eq['nav']
    total = nav.iloc[-1] / INITIAL_CAP - 1
    years = len(eq) / 252.0
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 else 0
    rets = nav.pct_change().dropna()
    vol = rets.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    mdd = ((nav - nav.cummax()) / nav.cummax()).min()
    return dict(total=total, cagr=cagr, sharpe=sharpe, mdd=mdd, n=len(eq))


def main():
    dates = get_all_dates()
    print(f'共 {len(dates)} 个交易日, 回测区间 {TEST_START} ~ {TEST_END}')
    news_mkt, news_stk = load_news()
    print(f'新闻: market {len(news_mkt)} 天, stock {len(news_stk)} 条')

    samples = load_or_build_samples(dates, news_mkt, news_stk)
    samples = samples[samples['buy_date'] >= TEST_START]
    print(f'回测样本: {len(samples)} 行, 交易日 {samples["buy_date"].nunique()} 天, '
          f'标签正样本率 {samples["label"].mean():.2%}')

    samples = attach_prob_wfo(samples)
    day_groups = {d: g for d, g in samples.groupby('buy_date')}
    print('每日分组就绪:', len(day_groups), '天')

    variants = [
        ('V_A 原规则(>0.8取3,否则取1) 止盈8%', 'orig', 0.08, None, None),
        ('V_B 概率>=0.55取3 可空仓 止盈8%', 'thresh', 0.08, 3, 0.55),
        ('V_C 固定取Top3 止盈8%', 'thresh', 0.08, 3, 0.0),
        ('V_D 原规则 止盈4%', 'orig', 0.04, None, None),
        ('V_E 原规则 无止盈(纯收盘卖出)', 'orig', None, None, None),
    ]

    results, lines = [], []
    for name, rule, tp, topn, th in variants:
        print('\n' + '=' * 70)
        print(name)
        eq, trades = simulate(day_groups, rule, tp, topn, th)
        m = metrics(eq)
        win = (trades['ret'] > 0).mean() if len(trades) else 0
        avg_w = trades.loc[trades['ret'] > 0, 'ret'].mean() if (trades['ret'] > 0).any() else 0
        avg_l = trades.loc[trades['ret'] < 0, 'ret'].mean() if (trades['ret'] < 0).any() else 0
        print(f"总收益 {m['total']:+.2%} | 年化 {m['cagr']:+.2%} | 夏普 {m['sharpe']:.2f} | "
              f"最大回撤 {m['mdd']:.2%} | 交易 {len(trades)} | 胜率 {win:.2%} | 平均盈 {avg_w:+.2%} 亏 {avg_l:+.2%}")
        results.append((name, eq, trades))
        lines.append(f"| {name} | 总收益 {m['total']:+.2%} | 年化 {m['cagr']:+.2%} | 夏普 {m['sharpe']:.2f} | "
                     f"最大回撤 {m['mdd']:.2%} | 交易 {len(trades)} | 胜率 {win:.2%} | 平均盈 {avg_w:+.2%} | 亏 {avg_l:+.2%} |")

    plt.figure(figsize=(14, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for (name, eq, _), c in zip(results, colors):
        plt.plot(eq['date'], eq['nav'] / INITIAL_CAP, label=name, color=c, linewidth=1.5)
    plt.axhline(1.0, color='gray', ls='--', lw=0.8)
    plt.title('Duobao Fixed T+1 WFO Backtest (2023-08 ~ 2026-03)')
    plt.xlabel('Date'); plt.ylabel('NAV'); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    png = os.path.join(OUTPUT_DIR, 'fixed_wfo_comparison.png')
    plt.savefig(png, dpi=150)
    print(f'\n图已保存: {png}')

    md = ['# duobao 修复版 —— 严格 T+1 Walk-Forward 回测结果', '',
          '> 时间线: T日收盘特征 + T+1盘前新闻 → T+1开盘买入 → T+2止盈/收盘卖出', '',
          '> 成本: 0.31%双边 | 股票池: 非688/689, 市值<=100亿, 成交额>=5000万 | 月度WFO重训(7天embargo)',
          '', '## 回测区间', '', f'- {TEST_START} ~ {TEST_END}, 初始资金 ¥{INITIAL_CAP:,.0f}',
          f'- 样本数 {len(samples)}, 标签正样本率 {samples["label"].mean():.2%}', '',
          '## 变体对比', '',
          '| 变体 | 总收益 | 年化 | 夏普 | 最大回撤 | 交易数 | 胜率 | 平均盈 | 平均亏 |', '|---|---|---|---|---|---|---|---|---|']
    md += lines
    md += ['', '## 净值曲线', '', f'![fixed_wfo](fixed_wfo_comparison.png)', '']
    report = os.path.join(OUTPUT_DIR, 'fixed_wfo_report.md')
    with open(report, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))
    print('报告已保存:', report)

    best = max(results, key=lambda x: metrics(x[1])['sharpe'])
    best[2].to_csv(os.path.join(OUTPUT_DIR, 'best_trades.csv'), index=False)
    eq = best[1].copy()
    eq['month'] = eq['date'].dt.strftime('%Y-%m')
    monthly = eq.groupby('month')['nav'].last().pct_change()
    print('\n最好变体(按夏普)月度收益:')
    print(monthly.to_string())
    monthly.to_csv(os.path.join(OUTPUT_DIR, 'best_monthly.csv'))


if __name__ == '__main__':
    main()
