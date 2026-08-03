# -*- coding: utf-8 -*-
"""
筹码边际因子研究 - 数据拉取
1. 中证1000 (000852.SH) 历史成分股（月末，避免 survivorship bias）
2. 股东户数 stk_holdernumber（含 ann_date 公告日，PIT 对齐关键）

用法:
    python research/chip_momentum/fetch_data.py --limit 20   # 小样本测试
    python research/chip_momentum/fetch_data.py              # 全量
"""
import os
import sys
import time
import argparse
import traceback
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

# 中证1000 指数代码
INDEX_CODE = "000852.SH"
# 成分股拉取起始年份
START_YEAR = 2020
# 数据缓存目录（项目内，避免沙箱对 D 盘写入限制）
_DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HN_DIR = os.path.join(_DATA_ROOT, "holdernumber")
IW_DIR = os.path.join(_DATA_ROOT, "index_weight")


def get_pro():
    import tushare as ts
    pro = ts.pro_api(settings.TUSHARE_TOKEN)
    return pro


def fetch_index_weights(pro, force=False):
    """拉取 2020-至今 每月末中证1000成分股, 存 parquet(按月)"""
    os.makedirs(IW_DIR, exist_ok=True)
    # 用 tushare 交易日历生成每月最后一个交易日
    cal = pro.trade_cal(exchange="SSE", start_date=f"{START_YEAR}0101",
                        end_date="20260731", is_open="1")
    trade_dates = sorted(cal["cal_date"].tolist())
    # 每月取最后一个交易日
    by_month = {}
    for d in trade_dates:
        by_month[d[:6]] = d
    targets = sorted(by_month.values())
    out = []
    for d in targets:
        fp = os.path.join(IW_DIR, f"iw_{d}.parquet")
        if os.path.exists(fp) and not force:
            continue
        try:
            df = pro.index_weight(index_code=INDEX_CODE, trade_date=d)
            if df is not None and not df.empty:
                df["trade_date"] = d
                df.to_parquet(fp)
                out.append(d)
                print(f"index_weight {d}: {len(df)} stocks")
            time.sleep(0.12)
        except Exception as e:
            print(f"index_weight {d} ERR: {e}")
            time.sleep(1)
    print(f"[done] index_weight: {len(out)} months fetched")


def load_constituent_universe():
    """读全部已下载成分股, 返回去重股票列表"""
    codes = set()
    for f in os.listdir(IW_DIR):
        if not f.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(IW_DIR, f))
        codes.update(df["con_code"].astype(str).str.strip().tolist())
    return sorted(codes)


def fetch_holdernumber(pro, codes, limit=None, force=False):
    """逐只拉取股东户数全历史, 每只一个 parquet"""
    os.makedirs(HN_DIR, exist_ok=True)
    if limit:
        codes = codes[:limit]
    ok, fail = 0, []
    for i, code in enumerate(codes):
        fp = os.path.join(HN_DIR, f"{code}.parquet")
        if os.path.exists(fp) and not force:
            ok += 1
            continue
        for attempt in range(4):
            try:
                df = pro.stk_holdernumber(ts_code=code)
                if df is not None and not df.empty:
                    df.to_parquet(fp)
                ok += 1
                break
            except Exception as e:
                if attempt == 3:
                    fail.append((code, str(e)[:120]))
                    print(f"[FAIL] {code}: {e}")
                else:
                    time.sleep(2 + attempt * 2)
        if (i + 1) % 50 == 0:
            print(f"  progress {i+1}/{len(codes)} (ok={ok}, fail={len(fail)})")
        time.sleep(0.15)
    print(f"[done] holdernumber: ok={ok}, fail={len(fail)}")
    if fail:
        print("failed samples:", fail[:10])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只拉前 N 只股票(测试)")
    parser.add_argument("--skip_weight", action="store_true", help="跳过成分股拉取")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pro = get_pro()
    if not args.skip_weight:
        fetch_index_weights(pro, force=args.force)
    codes = load_constituent_universe()
    print(f"[universe] {len(codes)} unique stocks in CSI1000 history")
    fetch_holdernumber(pro, codes, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
