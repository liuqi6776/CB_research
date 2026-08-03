#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可转债全量历史 5 分钟 K 线数据采集工具 (2020年-至今全量超长跨度版)
Convertible Bond 5-Min K-Line Data Ingestion Tool (2020 - Present Full History)
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime
import pandas as pd
import tushare as ts
from pytdx.hq import TdxHq_API

# 配置日志 / Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

DATA_DIR = r"D:\CB_mins_data"
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")
CSV_DIR = os.path.join(DATA_DIR, "csv")
MIN_DATE = "2020-01-01"

TDX_SERVERS = [
    ('115.238.56.198', 7709),
    ('180.153.18.170', 7709),
    ('218.75.126.9', 7709),
    ('60.191.117.167', 7709),
    ('124.160.88.183', 7709)
]

def ensure_directories():
    """建立保存目录"""
    for d in [DATA_DIR, PARQUET_DIR, CSV_DIR]:
        os.makedirs(d, exist_ok=True)

def load_tushare_token(custom_token=None):
    """获取 Tushare Token"""
    if custom_token and custom_token.strip():
        return custom_token.strip()
    
    token = os.getenv("TUSHARE_TOKEN")
    if token and token.strip() and token != "your_tushare_token_here":
        return token.strip()
    
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("TUSHARE_TOKEN="):
                    t = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if t and t != "your_tushare_token_here":
                        return t
    return None

def get_all_convertible_bonds(pro):
    """获取全量可转债名录"""
    logging.info("获取全部可转债基本信息 (Fetching convertible bond basic info)...")
    dfs = []
    for status in ['L', 'D', 'P']:
        try:
            df = pro.cb_basic(
                list_status=status,
                fields='ts_code,bond_full_name,bond_short_name,stk_code,stk_short_name,list_date,delist_date,issue_size'
            )
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception as e:
            logging.warning(f"获取状态 {status} 可转债失败: {e}")
            
    if not dfs:
        raise ValueError("未能获取到任何可转债基本信息。")
    
    all_bonds = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['ts_code'])
    all_bonds['list_date_clean'] = all_bonds['list_date'].fillna('19000101')
    all_bonds['delist_date_clean'] = all_bonds['delist_date'].fillna('20991231')
    
    # 筛选在 2020 年及以后交易过的债券
    valid_bonds = all_bonds[all_bonds['delist_date_clean'] >= '20200101'].copy()
    valid_bonds = valid_bonds.sort_values(by='list_date_clean', ascending=False)
    return valid_bonds

def connect_tdx():
    """连接最快的 TDX 行情服务器"""
    api = TdxHq_API()
    for ip, port in TDX_SERVERS:
        try:
            if api.connect(ip, port):
                logging.info(f"成功连接 TDX 高速行情服务器: {ip}:{port}")
                return api
        except Exception:
            pass
    raise ConnectionError("无法连接至任何 TDX 行情服务器。")

def fetch_full_history_5min(api, ts_code, min_date=MIN_DATE):
    """
    多页历史游标回溯拉取 2020 年至今的全部 5 分钟 K 线数据 (最高支持 100 页 = 80,000 根 5分钟线)
    """
    code, exchange = ts_code.split('.')
    market = 1 if exchange == 'SH' else 0
    category = 0 # 0 代表 5分钟线
    
    all_bars = []
    max_pages = 100 # 最多翻 100 页 (80,000 根 5分钟 K 线，覆盖 2020 年至今全部约 6.5 年历史)
    
    for page in range(max_pages):
        try:
            bars = api.get_security_bars(category, market, code, page * 800, 800)
            if not bars:
                break
            
            all_bars.extend(bars)
            earliest_dt = bars[-1]['datetime']
            
            # 若拉取到的最早时间已早于等于 2020-01-01，停止向更早期翻页
            if earliest_dt <= min_date:
                break
        except Exception as e:
            logging.warning(f"[{ts_code}] 翻页 {page} 发生异常: {e}")
            break
            
    if not all_bars:
        return pd.DataFrame()
        
    df = api.to_df(all_bars)
    df.rename(columns={
        'datetime': 'trade_time',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'vol': 'vol',
        'amount': 'amount'
    }, inplace=True)
    
    df['trade_time'] = df['trade_time'].astype(str)
    df = df[df['trade_time'] >= min_date].copy()
    df = df.sort_values(by='trade_time', ascending=True).drop_duplicates(subset=['trade_time']).reset_index(drop=True)
    df['ts_code'] = ts_code
    return df

def save_bond_data(ts_code, df):
    """保存债券数据至 Parquet 和 CSV"""
    if df.empty:
        logging.info(f"[{ts_code}] 无 2020 至今的 5分钟线数据，跳过保存。")
        return False
    
    parquet_file = os.path.join(PARQUET_DIR, f"{ts_code}.parquet")
    csv_file = os.path.join(CSV_DIR, f"{ts_code}.csv")
    
    df.to_parquet(parquet_file, index=False)
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    logging.info(f"[{ts_code}] 成功保存 {len(df)} 条全量 5分钟线 ({df['trade_time'].iloc[0]} ~ {df['trade_time'].iloc[-1]}) -> {parquet_file}")
    return True

def main():
    parser = argparse.ArgumentParser(description="可转债全量历史 5 分钟 K 线数据采集工具 (2020 - 至今)")
    parser.add_argument("--token", type=str, help="Tushare API Token")
    parser.add_argument("--min_date", type=str, default="2020-01-01", help="起始日期 (默认 2020-01-01)")
    parser.add_argument("--limit", type=int, default=0, help="测试用限制债券下载数量 (0 表示不限制)")
    parser.add_argument("--overwrite", action="store_true", help="强制覆盖已存在文件")
    args = parser.parse_args()

    ensure_directories()
    
    token = load_tushare_token(args.token)
    if not token:
        logging.error("ERROR: 未找到有效的 TUSHARE_TOKEN。")
        sys.exit(1)

    logging.info("初始化 Tushare Client 及 TDX 高速引擎...")
    ts.set_token(token)
    pro = ts.pro_api()

    try:
        bonds_df = get_all_convertible_bonds(pro)
    except Exception as e:
        logging.error(f"获取可转债列表失败: {e}")
        sys.exit(1)

    bonds_meta_path = os.path.join(DATA_DIR, "cb_basic_info.csv")
    bonds_df.to_csv(bonds_meta_path, index=False, encoding='utf-8-sig')

    bonds_list = bonds_df.to_dict('records')
    if args.limit > 0:
        bonds_list = bonds_list[:args.limit]

    total_bonds = len(bonds_list)
    success_count = 0

    tdx_api = connect_tdx()

    for idx, bond in enumerate(bonds_list, 1):
        ts_code = bond['ts_code']
        bond_name = bond.get('bond_short_name', '')
        
        parquet_file = os.path.join(PARQUET_DIR, f"{ts_code}.parquet")
        if not args.overwrite and os.path.exists(parquet_file):
            try:
                existing_df = pd.read_parquet(parquet_file)
                if len(existing_df) >= 30000:
                    logging.info(f"[{idx}/{total_bonds}] {ts_code} ({bond_name}) 已有全量历史 ({len(existing_df)} 条)，跳过。")
                    success_count += 1
                    continue
            except Exception:
                pass

        logging.info(f"[{idx}/{total_bonds}] 正在拉取 2020 至今全量 5分钟线: {ts_code} ({bond_name})...")
        
        try:
            df_mins = fetch_full_history_5min(tdx_api, ts_code, min_date=args.min_date)
            if save_bond_data(ts_code, df_mins):
                success_count += 1
        except Exception as e:
            logging.warning(f"[{ts_code}] 处理失败，断线重连: {e}")
            try:
                tdx_api = connect_tdx()
            except Exception:
                pass
                
        time.sleep(0.05)

    try:
        tdx_api.disconnect()
    except Exception:
        pass

    logging.info(f"采集完成！成功处理: {success_count}/{total_bonds} 只可转债 2020 至今全量 5分钟 K 线。数据保存在: {DATA_DIR}")

if __name__ == "__main__":
    main()
