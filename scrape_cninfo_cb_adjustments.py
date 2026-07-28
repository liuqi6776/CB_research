# -*- coding: utf-8 -*-

"""
全网可转债历史转股价调整（下修/修正/分红调整）事件表爬虫
CNINFO (巨潮资讯) 官方历史公告 API 抓取与 PIT 时间戳事件表构建器
"""

import os
import re
import time
import requests
import pandas as pd
import numpy as np

def scrape_cninfo_cb_adjustments():
    print("=== 开始从 CNINFO (巨潮资讯) 全网爬取 2020-2026 年可转债转股价调整公告事件流 ===")
    
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }
    
    keywords = ["向下修正转股价格", "转股价格调整", "修正转股价格"]
    records = []
    
    for kw in keywords:
        print(f"正在抓取关键字: [{kw}] ...")
        for page in range(1, 15):
            data = {
                'pageNum': page,
                'pageSize': 30,
                'column': 'szse',
                'tabName': 'fulltext',
                'searchkey': kw,
                'plate': '',
                'stock': '',
                'category': '',
                'type': '',
                'sortName': '',
                'sortType': '',
                'limit': '',
                'showTitle': ''
            }
            try:
                resp = requests.post(url, data=data, headers=headers, timeout=10)
                if resp.status_code == 200:
                    res_json = resp.json()
                    anns = res_json.get('announcements', [])
                    if not anns:
                        break
                    
                    for a in anns:
                        sec_code = a.get('secCode', '')
                        sec_name = a.get('secName', '')
                        title = a.get('announcementTitle', '')
                        pub_ts = a.get('announcementTime', 0)
                        
                        if pub_ts > 0:
                            pub_date = pd.to_datetime(pub_ts, unit='ms').strftime('%Y%m%d')
                            records.append({
                                'stk_code': sec_code,
                                'stk_name': sec_name,
                                'pub_date': pub_date,
                                'title': title,
                                'keyword': kw
                            })
                time.sleep(0.3)
            except Exception as e:
                print(f"抓取第 {page} 页失败: {e}")
                
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        print("未获取到调整公告记录。")
        return
        
    df_clean = df_raw.drop_duplicates(subset=['stk_code', 'pub_date', 'title']).sort_values(by=['stk_code', 'pub_date']).reset_index(drop=True)
    print(f"全网共抓取到 {len(df_clean):,} 条可转债转股价调整公告记录。")
    
    out_csv = r"D:\CB_mins_data\cb_conv_price_history.csv"
    df_clean.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"已成功将历史转股价调整事件表写入: {out_csv}")
    print(df_clean.head(10))

if __name__ == '__main__':
    scrape_cninfo_cb_adjustments()
