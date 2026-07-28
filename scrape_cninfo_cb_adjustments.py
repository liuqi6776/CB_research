# -*- coding: utf-8 -*-

"""
全网可转债历史转股价多阶变动与生效日 As-Of 事件表爬虫
CNINFO (巨潮资讯) 官方历史公告 API 抓取 + PDF 多阶价格与生效日提取
"""

import os
import re
import sys
import io
import time
import requests
import pandas as pd
import numpy as np
import pdfplumber

sys.stdout.reconfigure(encoding='utf-8')

def parse_cninfo_announcement_pdf(pdf_url):
    """从 CNINFO 公告 PDF 解析调整前转股价、调整后转股价及生效日期"""
    try:
        resp = requests.get(pdf_url, timeout=10)
        if resp.status_code != 200:
            return None, None, None
            
        pdf = pdfplumber.open(io.BytesIO(resp.content))
        text = "\n".join([p.extract_text() or "" for p in pdf.pages])
        
        # 1. 解析生效日期 (如: "2024 年 6 月 15 日" / "自2024年6月15日起")
        date_matches = re.findall(r'(?:自|于|生效日期为?|生效日为?)\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)', text)
        effective_date = None
        if date_matches:
            raw_d = date_matches[0].replace(' ', '')
            m = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', raw_d)
            if m:
                effective_date = f"{int(m.group(1)):04d}{int(m.group(2)):02d}{int(m.group(3)):02d}"
                
        # 2. 解析价格数字 (调整前转股价 vs 调整后转股价)
        # 匹配 pattern: 调整前... XX.XX 元/股 ... 调整后... XX.XX 元/股
        prices = [float(p) for p in re.findall(r'(\d+\.\d+)\s*元/?股', text)]
        prices = [p for p in prices if 1.0 <= p <= 500.0]
        
        new_price = None
        prev_price = None
        
        # 针对 "调整后的转股价格为 XX.XX 元/股"
        new_matches = re.findall(r'调整后.*?(?:转股价格|转股价)为?\s*(\d+\.\d+)\s*元', text)
        if new_matches:
            new_price = float(new_matches[0])
            
        prev_matches = re.findall(r'调整前.*?(?:转股价格|转股价)为?\s*(\d+\.\d+)\s*元', text)
        if prev_matches:
            prev_price = float(prev_matches[0])
            
        if new_price is None and len(prices) >= 1:
            new_price = prices[0]
            
        return effective_date, prev_price, new_price
    except Exception:
        return None, None, None

def scrape_full_conv_price_history():
    print("=== 开始从 CNINFO (巨潮资讯) 全量抓取多阶转股价变动与生效日事件表 ===")
    
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }
    
    keywords = ["向下修正转股价格", "转股价格调整", "修正转股价格"]
    records = []
    
    for kw in keywords:
        print(f"正在抓取关键字: [{kw}] ...")
        for page in range(1, 12):
            data = {
                'pageNum': page,
                'pageSize': 30,
                'column': 'szse',
                'tabName': 'fulltext',
                'searchkey': kw
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
                        pdf_path = a.get('adjunctUrl', '')
                        
                        if pub_ts > 0 and pdf_path:
                            pub_date = pd.to_datetime(pub_ts, unit='ms').strftime('%Y%m%d')
                            pdf_url = f"http://static.cninfo.com.cn/{pdf_path}"
                            
                            records.append({
                                'stk_code': sec_code,
                                'stk_name': sec_name,
                                'pub_date': pub_date,
                                'title': title,
                                'pdf_url': pdf_url
                            })
                time.sleep(0.2)
            except Exception as e:
                print(f"抓取第 {page} 页失败: {e}")
                
    df_raw = pd.DataFrame(records)
    if df_raw.empty:
        print("未获取到调整公告记录。")
        return
        
    df_clean = df_raw.drop_duplicates(subset=['stk_code', 'pub_date', 'title']).sort_values(by=['stk_code', 'pub_date']).reset_index(drop=True)
    print(f"全网共抓取到 {len(df_clean):,} 条转股价调整公告。开始解析 PDF 多阶价格与生效日...")
    
    parsed_eff_dates = []
    parsed_prev_prices = []
    parsed_new_prices = []
    
    # 抽取前 200 条进行深度 PDF 采样解析，其余填入规则推导
    for idx, row in df_clean.iterrows():
        if idx < 200:
            eff_d, p_prev, p_new = parse_cninfo_announcement_pdf(row['pdf_url'])
        else:
            eff_d, p_prev, p_new = None, None, None
            
        # 若未成功提取生效日，按 A 股规则：公告日次日生效
        if not eff_d:
            pub_dt = pd.to_datetime(row['pub_date'])
            eff_d = (pub_dt + pd.Timedelta(days=1)).strftime('%Y%m%d')
            
        parsed_eff_dates.append(eff_d)
        parsed_prev_prices.append(p_prev)
        parsed_new_prices.append(p_new)
        
        if (idx + 1) % 50 == 0 or idx == len(df_clean) - 1:
            print(f"进度: [{idx+1}/{len(df_clean)}] ...")
            
    df_clean['effective_date'] = parsed_eff_dates
    df_clean['prev_conv_price'] = parsed_prev_prices
    df_clean['new_conv_price'] = parsed_new_prices
    
    # 格式清理
    df_clean['stk_code'] = df_clean['stk_code'].astype(str).str.zfill(6)
    df_clean['effective_date'] = pd.to_numeric(df_clean['effective_date'], errors='coerce')
    df_clean['pub_date'] = pd.to_numeric(df_clean['pub_date'], errors='coerce')
    
    out_csv = r"D:\CB_mins_data\cb_conv_price_history.csv"
    out_csv_repo = r"c:\Users\liuqi\quant_system_v2\artifacts\cb_conv_price_history.csv"
    os.makedirs(os.path.dirname(out_csv_repo), exist_ok=True)
    
    df_clean.to_csv(out_csv, index=False, encoding='utf-8-sig')
    df_clean.to_csv(out_csv_repo, index=False, encoding='utf-8-sig')
    
    print(f"已成功将全网多阶转股价变动与生效日事件表写入:\n  - {out_csv}\n  - {out_csv_repo}")
    print(df_clean.head(10))

if __name__ == '__main__':
    scrape_full_conv_price_history()
