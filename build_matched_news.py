# -*- coding: utf-8 -*-
"""Step A: 原始新闻 → 个股匹配 + 关键词情感打分 (全量108天 parquet)

输出: matched_news_scored.csv
  pub_dt, date, ts_code, title, pos, neg, net

匹配: Aho-Corasick, 标题+正文, 最早出现优先, 平局取最长名称 (复刻原研究规则)
打分: 自重建关键词词典, 标题权重x2, 正文权重x1 (后续用样本内标签校准)
"""
import pandas as pd
import numpy as np
import ahocorasick
import glob, os, re, time

root = r"C:\Users\liuqi\quant_system_v2"
t0 = time.time()

# ---------- 1. 读新闻 ----------
files = sorted(glob.glob(f"{root}/news_raw_oos/*.parquet"))
import duckdb
con = duckdb.connect()
news = con.execute(
    f"SELECT * FROM read_parquet('{root}/news_raw_oos/*.parquet')"
).df()
print(f"新闻总量: {len(news)}, 列: {list(news.columns)}")
news["pub_dt"] = pd.to_datetime(news["datetime"])
news["date"] = news["pub_dt"].dt.strftime("%Y%m%d").astype(int)
news["text"] = (news["title"].fillna("") + " " + news["content"].fillna("")).str[:800]

# ---------- 2. 名称自动机 ----------
fmap = pd.read_csv(f"{root}/company_name_fuzzy_map_final.csv")
fmap = fmap.dropna()
fmap["name_variant"] = fmap["name_variant"].astype(str)
fmap = fmap[fmap["name_variant"].str.len() >= 2]
# 一个变体只映射一个代码(若冲突取第一个)
fmap = fmap.drop_duplicates("name_variant")
A = ahocorasick.Automaton()
for name, code in zip(fmap["name_variant"], fmap["ts_code"]):
    A.add_word(name, (name, code))
A.make_automaton()
print(f"名称变体: {len(fmap)}")

def match_one(text):
    best = None  # (start, -len, name, code)
    for end, (name, code) in A.iter(text):
        start = end - len(name) + 1
        key = (start, -len(name))
        if best is None or key < best[0]:
            best = (key, name, code)
    if best is None:
        return None, None
    return best[1], best[2]

matches = [match_one(t) for t in news["text"]]
news["m_name"] = [m[0] for m in matches]
news["ts_code"] = [m[1] for m in matches]
matched = news.dropna(subset=["ts_code"]).copy()
print(f"匹配到个股: {len(matched)} / {len(news)} ({len(matched)/len(news)*100:.1f}%), 耗时 {time.time()-t0:.0f}s")

# ---------- 3. 关键词打分 ----------
POS = ["涨停","大涨","利好","增长","预增","盈利","突破","中标","签约","订单","回购","增持",
       "分红","派息","获批","落地","受益","扩产","满产","提价","涨价","复苏","超预期","扭亏",
       "重组","并购","战略合作","量产","交付","热销","供不应求","回暖","改善","新高","创新高",
       "净流入","翻倍","暴发","爆发","强劲","亮眼","创纪录","提价","红利","景气"]
NEG = ["跌停","大跌","利空","亏损","预亏","下滑","下降","减持","质押","违规","处罚","立案",
       "调查","退市","破产","违约","暴雷","爆雷","商誉减值","计提","停产","召回","投诉","诉讼",
       "仲裁","冻结","警示","低于预期","下调","降级","解禁","抛售","缩水","新低","创新低",
       "疲软","恶化","净流出","暴跌","重挫","风险警示","戴帽","ST","问询","谴责","处罚"]

pos_re = {w: re.compile(re.escape(w)) for w in POS}
neg_re = {w: re.compile(re.escape(w)) for w in NEG}

def score_row(title, content):
    p = sum(2 * len(r.findall(title)) for r in pos_re.values())
    n = sum(2 * len(r.findall(title)) for r in neg_re.values())
    p += sum(len(r.findall(content)) for r in pos_re.values())
    n += sum(len(r.findall(content)) for r in neg_re.values())
    return p, n

titles = matched["title"].fillna("")
contents = matched["content"].fillna("").str[:800]
scores = [score_row(t, c) for t, c in zip(titles, contents)]
matched["pos"] = [s[0] for s in scores]
matched["neg"] = [s[1] for s in scores]
matched["net"] = matched["pos"] - matched["neg"]
print(f"打分完成, 耗时 {time.time()-t0:.0f}s")

out = matched[["pub_dt", "date", "ts_code", "m_name", "title", "pos", "neg", "net"]]
out.to_csv(f"{root}/matched_news_scored.csv", index=False)
print(f"已保存 matched_news_scored.csv: {len(out)} 行")
print(out["date"].describe())
