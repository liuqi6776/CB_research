# -*- coding: utf-8 -*-
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings
import tushare as ts

pro = ts.pro_api(settings.TUSHARE_TOKEN)
# 测试: fina_indicator 不带 ts_code, 带 ann_date 范围
try:
    fi = pro.fina_indicator(ann_date="20200331", fields="ts_code,ann_date,netprofit_yoy")
    print("fina_indicator ann_date 单日 rows:", len(fi), "cols:", fi.columns.tolist())
except Exception as e:
    print("ann_date单日失败:", e)
try:
    fi2 = pro.fina_indicator(start_date="20200301", end_date="20200430", fields="ts_code,ann_date,netprofit_yoy")
    print("fina_indicator 范围 rows:", len(fi2))
except Exception as e:
    print("范围失败:", e)
