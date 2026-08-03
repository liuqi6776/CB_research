import requests
import re

main_js_text = requests.get('https://data.eastmoney.com/newstatic/js/common/emdataview.js').text
reports = set(re.findall(r'RPT_[A-Z0-9_]+', main_js_text))
print("emdataview.js reports:", [r for r in reports if 'CB' in r or 'BOND' in r])

detail_js_text = requests.get('https://data.eastmoney.com/newstatic/js/kzz/detail.js').text
reports_detail = set(re.findall(r'RPT_[A-Z0-9_]+', detail_js_text))
print("detail.js reports:", [r for r in reports_detail if 'CB' in r or 'BOND' in r])
