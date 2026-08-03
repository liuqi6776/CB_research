import requests
import re

url = "https://data.eastmoney.com/kzz/detail/113052.html"
resp = requests.get(url)
js_files = re.findall(r'src=["\']([^"\']+\.js)["\']', resp.text)
print("JS Files:", js_files)

for js in js_files:
    if not js.startswith('http'):
        js = 'https:' + js if js.startswith('//') else 'https://data.eastmoney.com' + js
    try:
        t = requests.get(js).text
        reports = re.findall(r'RPT_[A-Z0-9_]+', t)
        if reports:
            print(f"Found in {js}: {set(reports)}")
    except Exception as e:
        pass
