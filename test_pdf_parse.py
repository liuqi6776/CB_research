import sys
import requests
import pdfplumber
import io
import re

sys.stdout.reconfigure(encoding='utf-8')

url = "http://static.cninfo.com.cn/finalpage/2026-07-25/1225440419.PDF"
resp = requests.get(url)
pdf = pdfplumber.open(io.BytesIO(resp.content))
text = "\n".join([p.extract_text() or "" for p in pdf.pages])

print("Extracted text length:", len(text))
dates = re.findall(r'(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)', text)
prices = re.findall(r'(\d+\.\d+)\s*元/?股', text)

print("Found Dates:", dates[:5])
print("Found Prices:", prices[:5])
