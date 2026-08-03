@echo off
rem 量化策略每日信号 Dashboard - 开机自启 (常驻 127.0.0.1:8000)
cd /d C:\Users\liuqi\quant_system_v2
start "" "C:\Users\liuqi\anaconda3\python.exe" "C:\Users\liuqi\quant_system_v2\research\serve\app.py"
