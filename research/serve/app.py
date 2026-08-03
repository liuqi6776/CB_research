# -*- coding: utf-8 -*-
"""
量化策略每日信号 Dashboard (research/serve)

启动:  python research/serve/app.py   (默认 http://127.0.0.1:8000)
接口:
  GET /            仪表盘主页
  GET /api/today   今日信号 (最新 daily/*.json)
  GET /api/history 历史信号列表
  GET /api/picks?date=YYYY-MM-DD  指定日期的持仓明细
"""
import os
import json
import glob

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

SERVE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SERVE_DIR, "data", "daily")
ASSETS_DIR = os.path.join(SERVE_DIR, "assets")

app = FastAPI(title="Quant Daily Signal", version="1.0")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")


def _load_daily():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))
    out = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            d["file_date"] = os.path.splitext(os.path.basename(fp))[0]
            out.append(d)
        except Exception:
            continue
    out.sort(key=lambda x: x["file_date"], reverse=True)
    return out


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(SERVE_DIR, "templates", "index.html"), "r", encoding="utf-8") as fh:
        return fh.read()


@app.get("/api/today")
def api_today():
    daily = _load_daily()
    if not daily:
        raise HTTPException(404, "暂无信号, 请先运行 daily_signal.py")
    return JSONResponse(daily[0])


@app.get("/api/history")
def api_history(limit: int = 90):
    daily = _load_daily()
    return JSONResponse(daily[:limit])


@app.get("/api/picks")
def api_picks(date: str):
    fp = os.path.join(DATA_DIR, f"{date}.json")
    if not os.path.exists(fp):
        raise HTTPException(404, f"无 {date} 记录")
    with open(fp, "r", encoding="utf-8") as fh:
        return JSONResponse(json.load(fh))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
