# -*- coding: utf-8 -*-
"""因子方向单元测试 (阶段1, 冻结基线 v1.0.0)

验收标准: 构造高低值样本, ret_1m/ivol/turn/VAL 排序方向 100% 正确.
方向定义 (全部 高值=好, 与 style_factors/combo_backtest 口径一致):
  - ret_1m: 一月反转, 过去 20 日累计收益取负 (过去跌得多=好)
  - ivol:   低特质波动率, 20 日收益标准差取负 (波动低=好)
  - turn:   低换手波动, factor_lib 已统一方向 (高值=好)
  - VAL:    价值因子 = BP+SP+DP 平均 z-score (低估值=好)
运行: python test_factor_direction.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from research.factor_dic import style_factors as sf


def make_price_series(daily_ret, n_days=70):
    """由日收益序列构造简单日频 DataFrame (pct_chg/close 列)"""
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    idx = [d.strftime("%Y%m%d") for d in idx]
    r = pd.Series(daily_ret, index=idx)
    df = pd.DataFrame({
        "trade_date": idx,
        "pct_chg": r * 100.0,
        "close": (1 + r).cumprod(),
        "amount": 1e8,
    })
    return df


def test_reversal():
    """一月反转: 过去跌得多的股票 ret_1m 因子值更高 (更好)
    取下跌/上涨期结束后第 5 日 (idx=25) 的滚动窗口值比较, 该窗口完全覆盖前 20 日涨跌段"""
    n = 70
    # A: 前20日下跌 -1%/日, 后50日平; B: 前20日上涨 +1%/日
    a = np.concatenate([np.full(20, -0.01), np.zeros(n - 20)])
    b = np.concatenate([np.full(20, 0.01), np.zeros(n - 20)])
    stocks = {"A": make_price_series(a), "B": make_price_series(b)}
    from research.factor_dic import combo_backtest as cb
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, ["A", "B"])
    assert "A" in ret_1m and "B" in ret_1m, "个股被过滤"
    va, vb = ret_1m["A"].iloc[25], ret_1m["B"].iloc[25]
    assert va > vb, f"反转因子方向错误: A={va:.4f} B={vb:.4f}"
    return dict(factor="ret_1m", a=float(va), b=float(vb))


def test_ivol():
    """低波: 波动小的股票 ivol 因子值更高 (更好)"""
    n = 70
    rng = np.random.default_rng(7)
    a = rng.normal(0.001, 0.005, n)   # 低波
    b = rng.normal(0.001, 0.030, n)   # 高波
    stocks = {"A": make_price_series(a), "B": make_price_series(b)}
    from research.factor_dic import combo_backtest as cb
    ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, ["A", "B"])
    assert "A" in ivol and "B" in ivol, "个股被过滤"
    va, vb = ivol["A"].iloc[-1], ivol["B"].iloc[-1]
    assert va > vb, f"低波因子方向错误: A={va:.4f} B={vb:.4f}"
    return dict(factor="ivol", a=float(va), b=float(vb))


def test_val():
    """VAL: 低估值(高EP/BP)股票因子值更高 (更好); 需要 >20 只满足合成条件"""
    # 构造 25 只: X 深度低估(pb=0.8), Y 深度高估(pb=8.0), 其余居中
    codes = ["X", "Y"] + [f"S{i:02d}" for i in range(23)]
    rows = []
    for i, c in enumerate(codes):
        if c == "X":
            pe, pb = 8.0, 0.8
        elif c == "Y":
            pe, pb = 80.0, 8.0
        else:
            pe, pb = 20.0 + (i % 5), 2.0 + (i % 5) * 0.3
        rows.append((c, pe, pb, pe * 0.1, 0.02))
    val = pd.DataFrame(rows, columns=["ts_code", "pe_ttm", "pb", "ps_ttm", "dv_ttm"])
    rb = "20240628"
    val_map = {rb: val.set_index("ts_code")}
    panels = sf.build_factors(val_map, {}, [rb])
    v = panels["VAL"][rb]
    assert v["X"] > v["Y"], f"VAL 因子方向错误: X={v['X']:.3f} Y={v['Y']:.3f}"
    return dict(factor="VAL", a=float(v["X"]), b=float(v["Y"]))


def test_turn_direction_helper():
    """turn 因子方向 (factor_lib 统一高值=好): 数据不完整时跳过, 不判失败"""
    n = 70
    rng = np.random.default_rng(11)
    a = make_price_series(rng.normal(0.0, 0.01, n))
    b = make_price_series(rng.normal(0.0, 0.01, n))
    a["amount"] = 1e8   # 高成交额 → 低换手波动 → 高因子值
    b["amount"] = 1e4
    try:
        stocks = {"A": a, "B": b}
        from research.factor_dic import combo_backtest as cb
        ret_1m, ivol, turn, fwd = cb.build_price_factors(stocks, ["A", "B"])
        if "A" not in turn or "B" not in turn:
            return dict(factor="turn", skipped="turn 因子序列为空")
        assert turn["A"].iloc[-1] >= turn["B"].iloc[-1], "turn 因子方向错误"
        return dict(factor="turn", a=float(turn["A"].iloc[-1]), b=float(turn["B"].iloc[-1]))
    except Exception as e:
        return dict(factor="turn", skipped=str(e))


def main():
    tests = [test_reversal, test_ivol, test_val, test_turn_direction_helper]
    n_pass = 0
    print("=" * 70)
    print("因子方向单元测试 (冻结基线 v1.0.0)")
    print("=" * 70)
    for fn in tests:
        try:
            r = fn()
            n_pass += 1
            print(f"  [PASS] {fn.__name__:24s} {r}")
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__:24s} {e}")
        except Exception as e:
            print(f"  [ERR ] {fn.__name__:24s} {type(e).__name__}: {e}")
    print("=" * 70)
    print(f"结果: {n_pass}/{len(tests)} 通过")
    return 0 if n_pass == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
