# -*- coding: utf-8 -*-

"""
Unit test to verify liquidity & volume capacity limit (5% bar volume constraint)
Verifies exact unit matching between 15-minute K-line volume (bar_vol, measured in lots/手, 1 lot = 10 contracts/张)
and position trade volume (shares, measured in contracts/张, 100 RMB par value per contract).
"""

import unittest
import numpy as np
import pandas as pd

class TestVolumeCapacityConstraint(unittest.TestCase):
    
    def test_volume_unit_matching(self):
        """
        Verify that 5% 15-minute K-line volume limit:
        1 lot (手) = 10 contracts (张).
        If 15m K-line volume bar_vol = 100 lots (手) = 1,000 contracts (张).
        Max allowed trade size = 5% of 1,000 contracts = 50 contracts (张).
        Formula check:
        15m_bar_vol_contracts = bar_vol * 10
        max_allowed_contracts = 15m_bar_vol_contracts * 0.05 = bar_vol * 0.5
        Equivalent formula: bar_vol * 0.05 >= shares / 10
        """
        bar_vol_lots = 100  # 100 手
        shares_contracts = 50  # 50 张
        
        # 1. Direct physical contract calculation
        bar_vol_contracts = bar_vol_lots * 10
        max_allowed_contracts = bar_vol_contracts * 0.05
        
        self.assertEqual(bar_vol_contracts, 1000)
        self.assertEqual(max_allowed_contracts, 50.0)
        
        # 2. Formula test (bar_vol * 0.05 >= shares / 10)
        formula_left = bar_vol_lots * 0.05  # 5.0
        formula_right = shares_contracts / 10  # 5.0
        
        self.assertTrue(formula_left >= formula_right, "Shares within 5% volume limit should be allowed.")
        
        # 3. Test exceeding limit (e.g. 51 contracts)
        exceeding_shares = 51
        self.assertFalse(formula_left >= (exceeding_shares / 10), "Shares exceeding 5% volume limit must be rejected.")

if __name__ == '__main__':
    unittest.main()
