"""Test script to reproduce the unit conversion bug."""

import pandas as pd
from attriblink import link

# Quarterly returns in percent (5%, -2%, 8%, 3%)
portfolio_returns = pd.Series([0.05, -0.02, 0.08, 0.03], name="portfolio")
benchmark_returns = pd.Series([0.03, 0.01, 0.05, 0.02], name="benchmark")

# Effects in bps (allocation=180, selection=120 = 1.8% and 1.2%)
# These sum to 3% (180+120=300 bps = 3%)
effects = pd.DataFrame({
    "allocation": [0.018, -0.005, 0.025, 0.012],  # in decimal
    "selection": [0.012, -0.015, 0.015, 0.003],   # in decimal
}, index=portfolio_returns.index)

print("=== Test 1: Using decimal (should work) ===")
result = link(effects, portfolio_returns, benchmark_returns, check_effects_sum=False)
print(result.summary())
print()

print("=== Test 2: Using unit='bps' with decimal inputs (BUG) ===")
# If user passes decimal values but specifies unit='bps', the code wrongly
# divides returns by 10000, causing the cumulative calc to fail
portfolio_bps = pd.Series([500, -200, 800, 300], name="portfolio")  # 5%, -2%, 8%, 3% in bps
benchmark_bps = pd.Series([300, 100, 500, 200], name="benchmark")  # 3%, 1%, 5%, 2% in bps
effects_bps = pd.DataFrame({
    "allocation": [180, -50, 250, 120],  # in bps
    "selection": [120, -150, 150, 30],    # in bps
}, index=portfolio_returns.index)

result = link(effects_bps, portfolio_bps, benchmark_bps, unit="bps", check_effects_sum=False)
print(result.summary())
print()

# Check the data property
print("=== Data property ===")
print(result.data)
