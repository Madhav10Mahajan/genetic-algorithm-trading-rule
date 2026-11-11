"""
STEP 2: Download the data - daily adjusted closing prices of SPY from Yahoo Finance.

The first download is saved to data/SPY.csv and reused afterwards, so every run
(and anyone cloning the repo) works on exactly the same numbers.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DOWNLOAD_START, TEST_END, TICKER

CACHE = Path(__file__).parent / "data" / f"{TICKER}.csv"


def load_prices():
    if CACHE.exists():
        prices = pd.read_csv(CACHE, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        end = pd.Timestamp(TEST_END) + pd.Timedelta(days=1)  # yfinance's end date is exclusive
        # auto_adjust=True -> prices already adjusted for dividends and splits
        close = yf.download(TICKER, start=DOWNLOAD_START, end=end, auto_adjust=True, progress=False)["Close"]
        prices = close[TICKER] if isinstance(close, pd.DataFrame) else close
        prices = prices.dropna().rename(TICKER)
        CACHE.parent.mkdir(exist_ok=True)
        prices.to_csv(CACHE)
    print(f"[Step 2] {TICKER}: {prices.index[0].date()} -> {prices.index[-1].date()} | {len(prices)} daily closes")
    return prices
