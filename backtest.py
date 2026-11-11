"""
STEPS 4-7: The trading rule, the backtester, the two scores and the buy-and-hold benchmark.

Rule (moving-average crossover, two unknowns: fast < slow days)
  fast MA > slow MA at today's close  -> signal 1 (own SPY)
  otherwise                           -> signal 0 (hold cash, earns 0)

Backtester
  1. daily return       r_t = P_t / P_(t-1) - 1
  2. position_t         = signal_(t-1)   -> decide at today's close, hold from tomorrow
                                            (no look-ahead bias: we never use a price we did not have yet)
  3. strategy return    = position_t * r_t - COST * |position_t - position_(t-1)|
  The signal is computed on the full price history and the returns are then cut to a
  period, so on 1 Jan 2020 the averages already "remember" late 2019, exactly as they
  would have for a real trader.

Scores (section 4 of the guide)
  Sharpe        = mean(daily) / std(daily, ddof=1) * sqrt(252)   (risk-free rate taken as 0)
  Max drawdown  = min over time of (equity / running peak - 1)
  CAGR          = yearly growth rate of the equity curve
  Calmar        = CAGR / |max drawdown|                          (return per unit of pain)

Assumptions
  - We trade at the close the signal was computed on (the guide's convention); slippage ignored.
  - Cash earns 0 and there is no shorting or leverage - position is always 0 or 1.
"""
import numpy as np
import pandas as pd

from config import COST, TRADING_DAYS


def sharpe(r, periods=TRADING_DAYS):
    """Annualised Sharpe of daily returns. periods=1 gives the daily Sharpe used in the worked example."""
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    if len(r) < 2 or sd < 1e-12:   # never invested -> no risk and no return; score it as 0
        return 0.0
    return r.mean() / sd * np.sqrt(periods)


def equity_curve(r, start=100.0):
    """Value of `start` dollars grown by each day's return, with the starting value in front."""
    return start * np.concatenate([[1.0], np.cumprod(1 + np.asarray(r, dtype=float))])


def drawdowns(r):
    """Fall from the running peak on each day (0 at a new high); the starting $100 counts as a peak."""
    equity = equity_curve(r)
    return (equity / np.maximum.accumulate(equity) - 1)[1:]


def max_drawdown(r):
    return drawdowns(r).min()


def summarize(r, periods=TRADING_DAYS):
    equity = equity_curve(r)
    mdd = max_drawdown(r)
    cagr = (equity[-1] / equity[0]) ** (periods / len(r)) - 1
    return {
        "Sharpe": sharpe(r, periods),
        "Max DD": mdd,
        "CAGR": cagr,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
        "Final $": equity[-1],
        "Days": len(r),
    }


class Backtester:
    """Holds one price series and scores any (fast, slow) pair on any date range.

    Moving averages are cached per window, so scoring thousands of pairs only computes
    each average once.
    """

    def __init__(self, prices, cost=COST):
        self.prices = prices
        self.dates = prices.index
        self.cost = cost
        self.returns = prices.pct_change().to_numpy()   # first day has no previous price -> NaN
        self._ma = {}

    def moving_average(self, n):
        if n not in self._ma:
            self._ma[n] = self.prices.rolling(n).mean().to_numpy()
        return self._ma[n]

    def signal(self, fast, slow):
        # comparisons with NaN are False, so we hold cash until both averages exist
        return (self.moving_average(fast) > self.moving_average(slow)).astype(float)

    def positions(self, fast, slow):
        pos = np.zeros(len(self.prices))
        pos[1:] = self.signal(fast, slow)[:-1]          # today's signal -> tomorrow's position
        return pos

    def _period(self, start, end):
        lo = 0 if start is None else self.dates.searchsorted(pd.Timestamp(start))
        hi = len(self.dates) if end is None else self.dates.searchsorted(pd.Timestamp(end), side="right")
        return slice(max(lo, 1), hi)                     # day 0 never has a return

    def strategy_returns(self, fast, slow, start=None, end=None):
        pos = self.positions(fast, slow)
        trades = np.abs(np.diff(pos, prepend=0.0))       # 1 on every day we buy or sell
        r = pos * self.returns - self.cost * trades
        sl = self._period(start, end)
        return pd.Series(r[sl], index=self.dates[sl])

    def buy_and_hold_returns(self, start=None, end=None):
        sl = self._period(start, end)
        return pd.Series(self.returns[sl], index=self.dates[sl])

    def sharpe(self, pair, start=None, end=None, periods=TRADING_DAYS):
        return sharpe(self.strategy_returns(*pair, start, end).to_numpy(), periods)

    def evaluate(self, pair, start=None, end=None, periods=TRADING_DAYS):
        return summarize(self.strategy_returns(*pair, start, end).to_numpy(), periods)

    def sharpe_grid(self, fast_range, slow_range, start=None, end=None, periods=TRADING_DAYS):
        """Sharpe of every valid pair - the full map the search methods are exploring."""
        fasts = range(fast_range[0], fast_range[1] + 1)
        slows = range(slow_range[0], slow_range[1] + 1)
        grid = pd.DataFrame(np.nan, index=pd.Index(fasts, name="fast"), columns=pd.Index(slows, name="slow"))
        for f in fasts:
            for s in slows:
                if f < s:
                    grid.loc[f, s] = self.sharpe((f, s), start, end, periods)
        return grid
