"""
The guide's worked example (README section 4), run through the real backtester.

15 made-up prices, windows of 2-6 days, no transaction costs. Prints the numbers worked out
by hand and draws results/0_worked_example.png - $100 under buy-and-hold, the GA's pick (1, 4)
and random search's pick (1, 6), shaded on the days the GA's pick sat in cash.

    python3 worked_example.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import charts
from backtest import Backtester, equity_curve, summarize

PRICES = [100, 102, 101, 104, 106, 105, 103, 100, 98, 99, 102, 105, 107, 106, 109]
GA_PICK, RS_PICK = (1, 4), (1, 6)   # best pairs found by hand in sections 4.4-4.5
RESULTS = Path(__file__).parent / "results"


def main():
    days = np.arange(1, len(PRICES) + 1)
    bt = Backtester(pd.Series(PRICES, index=pd.bdate_range("2024-01-01", periods=len(PRICES)), dtype=float),
                    cost=0.0)

    strategies = {
        "Buy-and-hold": (bt.buy_and_hold_returns(), charts.BH_COLOR),
        f"GA's best {GA_PICK}": (bt.strategy_returns(*GA_PICK), charts.GA_COLOR),
        f"Random search's best {RS_PICK}": (bt.strategy_returns(*RS_PICK), charts.RS_COLOR),
    }
    table = pd.DataFrame({name: summarize(r.to_numpy(), periods=1) for name, (r, _) in strategies.items()}).T
    print("=== Worked example (daily Sharpe, no costs) ===")
    print(table[["Final $", "Sharpe", "Max DD"]].to_string(
        formatters={"Final $": "${:.2f}".format, "Sharpe": "{:.3f}".format, "Max DD": "{:.2%}".format}))

    print("\nDaily Sharpe of every pair:")
    print(bt.sharpe_grid((1, 3), (4, 6), periods=1).round(3).to_string())

    RESULTS.mkdir(exist_ok=True)
    charts.plot_worked_example(days, {name: (equity_curve(r.to_numpy()), color)   # day 1 = $100 start
                                      for name, (r, color) in strategies.items()},
                               in_cash=bt.positions(*GA_PICK) == 0, path=RESULTS / "0_worked_example.png")
    print(f"\nChart written to {RESULTS / '0_worked_example.png'}")


if __name__ == "__main__":
    main()
