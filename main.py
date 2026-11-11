"""
Runs the whole project end to end and writes tables and charts to results/.

    pip install -r requirements.txt
    python3 main.py

Order matters - it is what keeps the test honest:
  Steps 8-9   GA and random search see ONLY 2010-2019. Run 0 (seed SEED) is the headline pick.
  Step 10     The picks are frozen, then scored once on 2020-2024 next to buy-and-hold.
  Step 11     Overfitting check. Everything here reads test data only to *describe* the frozen
              picks (and the whole search space); none of it feeds back into a choice.
  Step 12     Charts.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import charts
from backtest import Backtester, drawdowns, equity_curve, summarize
from config import (FAST_RANGE, MAX_DRAWDOWN_LIMIT, N_RUNS, SEED, SLOW_RANGE, TEST_END, TEST_START,
                    TRAIN_END, TRAIN_START)
from data import load_prices
from ga import run_ga
from random_search import run_random_search

RESULTS = Path(__file__).parent / "results"
TRAIN, TEST = (TRAIN_START, TRAIN_END), (TEST_START, TEST_END)


def main():
    RESULTS.mkdir(exist_ok=True)
    bt = Backtester(load_prices())
    fitness = lambda pair: bt.sharpe(pair, *TRAIN)    # the search methods never see anything else

    # ---- Steps 8-9: search on training data, N_RUNS seeds each ----
    ga_runs = [run_ga(fitness, SEED + i) for i in range(N_RUNS)]
    rs_runs = [run_random_search(fitness, SEED + i) for i in range(N_RUNS)]
    ga_pick, rs_pick = ga_runs[0].best_pair, rs_runs[0].best_pair
    print(f"[Steps 8-9] {ga_runs[0].n_evaluations:,} evaluations per run "
          f"(GA tried {len(ga_runs[0].evaluated)} distinct pairs, random search {len(rs_runs[0].evaluated)})")
    print(f"            GA pick {ga_pick}: training Sharpe {ga_runs[0].best_fitness:.3f} | "
          f"random-search pick {rs_pick}: {rs_runs[0].best_fitness:.3f}")

    # ---- Step 10: final exam, once ----
    returns = {
        "Buy-and-hold": (bt.buy_and_hold_returns(*TRAIN), bt.buy_and_hold_returns(*TEST)),
        f"GA {ga_pick}": (bt.strategy_returns(*ga_pick, *TRAIN), bt.strategy_returns(*ga_pick, *TEST)),
        f"Random search {rs_pick}": (bt.strategy_returns(*rs_pick, *TRAIN), bt.strategy_returns(*rs_pick, *TEST)),
    }
    rows = {}
    for name, (r_train, r_test) in returns.items():
        tr, te = summarize(r_train.to_numpy()), summarize(r_test.to_numpy())
        rows[name] = {
            "Train Sharpe": tr["Sharpe"], "Test Sharpe": te["Sharpe"], "Sharpe drop": tr["Sharpe"] - te["Sharpe"],
            "Train Max DD": tr["Max DD"], "Test Max DD": te["Max DD"], "Test CAGR": te["CAGR"],
            "Test Calmar": te["Calmar"], "Test $100 ->": te["Final $"],
            "Test DD within limit": te["Max DD"] >= MAX_DRAWDOWN_LIMIT,
        }
    table = pd.DataFrame(rows).T
    table.index.name = "Strategy"
    table.to_csv(RESULTS / "results_table.csv", float_format="%.4f")
    print(f"\n=== [Step 10] FINAL EXAM: train {TRAIN_START[:4]}-{TRAIN_END[:4]}, "
          f"test {TEST_START[:4]}-{TEST_END[:4]} (scored once) ===")
    print(_format(table).to_string())

    # ---- Step 11: overfitting check ----
    runs = pd.DataFrame(
        [{"Method": "GA", "Seed": SEED + i, "Pair": r.best_pair, "Train Sharpe": r.best_fitness,
          "Test Sharpe": bt.sharpe(r.best_pair, *TEST)} for i, r in enumerate(ga_runs)]
        + [{"Method": "Random search", "Seed": SEED + i, "Pair": r.best_pair, "Train Sharpe": r.best_fitness,
            "Test Sharpe": bt.sharpe(r.best_pair, *TEST)} for i, r in enumerate(rs_runs)])
    runs.to_csv(RESULTS / "runs_by_seed.csv", index=False, float_format="%.4f")

    grid_train = bt.sharpe_grid(FAST_RANGE, SLOW_RANGE, *TRAIN)
    grid_test = bt.sharpe_grid(FAST_RANGE, SLOW_RANGE, *TEST)
    grid_train.to_csv(RESULTS / "sharpe_grid_train.csv", float_format="%.4f")
    grid_test.to_csv(RESULTS / "sharpe_grid_test.csv", float_format="%.4f")
    flat = pd.DataFrame({"train": grid_train.stack(), "test": grid_test.stack()}).dropna()   # only fast < slow
    best_train_pair = flat["train"].idxmax()
    bh_train, bh_test = table.loc["Buy-and-hold", ["Train Sharpe", "Test Sharpe"]]

    ga_generations = [{"generation": h["generation"], "pair": h["best_pair"], "train": h["best_fitness"],
                       "test": bt.sharpe(h["best_pair"], *TEST)} for h in ga_runs[0].history]
    pd.DataFrame(ga_generations).to_csv(RESULTS / f"ga_generations_seed{SEED}.csv", index=False, float_format="%.4f")

    print("\n=== [Step 11] OVERFITTING CHECK ===")
    print(runs.groupby("Method").agg(
        distinct_pairs=("Pair", "nunique"), train_mean=("Train Sharpe", "mean"), test_mean=("Test Sharpe", "mean"),
        test_min=("Test Sharpe", "min"), test_max=("Test Sharpe", "max")).round(3).to_string())
    print(f"GA winners found the best possible training pair {best_train_pair} "
          f"in {sum(r.best_pair == best_train_pair for r in ga_runs)}/{N_RUNS} seeds "
          f"(random search: {sum(r.best_pair == best_train_pair for r in rs_runs)}/{N_RUNS})")
    print(f"Across all {len(flat):,} pairs: rank correlation of train vs test Sharpe = "
          f"{flat.rank().corr().iloc[0, 1]:.2f}")   # Spearman = Pearson on ranks
    print(f"GA pick's test Sharpe beats {(flat['test'] < flat.loc[ga_pick, 'test']).mean():.0%} of all pairs; "
          f"{(flat['test'] > bh_test).mean():.1%} of all pairs beat buy-and-hold on test")

    # ---- Step 12: charts ----
    charts.plot_convergence(ga_runs, rs_runs, flat["train"].max(), RESULTS / "1_convergence.png")
    ga_test = returns[f"GA {ga_pick}"][1]
    curves = {name: (pd.Series(equity_curve(r_test)[1:], index=r_test.index), color)
              for (name, (_, r_test)), color in zip(returns.items(), (charts.BH_COLOR, charts.GA_COLOR, charts.RS_COLOR))}
    in_cash = pd.Series(bt.positions(*ga_pick), index=bt.dates).loc[ga_test.index] == 0
    charts.plot_equity(curves, in_cash, RESULTS / "2_equity_test.png",
                       f"$100 invested on the test years {TEST_START[:4]}-{TEST_END[:4]} (after 0.05% costs)")
    charts.plot_overfitting(grid_train, grid_test,
                            ga_points=list(zip(runs.query("Method == 'GA'")["Train Sharpe"],
                                               runs.query("Method == 'GA'")["Test Sharpe"])),
                            rs_points=list(zip(runs.query("Method != 'GA'")["Train Sharpe"],
                                               runs.query("Method != 'GA'")["Test Sharpe"])),
                            bh_point=(bh_train, bh_test), ga_generations=ga_generations, bh_test=bh_test, seed=SEED,
                            path=RESULTS / "3_overfitting.png")
    charts.plot_sharpe_maps(grid_train, grid_test, ga_pick, RESULTS / "4_sharpe_maps.png")
    charts.plot_drawdowns(
        {name: (color, [pd.Series(drawdowns(r), index=r.index) for r in period_returns])
         for (name, period_returns), color in zip(returns.items(), (charts.BH_COLOR, charts.GA_COLOR, charts.RS_COLOR))},
        [f"Drawdown - training {TRAIN_START[:4]}-{TRAIN_END[:4]}", f"Drawdown - test {TEST_START[:4]}-{TEST_END[:4]}"],
        RESULTS / "5_drawdowns.png")
    print(f"\n[Step 12] tables and charts written to {RESULTS}/")


def _format(table):
    out = table.copy()
    for col in ["Train Sharpe", "Test Sharpe", "Sharpe drop", "Test Calmar"]:
        out[col] = out[col].map(lambda v: f"{v:.2f}")
    for col in ["Train Max DD", "Test Max DD", "Test CAGR"]:
        out[col] = out[col].map(lambda v: f"{v:.1%}")
    out["Test $100 ->"] = out["Test $100 ->"].map(lambda v: f"${v:.0f}")
    return out


if __name__ == "__main__":
    main()
