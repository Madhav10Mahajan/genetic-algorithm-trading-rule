"""
Checks the code against section 5 of the project guide: 15 made-up prices, windows of
2-6 days, no transaction costs. Every number below was worked out by hand in the guide,
so if these pass the backtester and the GA operators do exactly what the guide describes.

    python -m pytest tests
"""
import numpy as np
import pandas as pd
import pytest

from backtest import Backtester, summarize
from ga import crossover, repair, run_ga
from random_search import run_random_search

PRICES = [100, 102, 101, 104, 106, 105, 103, 100, 98, 99, 102, 105, 107, 106, 109]
FAST, SLOW = (1, 3), (4, 6)


@pytest.fixture
def bt():
    dates = pd.bdate_range("2024-01-01", periods=len(PRICES))
    return Backtester(pd.Series(PRICES, index=dates, dtype=float), cost=0.0)


def daily(bt, pair):
    return bt.evaluate(pair, periods=1)   # the guide uses daily (not annualised) Sharpe


def test_section_5a_moving_averages_and_positions(bt):
    assert bt.moving_average(2)[3] == pytest.approx(102.5)    # day 4 fast MA
    assert bt.moving_average(4)[3] == pytest.approx(101.75)   # day 4 slow MA
    expected_signal = [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    expected_position = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
    assert bt.signal(2, 4).tolist() == expected_signal
    assert bt.positions(2, 4).tolist() == expected_position


def test_section_5a_strategy_returns_sharpe_and_drawdown(bt):
    r = bt.strategy_returns(2, 4)
    assert len(r) == 14                                        # days 2-15
    assert np.round(r.to_numpy() * 100, 2).tolist() == [0, 0, 0, 1.92, -0.94, -1.90, 0, 0, 0, 0,
                                                        2.94, 1.90, -0.93, 2.83]
    s = daily(bt, (2, 4))
    assert r.mean() * 100 == pytest.approx(0.416, abs=1e-3)
    assert r.std() * 100 == pytest.approx(1.44, abs=5e-3)
    assert s["Sharpe"] == pytest.approx(0.288, abs=1e-3)
    assert s["Max DD"] == pytest.approx(-0.0283, abs=1e-4)
    assert s["Final $"] == pytest.approx(105.84, abs=1e-2)
    assert s["Sharpe"] * np.sqrt(252) == pytest.approx(4.58, abs=1e-2)   # section 6: yearly Sharpe


def test_section_5b_buy_and_hold(bt):
    s = summarize(bt.buy_and_hold_returns().to_numpy(), periods=1)
    assert s["Sharpe"] == pytest.approx(0.297, abs=1e-3)
    assert s["Max DD"] == pytest.approx(-0.0755, abs=1e-4)
    assert s["Final $"] == pytest.approx(109.0)


def test_section_5c_every_pair(bt):
    expected = pd.DataFrame([[0.288, 0.202, 0.259],
                             [0.288, 0.043, -0.053],
                             [-0.002, -0.100, -0.053]], index=[1, 2, 3], columns=[4, 5, 6])
    grid = bt.sharpe_grid(FAST, SLOW, periods=1)
    np.testing.assert_allclose(grid.to_numpy(), expected.to_numpy(), atol=1e-3)


def test_section_5f_random_search_best(bt):
    s = daily(bt, (1, 6))
    assert s["Final $"] == pytest.approx(104.83, abs=1e-2)
    assert s["Max DD"] == pytest.approx(-0.0190, abs=1e-4)


def test_section_5e_crossover():
    c, d = (1, 5), (3, 4)
    assert crossover(c, d) == ((1, 4), (3, 5))


def test_repair_keeps_fast_below_slow():
    assert repair((60, 30)) == (30, 50)        # clipped to fast <= 50, then swapped
    assert repair((30, 30)) == (30, 31)
    assert repair((3, 100), (1, 3), (4, 6)) == (3, 6)


def test_ga_and_random_search_find_toy_optimum(bt):
    fitness = lambda pair: bt.sharpe(pair, periods=1)
    ga = run_ga(fitness, seed=0, pop_size=4, n_generations=5, fast_range=FAST, slow_range=SLOW)
    rs = run_random_search(fitness, seed=0, n_tries=20, batch=4, fast_range=FAST, slow_range=SLOW)
    assert ga.best_fitness == pytest.approx(0.288, abs=1e-3)
    assert rs.best_fitness == pytest.approx(0.288, abs=1e-3)
    assert ga.n_evaluations == 20 and len(ga.history) == 5
    fits = [h["best_fitness"] for h in ga.history]
    assert fits == sorted(fits)                # elitism: the best never gets worse
