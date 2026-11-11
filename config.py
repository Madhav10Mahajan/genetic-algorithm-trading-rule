"""Shared settings for the whole experiment. Change values here to re-run with different choices."""

TICKER = "SPY"                           # S&P 500 ETF: owning it ~ owning the US large-cap market
DOWNLOAD_START = "2009-01-01"            # one extra year so a 250-day average is ready on the first day of 2010
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"   # the years the search methods learn from
TEST_START, TEST_END = "2020-01-01", "2024-12-31"     # hidden until the final exam, scored once

COST = 0.0005                            # 0.05% transaction cost, charged every time we buy or sell
TRADING_DAYS = 252                       # trading days per year, used to annualise (x sqrt(252) for Sharpe)
MAX_DRAWDOWN_LIMIT = -0.20               # drawdown we would tolerate; reported, not optimised

# Search space: the two unknowns of the moving-average crossover rule (fast must be < slow)
FAST_RANGE = (5, 50)                     # fast moving-average window, in days (inclusive)
SLOW_RANGE = (20, 250)                   # slow moving-average window, in days (inclusive)

# Genetic algorithm
POP_SIZE = 50                            # pairs per generation
N_GENERATIONS = 30                       # 50 x 30 = 1,500 fitness evaluations
TOURNAMENT_SIZE = 2                      # pick 2 at random, the fitter one becomes a parent
ELITE = 2                                # best pairs copied unchanged into the next generation
MUTATION_RATE = 0.2                      # chance that each number in a child gets nudged
MUTATION_STEP = 0.1                      # nudge size: std dev as a share of that number's range

# Random search gets exactly the same budget as the GA so the comparison is fair
N_RANDOM_TRIES = POP_SIZE * N_GENERATIONS

N_RUNS = 10                              # repeat GA and random search with different seeds (luck check)
SEED = 42                                # run i uses seed SEED + i; run 0 is the headline result
