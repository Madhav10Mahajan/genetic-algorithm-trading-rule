# Evolving a Trading Rule with a Genetic Algorithm

A genetic algorithm (GA) tunes a moving-average crossover rule on SPY over 2010–2019. The tuned rule is then scored **once** on 2020–2024, years the search never saw, against random search with the same budget and against buy-and-hold. The project also measures how much of the training performance was overfitting.

**Rule:** own SPY when the `fast`-day average close is above the `slow`-day average, otherwise hold cash. The signal is decided at today's close and applied from tomorrow, with a 0.05% cost on every trade. The search space is fast ∈ [5, 50] and slow ∈ [20, 250] with fast < slow, which gives **10,130 pairs**.

## Results

### Final exam: test set scored once (`results/results_table.csv`)

| Strategy | Train Sharpe | Test Sharpe | Sharpe drop | Train Max DD | Test Max DD | Test CAGR | Test Calmar | $100 → |
|---|---|---|---|---|---|---|---|---|
| Buy-and-hold | 0.93 | **0.75** | 0.18 | −19.3% | −33.7% | 14.5% | 0.43 | **$196** |
| GA (37, 108) | **0.99** | 0.59 | 0.39 | −13.4% | **−29.1%** | 8.7% | 0.30 | $151 |
| Random search (38, 108) | 0.96 | 0.57 | 0.39 | −13.4% | **−29.1%** | 8.3% | 0.29 | $149 |

Sharpe ratios are annualised with a 0% risk-free rate. No strategy kept its test drawdown within the −20% limit; COVID-19 in March 2020 fell faster than any 108-day average could react.

### What the experiment shows

1. **The GA is a better optimiser than random search.** With the same 1,500 evaluations, the GA found the best training pair in the whole grid, (37, 108), in **6 of 10 seeds**. Random search found it in 1 of 10 (`1_convergence.png`). The GA also got there by testing only ~600 distinct pairs, because evolution concentrates its tries where the scores are good.
2. **Being a better optimiser did not help out of sample.** Both methods' training Sharpe fell by about 0.39 on the test years, and both ended below buy-and-hold. The extra training Sharpe the GA found was fitted noise.
3. **Training rank does not predict test rank.** Across all 10,130 pairs, the rank correlation between training and test Sharpe is **−0.27** (`3_overfitting.png`, left). Picking the training winner selects pairs from the bottom-right of the cloud, which is the definition of overfitting. The heatmaps (`4_sharpe_maps.png`) show why: the best region in training (fast ≈ 35–45, slow ≈ 100–130) is a weak region in 2020–2024, and the best test pairs moved to fast < 15.
4. **Over generations, test Sharpe does follow training Sharpe up for seed 0, from 0.50 to 0.59** (`3_overfitting.png`, right). The gap between the lines stays around 0.4 and the test line never reaches buy-and-hold. Tuning harder improved the pick a little, but did not close the gap.
5. **Luck check.** 9 of 10 GA seeds converged to the same small neighbourhood, (37–38, 106–108). Seed 47 got stuck in a local optimum at (23, 220), with training Sharpe 0.90 and test Sharpe 0.45. The GA's answer is mostly stable, but one run in ten would have given a noticeably worse rule.
6. **The drawdown protection was real but partial** (`5_drawdowns.png`). In the 2022 bear market, the GA rule's worst fall was about −18%, against −25% for buy-and-hold, because it stepped into cash. In March 2020 the crash was faster than a 108-day average could react, so the rule still fell 29%. It then sat in cash through the rebound and took until April 2021 to recover its peak, while buy-and-hold recovered by August 2020.
7. **Buy-and-hold is a hard benchmark.** Its training Sharpe of 0.93 is already higher than 99.4% of all pairs. Only 20% of pairs beat it on test, and nothing on training data tells you in advance which 20%.

**The one-line interview answer:** "The GA tuned the rule well, 0.99 training Sharpe versus 0.96 for random search, finding the global optimum 6 times out of 10. Out of sample, the Sharpe fell to 0.59 against 0.75 for buy-and-hold, because the parameter region that worked in 2010–2019 stopped working in 2020–2024. Training and test Sharpe are negatively rank-correlated across the search space. The rule did cut the drawdown from −33.7% to −29.1%, but that was a smaller cushion than the return it gave up."

### Charts

| | |
|---|---|
| ![convergence](results/1_convergence.png) | ![equity](results/2_equity_test.png) |
| ![overfitting](results/3_overfitting.png) | ![maps](results/4_sharpe_maps.png) |

![drawdowns](results/5_drawdowns.png)

## How it works

| Step | File | What it does |
|---|---|---|
| 1 | `config.py` | Every setting: dates, cost, search ranges, GA sizes, seeds |
| 2–3 | `data.py` | Downloads SPY adjusted closes from 2009 (one year of warm-up for the 250-day average), caches them to `data/SPY.csv` |
| 4–7 | `backtest.py` | Signal → shift one day → returns minus costs. Sharpe, max drawdown, CAGR, Calmar. Buy-and-hold. Full Sharpe grid |
| 8 | `ga.py` | Tournament selection (size 2), one-gene crossover, Gaussian mutation, elitism (2), repair to keep fast < slow. 50 × 30 = 1,500 evaluations |
| 9 | `random_search.py` | 1,500 random valid pairs, best kept |
| 10–11 | `main.py` | Searches on training only, freezes the picks, scores them once on test, then runs the overfitting diagnostics |
| 12 | `charts.py` | The five PNGs above |

**Keeping the test honest:** the search methods only ever see a fitness function that reads 2010–2019. Seed 0's picks are frozen before any 2020–2024 number is computed. The overfitting diagnostics (the full grid, per-generation test Sharpe, the other seeds' winners) read test data only to *describe* those frozen choices. None of them feeds back into a choice.

**Verified against the hand-worked example:** `tests/test_worked_example.py` runs the guide's 15-day example and checks every number it works out by hand. That includes the moving averages, the signal and position rows, the daily Sharpe of 0.288, the max drawdown of −2.83%, the final $105.84, the buy-and-hold figures (0.297 / −7.55%), the full 3×3 Sharpe table and the crossover (1, 5) × (3, 4) → (1, 4), (3, 5).

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py           # ~15 s, writes results/
python -m pytest tests   # worked-example checks
```

## Limitations and next steps

- One asset and one train/test split. A single 5-year test window, dominated by COVID-19 and 2022, is one draw. **Walk-forward testing** (train 3 years, test 1, roll forward) would show whether the decay is systematic.
- Trades happen at the close the signal was computed on, with no slippage, and cash earns 0% (rates were about 5% in 2023–24, which would help the cash periods).
- Only two parameters, so an exhaustive grid is feasible and the GA is a demonstration. GAs earn their keep on larger spaces, for example adding an RSI filter or a stop-loss, or **NSGA-II** to optimise Sharpe and drawdown together.
