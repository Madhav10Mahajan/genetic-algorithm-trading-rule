"""
STEP 8: Genetic algorithm - evolve (fast, slow) pairs on the TRAINING data only.

Vocabulary
  Individual  one pair of numbers, e.g. (20, 100)
  Population  POP_SIZE such pairs
  Fitness     the pair's annualised Sharpe on 2010-2019
  Generation  one round of: score -> select -> crossover -> mutate -> keep the best

One generation
  1. Score every pair (fitness).
  2. Elitism:   copy the ELITE best pairs straight into the next generation, so the best
                solution found so far is never lost.
  3. Selection: tournament - pick TOURNAMENT_SIZE pairs at random, the fittest becomes a parent.
  4. Crossover: child 1 = fast from parent 1 + slow from parent 2, child 2 = the other way round.
                This is how good parts of different solutions get combined.
  5. Mutation:  with probability MUTATION_RATE nudge each number by a random step, so the
                search keeps exploring new areas instead of only recombining what it has.
  6. Repair:    keep numbers inside their ranges and fast < slow (swap them if not).
  Repeat until N_GENERATIONS have been scored: POP_SIZE x N_GENERATIONS fitness evaluations.

Pros
  - Needs only a score, no gradients - works on a jagged, noisy Sharpe surface.
  - Concentrates its tries in promising regions instead of spreading them evenly.
Cons
  - Random: different seeds can end on different pairs, so it must be run several times.
  - It maximises training Sharpe as hard as it can, which is exactly what invites overfitting.
"""
from dataclasses import dataclass, field

import numpy as np

from config import (ELITE, FAST_RANGE, MUTATION_RATE, MUTATION_STEP, N_GENERATIONS, POP_SIZE,
                    SLOW_RANGE, TOURNAMENT_SIZE)


@dataclass
class SearchResult:
    best_pair: tuple
    best_fitness: float
    history: list = field(default_factory=list)      # one dict per generation (GA) / per POP_SIZE tries (random)
    evaluated: dict = field(default_factory=dict)    # every distinct pair tried -> its fitness
    n_evaluations: int = 0


def random_pair(rng, fast_range=FAST_RANGE, slow_range=SLOW_RANGE):
    while True:
        fast = int(rng.integers(fast_range[0], fast_range[1] + 1))
        slow = int(rng.integers(slow_range[0], slow_range[1] + 1))
        if fast < slow:
            return fast, slow


def repair(pair, fast_range=FAST_RANGE, slow_range=SLOW_RANGE):
    fast = int(np.clip(pair[0], *fast_range))
    slow = int(np.clip(pair[1], *slow_range))
    if fast > slow:
        fast, slow = slow, fast
    if fast == slow:   # move whichever number has room so that fast < slow
        if slow < slow_range[1]:
            slow += 1
        else:
            fast -= 1
    return fast, slow


def tournament(population, scores, rng, k=TOURNAMENT_SIZE):
    picks = rng.choice(len(population), size=k, replace=False)
    return population[max(picks, key=lambda i: scores[i])]


def crossover(parent1, parent2):
    return (parent1[0], parent2[1]), (parent2[0], parent1[1])


def mutate(pair, rng, rate=MUTATION_RATE, step=MUTATION_STEP, fast_range=FAST_RANGE, slow_range=SLOW_RANGE):
    genes = list(pair)
    for i, (lo, hi) in enumerate((fast_range, slow_range)):
        if rng.random() < rate:
            delta = int(round(rng.normal(0, step * (hi - lo))))
            genes[i] += delta if delta != 0 else int(rng.choice([-1, 1]))   # a mutation always changes the number
    return repair(genes, fast_range, slow_range)


def run_ga(fitness, seed, pop_size=POP_SIZE, n_generations=N_GENERATIONS, elite=ELITE,
           fast_range=FAST_RANGE, slow_range=SLOW_RANGE, **mutation_kwargs):
    """`fitness(pair) -> float` must only look at training data."""
    rng = np.random.default_rng(seed)
    result = SearchResult(best_pair=None, best_fitness=-np.inf)

    def score(pair):
        result.n_evaluations += 1
        if pair not in result.evaluated:            # cache: the same pair always gets the same Sharpe
            result.evaluated[pair] = fitness(pair)
        return result.evaluated[pair]

    population = [random_pair(rng, fast_range, slow_range) for _ in range(pop_size)]
    for gen in range(1, n_generations + 1):
        scores = [score(p) for p in population]
        order = np.argsort(scores)[::-1]
        best = population[order[0]]
        if scores[order[0]] > result.best_fitness:
            result.best_pair, result.best_fitness = best, scores[order[0]]
        result.history.append({"generation": gen, "best_pair": best, "best_fitness": scores[order[0]],
                               "mean_fitness": float(np.mean(scores))})
        if gen == n_generations:
            break

        children = [population[i] for i in order[:elite]]
        while len(children) < pop_size:
            p1 = tournament(population, scores, rng)
            p2 = tournament(population, scores, rng)
            for child in crossover(p1, p2):
                if len(children) < pop_size:
                    children.append(mutate(child, rng, fast_range=fast_range, slow_range=slow_range,
                                           **mutation_kwargs))
        population = children
    return result
