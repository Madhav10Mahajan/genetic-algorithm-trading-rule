"""
STEP 9: Random search - the control experiment for the GA.

Draw N_RANDOM_TRIES (= POP_SIZE x N_GENERATIONS = 1,500) random valid pairs, score each
on the training data, keep the best. Same budget as the GA, no learning between tries.
If random guessing matches the GA, the evolution is not adding anything.
"""
import numpy as np

from config import FAST_RANGE, N_RANDOM_TRIES, POP_SIZE, SLOW_RANGE
from ga import SearchResult, random_pair


def run_random_search(fitness, seed, n_tries=N_RANDOM_TRIES, batch=POP_SIZE,
                      fast_range=FAST_RANGE, slow_range=SLOW_RANGE):
    """`fitness(pair) -> float` must only look at training data.

    history records the best-so-far after every `batch` tries, so it lines up with GA generations.
    """
    rng = np.random.default_rng(seed)
    result = SearchResult(best_pair=None, best_fitness=-np.inf)
    for i in range(1, n_tries + 1):
        pair = random_pair(rng, fast_range, slow_range)
        result.n_evaluations += 1
        if pair not in result.evaluated:
            result.evaluated[pair] = fitness(pair)
        if result.evaluated[pair] > result.best_fitness:
            result.best_pair, result.best_fitness = pair, result.evaluated[pair]
        if i % batch == 0:
            result.history.append({"generation": i // batch, "best_pair": result.best_pair,
                                   "best_fitness": result.best_fitness})
    return result
