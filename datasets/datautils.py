"""
LF-Font
Copyright (c) 2020-present NAVER Corp.
MIT license
"""
import numpy as np


def cyclize(loader):
    """ Cyclize loader """
    while True:
        for x in loader:
            yield x


def uniform_indice(end, n_sample, duplicate=False, st=None):
    """ Sample from [0, end) with (almost) equidistant interval """
    if end <= 0:
        return np.empty(0, dtype=np.int)

    if not duplicate and n_sample > end:
        n_sample = end

    # NOTE with endpoint=False, np.linspace does not sample the `end` value
    indice = np.linspace(0, end, num=n_sample, dtype=np.int, endpoint=False)
    if st is None and end:
        st = (end-1 - indice[-1]) // 2
    return indice + st


def sample(population, n_sample, exception=None, seed=None):
    """ sampling without replacement N elements from set with exception

    Params:
        population: [1d] list or set or np.ndarray
    Return: np.ndarray
    """
    np.random.seed(seed)
    if exception is not None:
        population = set(population) - set(exception)
    if not isinstance(population, np.ndarray):
        population = np.asarray(list(population))

    replace = len(population) < n_sample
    ids = np.random.choice(len(population), size=n_sample, replace=replace)

    return population[ids]


def uniform_sample(population, n_sample, st=None):
    assert not isinstance(population, set), "population should have order"

    N = len(population)
    if n_sample is None:
        return population

    indice = uniform_indice(N, n_sample, st)

    if isinstance(population, np.ndarray):
        return population[indice]
    elif isinstance(population, list):
        return [population[idx] for idx in indice]
    elif isinstance(population, str):
        return ''.join([population[idx] for idx in indice])
    else:
        raise TypeError(type(population))
