from typing import Tuple, List

import numpy as np
import pandas as pd
import scipy.stats

"""
See this paper.
Data mining meets performance evaluation: Fast algorithms for modeling bursty traffic
"""


def __truncate(s: np.array) -> np.array:
    """
    Truncate the series so that its length is a power of 2.
    :param s: The series to truncate.
    :return: The truncated series.
    """
    length: int = len(s)
    length: int = int(np.exp2(np.floor(np.log2(length))))
    return s[:length]


def __compute_entropy(series: np.array) -> float:
    """
    Compute the entropy of a series.
    :param series: The series to compute entropy.
    :return: The entropy.
    """
    pd_series = pd.Series(series)
    counts = pd_series.value_counts()
    return scipy.stats.entropy(counts)


def estimate(series: np.array) -> Tuple[float, float]:
    """
    Estimate the bias b for the input series.
    :param series: The series to estimate.
    :return: The bias b (which is also the slope of the linear fit of
    entropy plot) and the y-intercept.
    """
    entropy_list = []
    for i in np.arange(int(np.log2(len(series))) + 1):
        bucket_size: int = int(np.exp2(i))
        aggregated = np.add.reduceat(series,
                                     np.arange(0, len(series), bucket_size))
        entropy = __compute_entropy(aggregated)
        entropy_list.append(entropy)
    entropy_list = np.flip(entropy_list)
    x = np.arange(0, len(entropy_list))
    [bias_b, y_intercept] = np.polyfit(x, entropy_list, 1)
    return bias_b, y_intercept


def generate(b: float, n: int, N: int) -> List[int]:
    """
    Generate series using b-model.
    :param b: Bias b.
    :param n: Aggregation level.
    :param N: Total volume.
    :return: The generated series.
    """
    stack: List[Tuple[int, int]] = [(0, N)]
    series: List[int] = []
    while len(stack) != 0:
        k, v = stack.pop()
        if k == n:
            series.append(v)
        else:
            r: int = np.random.randint(0, 2)
            v1: int = int(v * b)
            v2: int = v - v1
            if r == 0:
                stack.append((k + 1, v1))
                stack.append((k + 1, v2))
            else:
                stack.append((k + 1, v2))
                stack.append((k + 1, v1))
    return series

