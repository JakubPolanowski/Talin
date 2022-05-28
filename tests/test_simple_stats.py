from src.talin.stats import simple_stats
import numpy as np


def test_mean():
    array = np.random.rand(10)
    assert simple_stats.average(array) == np.mean(array)


def test_median():
    array = np.random.rand(10)
    assert simple_stats.median(array) == np.median(array)


def test_std():
    array = np.random.rand(10)
    assert simple_stats.std(array) == np.std(array, ddof=1)


def test_std_ddof():
    array = np.random.rand(10)
    for ddof in range(6):
        assert simple_stats.std(array, ddof=ddof) == np.std(array, ddof=ddof)


def test_typical_price():
    assert simple_stats.typical_price(10, 5, 3) == 6


def test_weighted_close():
    assert simple_stats.weighted_close(10, 5, 3) == 5.25
