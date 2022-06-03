from src.talin.stats import simple_stats
import numpy as np
import pandas as pd

series = pd.Series(np.random.rand(100))


def test_mean():
    assert simple_stats.average(series) == series.mean()


def test_median():
    assert simple_stats.median(series) == series.median()


def test_var():
    assert simple_stats.var(series) == series.var()  # note default is ddof=1


def test_var_ddof():
    for ddof in range(6):
        assert simple_stats.var(series, ddof=ddof) == series.var(ddof=ddof)


def test_std():
    assert simple_stats.std(series) == series.std()  # note default is ddof=1


def test_std_ddof():
    for ddof in range(6):
        assert simple_stats.std(series, ddof=ddof) == series.std(ddof=ddof)


def test_typical_price():
    assert simple_stats.typical_price(10, 5, 3).values[0] == 6


def test_weighted_close():
    assert simple_stats.weighted_close(10, 5, 3).values[0] == 5.25
