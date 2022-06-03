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
    """Typical price is a simple ratio

          (CurrentHigh + CurrentLow + CurrentClose)
    TP =   ---------------------------------------
                            3

    Source: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/typical-price
    """

    close = pd.Series(np.random.rand(100) + 1)
    low = pd.Series(close - np.random.rand())
    high = pd.Series(close + np.random.rand())

    tp = (high + low + close) / 3

    assert all(tp == simple_stats.typical_price(high, low, close))


def test_weighted_close():
    """Weighted close is another simple ratio

                    (High + Low + Close*2)
    Weight_Close =   --------------------
                              4

    Source: https://www.incrediblecharts.com/indicators/weighted_close.php
    """

    close = pd.Series(np.random.rand(100) + 1)
    low = pd.Series(close - np.random.rand())
    high = pd.Series(close + np.random.rand())

    wc = (high + low + close*2) / 4

    assert all(wc == simple_stats.weighted_close(high, low, close))
