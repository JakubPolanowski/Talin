from src.talin.indicators import volatility
import pandas as pd
import numpy as np

size = 100
close = pd.Series(np.random.rand(size) + 1)
low = pd.Series(close - np.random.rand())
high = pd.Series(close + np.random.rand())


def test_trange():
    """True Range

    TR = Max[(H - L), Abs(H - close_previous) - Abs(L - close_previous)]

    source: https://www.investopedia.com/terms/a/atr.asp

    Note, this test is implemented to be as simple as possible, not efficient
    """

    tr = [np.NaN]  # first value is nan since no close_prev before first close

    for i in range(1, size):
        highVLow = high[i] - low[i]
        highVClosep = abs(high[i] - close[i-1])
        lowVClosep = abs(low[i] - close[i-1])

        tr.append(max((highVLow, highVClosep, lowVClosep)))

    tr = pd.Series(tr)
    trange_implemented = volatility.trange(high, low, close)

    assert all(tr.dropna() == trange_implemented.dropna())


def test_atr():
    """Average True Range

    This is the N-Period (typically 14) rolling average of the true range

          ( 1 )  (n)
    ATR = ( - ) Sigma TR_i
          ( n ) (i=1)

    source: https://www.investopedia.com/terms/a/atr.asp

    Note, trange function tested with prior test above
    """

    tr = volatility.trange(high, close, close)

    # tr should be of type pd.Series, therefore use rolling method

    # test typical 14 days
    assert all(tr.rolling(14).mean().dropna() ==
               volatility.atr(high, low, close).dropna())


def test_atr_periods():
    """Average True Range

    Same as test_atr however this time testing 1-30 loopback periods
    """

    tr = volatility.trange(high, close, close)

    for i in range(1, 31):
        assert all(tr.rolling(i).mean().dropna() == volatility.atr(
            high, low, close, periods=i).dropna())


def test_natr():
    """Normalized Average True Range

    NATR = ATR(n_periods) / Close * 100

    source: https://mudrex.com/blog/normalized-average-true-range-trading-strategy/

    Note: ATR tested in tests above
    """

    atr = volatility.atr(high, low, close)
    natr = (atr / close) * 100

    assert all(natr.dropna() == volatility.natr(high, low, close).dropna())
