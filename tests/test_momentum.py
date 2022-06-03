import numpy as np
import pandas as pd
from src.talin.indicators import momentum
# note testing handle by test_volatility
from src.talin.indicators import volatility

size = 100
close = pd.Series(np.random.rand(size) + 1)
low = pd.Series(close - np.random.rand())
high = pd.Series(close + np.random.rand())


"""
TODO Implement

"plus_dm", "minus_dm", "di", "dx", "adx", "adxr",
"apo", "aroon", "aroon_osc", "bop", "cci", "cmo",
"macd", "mfi", "mom", "ppo", "roc", "rocp", "rocr",
"rocr100", "rsi", "stochf", "stoch", "stoch_rsi",
"trix", "ultosc", "willr",
"""


def test_plus_dm():
    """+Direction Indicator

    +DM = Current High - Previous High

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    pDM = high - high.shift(1)
    assert all(pDM == momentum.plus_dm(high))


def test_minus_dm():
    """-Direction Indicator

    -DM = Previous Low - Current Low

    source: https://www.investopedia.com/terms/a/adx.asp
    """
    nDM = low.shift(1) - low
    assert all(nDM == momentum.minus_dm(low))


def test_di_minus():
    pass


def test_di_plus():
    pass


def test_dx():
    pass


def test_adx():
    pass


def test_adxr():
    pass
