import numpy as np
import pandas as pd
from src.talin.indicators import momentum
# note testing handle by test_volatility
from src.talin.indicators import volatility

size = 100
close = pd.Series(np.random.rand(size) + 1)
low = pd.Series(close - np.random.rand())
high = pd.Series(close + np.random.rand())


def test_plus_dm():
    """+Direction Movement

    +DM = Current High - Previous High

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    pDM = high - high.shift(1)
    assert all(pDM.dropna() == momentum.plus_dm(high).dropna())


def test_minus_dm():
    """-Direction Movement

    -DM = Previous Low - Current Low

    source: https://www.investopedia.com/terms/a/adx.asp
    """
    nDM = low.shift(1) - low
    assert all(nDM.dropna() == momentum.minus_dm(low).dropna())


def test_di_minus():
    """-Direction Indicator

             (Smoothed -DM)
    -DI =  ------------------  * 100
          (Average True Range)

    Smoothing window same between Average True Range (ATR) and Smoothed -DM,
    typically 14 days

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    nDM = momentum.minus_dm(low)

    for i in range(1, 21):
        atr = volatility.atr(high, low, close, periods=i)
        nDI = nDM.rolling(i).mean() / atr * 100
        assert all(nDI.dropna() == momentum.di(nDM, atr, periods=i).dropna())


def test_di_plus():
    """+Direction Indicator

             (Smoothed +DM)
    +DI =  ------------------  * 100
          (Average True Range)

    Smoothing window same between Average True Range (ATR) and Smoothed +DM,
    typically 14 days

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    pDM = momentum.plus_dm(high)

    for i in range(1, 21):
        atr = volatility.atr(high, low, close, periods=i)
        pDI = pDM.rolling(i).mean() / atr * 100
        assert all(pDI.dropna() == momentum.di(pDM, atr, periods=i).dropna())


def test_dx():
    """Directional Index

          (|+DI - -DI|)
    DX =   -----------  * 100
          (|+DI + -DI|)

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    pDM = momentum.plus_dm(high)
    nDM = momentum.minus_dm(low)
    atr = volatility.atr(high, low, close, periods=14)

    pDI = momentum.di(pDM, atr, periods=14)
    nDI = momentum.di(nDM, atr, periods=14)

    dx = ((pDI - nDI).abs() / (pDI + nDI).abs()) * 100
    assert all(dx.dropna() == momentum.dx(pDI, nDI).dropna())


def test_adx():
    """Average Directional Movement Index

    ADX = N-period Simple Moving Average of DX

    source: https://www.investopedia.com/terms/a/adx.asp
    """

    pDM = momentum.plus_dm(high)
    nDM = momentum.minus_dm(low)
    atr = volatility.atr(high, low, close, periods=14)

    pDI = momentum.di(pDM, atr, periods=14)
    nDI = momentum.di(nDM, atr, periods=14)

    dx = momentum.dx(pDI, nDI)

    for i in range(1, 21):
        adx = dx.rolling(i).mean()
        assert all(adx.dropna() == momentum.adx(
            high, low, close, periods=i).dropna())


def test_adxr():
    """Average Directional Movement Rating

    ADXR = (ADX - ADX N Periods Ago) / 2

    Typically 14 period lookback

    Source: https://www.marketvolume.com/technicalanalysis/adxr.asp
    """

    adx = momentum.adx(high, low, close, periods=14)

    for i in range(1, 21):
        adxr = (adx - adx.shift(i)) / 2
        assert all(adxr.dropna() == momentum.adxr(adx, periods=i).dropna())


def test_apo():
    """Absolute Price Oscillator

    APO = Shorter Period EMA - Longer Period EMA

    Source: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/apo
    """

    for i in range(1, 21):
        apo = close.ewm(span=i, adjust=False).mean() - \
            close.ewm(span=i+5, adjust=False).mean()

        assert all(apo.dropna() == momentum.apo(
            close, short_periods=i, long_periods=i+5).dropna())


def test_aroon():
    pass


def test_aroon_osc():
    pass


def test_bop():
    pass


def test_cci():
    pass


def test_cmo():
    pass


"""
TODO Implement

"apo", "aroon", "aroon_osc", "bop", "cci", "cmo",
"macd", "mfi", "mom", "ppo", "roc", "rocp", "rocr",
"rocr100", "rsi", "stochf", "stoch", "stoch_rsi",
"trix", "ultosc", "willr",
"""
