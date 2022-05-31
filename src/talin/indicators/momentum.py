import numpy as np
import pandas as pd
from src.talin.stats import simple_stats

__all__ = ["tr", "attr", "adx", "adxr"]


def tr(high, low, prev_close):
    """Calculates the True Range given the highs, lows and the previous closes (period-1). Note that inputs must be given as numpy arrays or similar objects such as pandas series. 

    Args:
        high (numpy array): array of highs
        low (numpy array): array of lows
        prev_close (numpy array): array of previous closes (period-1)

    Returns:
        numpy array: array of the True range for each period
    """
    return np.r_[high-low, high-prev_close, prev_close-low].T.max(axis=1)


def atr(high, low, prev_close, loopback=14):
    """Calculates the Average True Range given the highs, lows and the previous closes (period-1) with a set loopback. Note that the input arrays (high, low and prev_close) must be given as numpy arrays or similar objects such as pandas series.

    Args:
        high (numpy array): array of highs
        low (numpy array): array of lows
        prev_close (numpy array): array of pervious closes (period-1)
        loopback (int, optional): The number of periods to use as the window for the simple moving average. Defaults to 14.

    Returns:
        numpy array: The Average True Range for each period
    """

    return pd.Series(tr(high, low, prev_close)).rolling(loopback).mean().values


def positive_DM(high):
    """Calculates the Positive Directional Movement. Note that the highs must be a list of numbers or similar object (numpy array, pandas series, etc.)

    Args:
        high (Numerical List): list of highs

    Returns:
        numpy array: positive directional movement
    """

    pDM = np.diff(high)
    pDM[pDM < 0] = 0

    return pDM


def negative_DM(low):
    """Calculates the Negative Directional Movement. Note that the lows must be a list of numbers or similar object (numpy array, pandas series, etc.)

    Args:
        low (Numerical List): list of lows

    Returns:
        numpy array: negative directional movement
    """

    nDM = np.diff(low)
    nDM[nDM > 0] = 0

    return nDM


def di(dm, avgTR, loopback=14):
    """Calculates the (postive or negative) Directional indicator. Note that dm and avgTR must be numpy arrays or similar objects such as pandas series.

    Args:
        dm (numpy array): the positive or negative directional movement
        avgTR (numpy array): average true range
        loopback (int): The number of periods to use as the window. Defaults to 14.

    Returns:
        pandas series: postive or negative (depending on the directional movement input) directional indicator.
    """

    return 100 * (pd.Series(dm).ewm(alpha=1/loopback).mean() / avgTR)


def dx(pDI, nDI):
    """Calculates the Directional Movement Index given the postive and negative Directional Indicators. Note that these must be given numerical lists that can be converted to pandas Series or are already pandas Series.

    Args:
        pDI (Numerical List): positive Directional Indicator
        nDI (Numierlca List): negative Directional Indicator

    Returns:
        pandas Series: Directional Movement Index
    """

    pDI, nDI = (pd.Series(pDI), pd.Series(nDI))
    return (pDI - nDI).abs() / (pDI + nDI).abs() * 100


def adx(high, low, close, loopback=14):
    """Calculates the Average Directional Index given the highs, lows, closes, and the loopback (window). Note that the input arrays (high, low, and close) must be guven as numpy arrays or similar objects such as pandas series. 

    Args:
        high (numpy array): array of highs
        low (numpy array): array of lows
        close (numpy array): array of closes 
        loopback (int, optional): The number of periods to use as the window. Defaults to 14.

    Returns:
        (pandas Series, pandas Series, pandas Series): Returns a tuple of the positive directional indicator, the negative direction indicator, and the average directional index
    """

    pDM = positive_DM(high)
    nDM = negative_DM(low)

    avgTR = atr(high, low, np.r_[np.NaN, close[:-1]], loopback)

    pDI = di(pDM, avgTR, loopback)
    nDI = di(nDM, avgTR, loopback)

    pDI = 100 * (pd.Series(pDM).ewm(alpha=1/loopback).mean() / avgTR)
    nDI = 100 * (pd.Series(nDM).ewm(alpha=1/loopback).mean() / avgTR)

    averageDX = dx(pDI, nDI).ewm(alpha=1/loopback).mean()

    return pDI, nDI, averageDX


def adxr(adx, lookback=2):
    """Calculates the Average Direction Movement Rating given the ADX and the number of periods to look backwards. ADX input must be given as a numpy array or similar object like pandas series. 

    Args:
        adx (numpy array): array/series of Average Directional Movement Indexes
        lookback (int): Number of periods to look backwards. Defaults to 2.

    Returns:
        pandas Series: The Average Direction Movement Rating
    """

    return (adx - pd.Series(adx).shift(lookback)) / 2


def apo(close, short_periods=14, long_periods=30):
    """Calculates Absolute Price Oscilliator given the a list of closing values and the number of periods for the short term and long term

    Args:
        close (Numerical List): List of closing values
        short_periods (int, optional): Number of periods for the short term. Defaults to 14.
        long_periods (int, optional): Number of periods for the long term. Defaults to 30.

    Returns:
        pandas Series: Absolute Price Oscillator series
    """

    close = pd.Series(close)
    return close.ewm(alpha=1/short_periods) - close.ewm(alpha=1/long_periods)


def aroon(high, low, periods=25):
    """Calculates the AROON indicator given the lists of highs and lows and the number of periods to use in the calculation.

    Args:
        high (Numerical List): List of highs
        low (Numerical List): List of lows
        periods (int, optional): Number of periods. Defaults to 25.

    Returns:
        (pandas Series, pandas Series): Returns the AROON up and AROON down series.
    """

    high, low = (pd.Series(high), pd.Series(low))

    aroon_up = 100 * \
        high.rolling(periods + 1).apply(lambda x: x.argmax()) / periods
    aroon_down = 100 * \
        low.rolling(periods + 1).apply(lambda x: x.argmin()) / periods

    return aroon_up, aroon_down


def aroon_osc(high, low, periods=25):
    """Calculates the AROON Oscillator given the lists of high and lows and the number of periods to use in the calculation.

    Args:
        high (Numerical List): List of highs
        low (Numerical List): List of lows
        periods (int, optional): Number of periods. Defaults to 25.
    Returns:
        pandas Series: AROON Oscillator series
    """

    aroon_up, aroon_down = aroon(high, low, periods)
    return aroon_up - aroon_down


def bop(high, low, open, close):
    """Calculates the Balance of Power indicator, given the input arrays of high, low, open, and close. Note that these input arrays must be numpy arrays or similar objects such as pandas Series

    Args:
        high (numpy array): array of highs
        low (numpy array): array of lows
        open (numpy array): array of opens
        close (numpy array): array of closes

    Returns:
        numpy array/same as input type: Balance of Power
    """

    return (close - open) / (high - low)


def cci(high, low, close, periods=20):
    """Calculates the Commodity Channel Index, given the input arrays of high, low, open and close. Note that these input arrays must be numpy arrays or similar objects such as pandas Series.

    Args:
        high (numpy array): array of highs
        low (numpy array): array of lows
        close (numpy array): array of closes
        periods (int, optional): Number of periods for the moving average. Defaults to 20.

    Returns:
        pandas Series: Commodity Channel Index series.
    """

    typicalP = pd.Series(simple_stats.typical_price(high, low, close))
    movingAvg = typicalP.rolling(periods).mean()
    meanDeviation = (
        typicalP - movingAvg).abs().rolling(periods).mean() / periods

    return (typicalP - movingAvg) / (0.015 * meanDeviation)


def cmo(close, periods=9):
    """Calculates the Chande Momentum Oscillator given the input list of closes and the number of periods to look back.

    Args:
        close (Numerical List): List of closes
        periods (int, optional): Number of periods to look back. Defaults to 9.

    Returns:
        pandas Series: Chande Momenum Oscillator series
    """

    close = pd.Series(close).diff()

    up_down = close.diff().apply(lambda x: 1 if x > 0 else 0 if x < 0 else x)
    nHigher = up_down.roling(periods).sum()
    nLower = (1-up_down).rolling(periods).sum()

    return (nHigher - nLower) / (nHigher + nLower) * 100


def macd():
    pass


def macd_ext():
    pass


def macd_fix():
    pass


def mfi():
    pass


"""

TODO IMPLEMENT

MACD                 Moving Average Convergence/Divergence
MACDEXT              MACD with controllable MA type
MACDFIX              Moving Average Convergence/Divergence Fix 12/26
MFI                  Money Flow Index
MOM                  Momentum
PPO                  Percentage Price Oscillator
ROC                  Rate of change : ((price/prevPrice)-1)*100
ROCP                 Rate of change Percentage: (price-prevPrice)/prevPrice
ROCR                 Rate of change ratio: (price/prevPrice)
ROCR100              Rate of change ratio 100 scale: (price/prevPrice)*100
RSI                  Relative Strength Index
STOCH                Stochastic
STOCHF               Stochastic Fast
STOCHRSI             Stochastic Relative Strength Index
TRIX                 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
ULTOSC               Ultimate Oscillator
WILLR                Williams' %R

"""
