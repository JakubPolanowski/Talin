import numpy as np
import pandas as pd

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


def directional_indicator(dm, avgTR, loopback=14):
    """Calculates the (postive or negative) Directional indicator. Note that dm and avgTR must be numpy arrays or similar objects such as pandas series.

    Args:
        dm (numpy array): the positive or negative directional movement
        avgTR (numpy array): average true range
        loopback (int): The number of periods to use as the window. Defaults to 14.

    Returns:
        pandas series: postive or negative (depending on the directional movement input) directional indicator.
    """

    return 100 * (pd.Series(dm).ewm(alpha=1/loopback).mean() / avgTR)


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

    pDI = directional_indicator(pDM, avgTR, loopback)
    nDI = directional_indicator(nDM, avgTR, loopback)

    pDI = 100 * (pd.Series(pDM).ewm(alpha=1/loopback).mean() / avgTR)
    nDI = 100 * (pd.Series(nDM).ewm(alpha=1/loopback).mean() / avgTR)

    averageDX = ((pDI - nDI).abs() / (pDI + nDI)).ewm(alpha=1/loopback).mean()

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


def apo(close, short_period=14, long_period=30):

    close = pd.Series(close)
    return close.ewm(alpha=1/short_period) - close.ewm(alpha=1/long_period)
    pass


def aroon():
    pass


def aroon_osc():
    pass


"""

TODO IMPLEMENT

APO                  Absolute Price Oscillator
AROON                Aroon
AROONOSC             Aroon Oscillator
BOP                  Balance Of Power
CCI                  Commodity Channel Index
CMO                  Chande Momentum Oscillator
DX                   Directional Movement Index
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
